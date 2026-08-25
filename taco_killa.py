"""AlphaZero-style Othello bot: dual-head CNN + MCTS + self-play training.

The network takes an 8x8x2 canonical board (my discs / opponent discs) and
produces two things at once:

  * policy -- 64 numbers, a prior over squares, used to decide which branch
    of the tree is worth exploring first;
  * value  -- one number in [-1, 1], "am I winning from here", which is
    what MCTS backs up the tree instead of playing random rollouts to the
    end of the game.

Self-play speed
---------------
Search is dominated by how many network calls it makes and what each one
costs, so both are attacked here:

  * cost per call -- `predict` below calls the model directly through a
    compiled tf.function instead of `model.predict()`, whose per-call
    setup cost is designed to be amortized over huge datasets, not paid
    3,000 times a game on a single board;
  * number of calls -- `run_self_play_batch` plays many games in lockstep
    and evaluates one leaf from each of them in a single batched call.
    Simulations *within* one search are strictly sequential (each one reads
    the statistics the previous one wrote), so the only positions that can
    legally be evaluated together come from different games.

The search also caches the position in each tree node, so descending to an
already-visited node costs a pointer hop instead of replaying the whole
line of moves from the root.
"""

import math
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from othello import OthelloGame

MODEL_PATH = "final_model.keras"

BOARD_SIZE = 8
NUM_SQUARES = BOARD_SIZE * BOARD_SIZE

# Exploration weight in the PUCT formula. Higher trusts the policy prior
# and explores more; lower leans harder on values already measured.
C_PUCT = 1.0


def move_to_index(row, col):
    return row * BOARD_SIZE + col


def board_planes(board, current_player):
    """One position as an (8, 8, 2) float32 tensor, from `current_player`'s
    point of view: plane 0 is their discs, plane 1 is the opponent's.

    Multiplying by current_player is what makes the view canonical -- the
    network only ever has to learn "how good is this for the side to move",
    not a separate theory for black and for white.
    """
    canonical = board * current_player
    my_stones = (canonical == 1).astype(np.float32)
    opp_stones = (canonical == -1).astype(np.float32)
    return np.stack([my_stones, opp_stones], axis=-1)


def board_to_tensor(board, current_player):
    """board_planes with a batch dimension bolted on, shape (1, 8, 8, 2)."""
    return np.expand_dims(board_planes(board, current_player), axis=0)


# ----------------------------------------------------------------------
# The network.
# ----------------------------------------------------------------------

# If a previously trained model exists, keep training it instead of starting
# over from random weights -- otherwise build and compile a fresh network.
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"loaded existing model from {MODEL_PATH}, continuing training")
else:
    board_input = layers.Input(shape=(8, 8, 2))

    first_layer = layers.Conv2D(activation="relu", filters=64, kernel_size=3, padding="same")(board_input)
    second_layer = layers.Conv2D(activation="relu", filters=64, kernel_size=3, padding="same")(first_layer)
    third_layer = layers.Conv2D(activation="relu", filters=64, kernel_size=3, padding="same")(second_layer)

    flattened = layers.Flatten()(third_layer)

    policy = layers.Dense(64, activation="softmax", name="policy")(flattened)
    value = layers.Dense(1, activation="tanh", name="value")(flattened)

    model = models.Model(inputs=board_input, outputs=[policy, value])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  metrics={
                      "policy": "accuracy",
                      "value": "mae",
                  },
                  loss={
                      "policy": tf.keras.losses.CategoricalCrossentropy(),
                      "value": tf.keras.losses.MeanSquaredError()
                  }

    )
    print(f"no existing model at {MODEL_PATH}, starting fresh")


# One compiled forward pass, reused for every search.
#
# model.predict() is Keras' bulk-inference API: it spins up a tf.data
# pipeline and runs the callback machinery on every call, which costs tens
# of milliseconds no matter how small the input. MCTS calls it once per
# simulation, so that fixed cost was ~99% of self-play time. Calling the
# model directly skips all of it.
#
# The tf.function is built lazily and rebuilt if the model object is
# swapped out (server.py replaces taco_killa.model with a checkpoint),
# because a traced graph bakes in the weights it saw at trace time and
# would otherwise keep serving the old network. The leading None in the
# signature lets one trace serve any batch size, which is what makes the
# batched self-play driver below possible without a second trace.
_PREDICT_SIGNATURE = [tf.TensorSpec(shape=(None, 8, 8, 2), dtype=tf.float32)]
_compiled_model = None
_compiled_fn = None


def predict(board_tensor, net=None):
    """Run a batch of boards through the network, returning (policy, value)
    as numpy arrays of shape (N, 64) and (N, 1).

    Returns numpy rather than tf.Tensors so the search tree keeps doing
    plain Python arithmetic -- building tiny tensor ops per node would be
    slower than what this replaced.
    """
    global _compiled_model, _compiled_fn

    net = model if net is None else net
    if net is not _compiled_model:
        _compiled_fn = tf.function(
            lambda x: net(x, training=False),
            input_signature=_PREDICT_SIGNATURE,
        )
        _compiled_model = net

    policy_output, value_output = _compiled_fn(board_tensor)
    return policy_output.numpy(), value_output.numpy()


# ----------------------------------------------------------------------
# The search tree.
# ----------------------------------------------------------------------

class Node:
    """One position in the search tree.

    `player` is whose turn it is at this node, read from the actual game
    state rather than assumed to alternate -- Othello has forced passes, so
    a child can have the *same* player as its parent. Every sign flip in
    the search is driven by comparing players, never by depth parity.

    `game` is the position itself, filled in the first time the search
    steps into this node and kept from then on. That is what turns the
    descent from "copy the root and replay every move" into a pointer walk.
    """

    # __slots__ keeps these nodes small and their attribute lookups fast;
    # a single game builds tens of thousands of them.
    __slots__ = ("visit_count", "prior", "value_sum", "children",
                 "player", "parent", "game", "expanded")

    def __init__(self, player, prior, parent):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0.0
        self.children = {}
        self.player = player
        self.parent = parent
        self.game = None
        self.expanded = False

    def mean_value(self):
        """Average backed-up value, from this node's own player's view."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def best_child(self):
        """The child with the highest PUCT score, as (move, score).

        score = Q + c_puct * prior * sqrt(parent visits) / (1 + child visits)

        Q is the exploitation term -- what we have actually measured -- and
        has to be converted into *this* node's point of view before use,
        which is a negation only when the child's player differs. The right
        hand term is exploration: it starts out proportional to the policy
        prior and decays as the child accumulates visits, so the prior
        decides what gets tried first and measured values take over later.
        """
        parent_player = self.player
        sqrt_total = math.sqrt(self.visit_count)
        best_score = None
        best_move = None
        for move, child in self.children.items():
            q = child.mean_value()
            if child.player != parent_player:
                q = -q
            score = q + C_PUCT * child.prior * sqrt_total / (1 + child.visit_count)
            if best_score is None or score > best_score:
                best_score = score
                best_move = move
        return best_move, best_score

    def select_final_move(self):
        """The move actually played: the most-visited child.

        Visit count rather than value, because a child with one lucky visit
        can have a great average; visits are what the search spent its
        budget on and are the robust choice.
        """
        final_move = None
        final_move_count = -1
        for move, child in self.children.items():
            if child.visit_count > final_move_count:
                final_move = move
                final_move_count = child.visit_count
        return final_move

    def total_visits(self):
        return sum(child.visit_count for child in self.children.values())

    def visit_distribution(self):
        """Visit counts as a probability vector over all 64 squares --
        the policy target this position contributes to training."""
        distribution = np.zeros(NUM_SQUARES, dtype=np.float32)
        total = self.total_visits()
        if total == 0:
            return distribution
        for move, child in self.children.items():
            distribution[move_to_index(*move)] = child.visit_count / total
        return distribution


def descend(root):
    """Walk from `root` down to a leaf, following best_child.

    A leaf is a node that has never been expanded. Each step materializes
    the child's position once, the first time we go through it, so later
    simulations that reuse the same path pay nothing.
    """
    node = root
    while node.expanded and node.children:
        move, _ = node.best_child()
        child = node.children[move]
        if child.game is None:
            child.game = node.game.copy()
            child.game.play(*move)
            # Read the real player rather than trusting alternation: after
            # a forced pass the mover goes again.
            child.player = child.game.current_player
        node = child
    return node


def backup(leaf, value):
    """Propagate `value` (from `leaf`'s player's point of view) to the root.

    Iterative rather than recursive: this runs once per simulation per
    level, and Python function calls are not free. The sign flips only when
    the player actually changes between a node and its parent, which is
    what makes forced passes come out right.
    """
    node = leaf
    running = value
    while node is not None:
        node.visit_count += 1
        node.value_sum += running
        parent = node.parent
        if parent is not None and parent.player != node.player:
            running = -running
        node = parent


def terminal_value(node):
    """The true game result at a finished position, from its player's view."""
    winner = node.game.winner()
    if winner == 0:
        return 0.0
    return 1.0 if winner == node.player else -1.0


def expand(node, policy):
    """Give `node` one child per legal move, seeded with its policy prior."""
    node.expanded = True
    for move in node.game.legal_moves():
        prior = float(policy[move_to_index(*move)])
        node.children[move] = Node(-node.player, prior, node)


def select_leaf(root):
    """One simulation, up to the point where the network is needed.

    Returns the leaf awaiting evaluation, or None if the simulation already
    finished on its own -- which happens at a game-over position, where the
    true result is known and asking the network would be both wasted time
    and worse information.
    """
    leaf = descend(root)
    if leaf.game.game_over:
        backup(leaf, terminal_value(leaf))
        return None
    return leaf


def evaluate_leaves(leaves):
    """Network-evaluate a list of leaves in one batched call.

    This is the whole point of the lockstep driver: one call for N leaves
    costs barely more than one call for a single leaf, because most of the
    cost is per-call, not per-board.
    """
    if not leaves:
        return
    batch = np.stack([board_planes(leaf.game.board, leaf.player) for leaf in leaves])
    policies, values = predict(batch)
    for leaf, policy, value in zip(leaves, policies, values):
        expand(leaf, policy)
        backup(leaf, float(value[0]))


def new_root(game):
    """A fresh, unexpanded root node holding a copy of `game`."""
    root = Node(game.current_player, 0.0, None)
    root.game = game.copy()
    return root


def run_mcts(game, num_simulations):
    """Search one position and return the root. Single-game convenience
    wrapper -- the browser opponent in server.py uses this.

    Self-play should use run_self_play_batch instead, which shares one
    network call across many games rather than making a batch of one.
    """
    root = new_root(game)
    for _ in range(num_simulations):
        leaf = select_leaf(root)
        if leaf is not None:
            evaluate_leaves([leaf])
    return root


# ----------------------------------------------------------------------
# Self-play, many games at once.
# ----------------------------------------------------------------------

class SelfPlayGame:
    """One self-play game, driven one simulation at a time from outside.

    Ordinary MCTS runs its whole search in a loop it controls. That cannot
    be batched, because the loop blocks on the network in the middle. So
    the loop is inverted: this object hands out one leaf at a time and
    waits to be given the answer, which lets a driver collect leaves from
    many games and evaluate them together.
    """

    def __init__(self, simulations_per_move):
        self.game = OthelloGame()
        self.simulations_per_move = simulations_per_move
        self.root = new_root(self.game)
        self.simulations_done = 0
        self.history = []
        self.finished = False
        self.training_data = None
        self.winner = None

    def request_leaf(self):
        """Start one simulation; return the leaf needing the network, or
        None if this simulation completed without it."""
        leaf = select_leaf(self.root)
        if leaf is None:
            self.simulations_done += 1
        return leaf

    def leaf_evaluated(self):
        """Called once the driver has expanded and backed up our leaf."""
        self.simulations_done += 1

    def ready_to_move(self):
        return self.simulations_done >= self.simulations_per_move

    def play_best_move(self):
        """Commit the search's choice and re-root the tree on it."""
        root = self.root
        move = root.select_final_move()

        self.history.append((
            board_planes(self.game.board, self.game.current_player),
            root.visit_distribution(),
            self.game.current_player,
        ))

        self.game.play(*move)

        # Tree reuse: the subtree under the move we just played was already
        # searched, and every statistic in it is still valid, so keep it as
        # the next root instead of throwing the work away. Detaching the
        # parent stops backup from walking into the discarded tree.
        child = root.children[move]
        child.parent = None
        child.game = self.game.copy()
        child.player = self.game.current_player
        self.root = child
        self.simulations_done = 0

        if self.game.game_over:
            self._finish()

    def _finish(self):
        """Backfill every recorded position with the real game result."""
        self.winner = self.game.winner()
        self.training_data = []
        for planes, distribution, player in self.history:
            if self.winner == 0:
                value = 0.0
            elif player == self.winner:
                value = 1.0
            else:
                value = -1.0
            self.training_data.append((planes, distribution, value))
        self.finished = True


def run_self_play_batch(num_games, simulations_per_move, progress=None):
    """Play `num_games` games concurrently, batching their network calls.

    Every pass of the loop advances every unfinished game by exactly one
    simulation, so the leaves collected in one pass are independent by
    construction -- they come from different games, which know nothing
    about each other. That is what makes them safe to evaluate together.

    Returns the finished SelfPlayGame objects, in their original order.
    """
    games = [SelfPlayGame(simulations_per_move) for _ in range(num_games)]
    active = list(games)
    completed = 0

    while active:
        # 1. Every active game descends to the leaf it wants evaluated.
        owners = []
        leaves = []
        for game in active:
            leaf = game.request_leaf()
            if leaf is not None:
                owners.append(game)
                leaves.append(leaf)

        # 2. One network call for all of them.
        evaluate_leaves(leaves)
        for game in owners:
            game.leaf_evaluated()

        # 3. Games that have used their simulation budget play their move.
        for game in active:
            while game.ready_to_move() and not game.finished:
                game.play_best_move()

        still_active = [game for game in active if not game.finished]
        if progress is not None and len(still_active) != len(active):
            completed += len(active) - len(still_active)
            progress(completed, num_games)
        active = still_active

    return games


def train_on_positions(training_data):
    """Stack collected positions into arrays and run one training pass.

    Each entry is (planes, policy_target, value_target), where planes is a
    bare (8, 8, 2) -- np.stack adds the batch dimension.
    """
    if not training_data:
        return
    boards = np.stack([planes for planes, _, _ in training_data])
    policy_targets = np.stack([distribution for _, distribution, _ in training_data])
    value_targets = np.array([value for _, _, value in training_data], dtype=np.float32)

    model.fit(
        boards,
        {"policy": policy_targets, "value": value_targets},
        epochs=1,
        verbose=0,
    )


# Kept under its original name so anything already calling it still works.
train_on_game = train_on_positions


def play_self_play_game(num_simulations):
    """Play a single game on its own. Returns (training_data, winner).

    Retained for one-off use and testing; the training loop below uses
    run_self_play_batch, which is far faster per game.
    """
    game = run_self_play_batch(1, num_simulations)[0]
    return game.training_data, game.winner


# ----------------------------------------------------------------------
# Training loop.
# ----------------------------------------------------------------------

NUM_GAMES = 8000
SIMULATIONS_PER_MOVE = 400

# How many games run in lockstep, which is also the width of every network
# call. Bigger batches amortize the per-call cost over more boards, with
# diminishing returns once the GPU is actually saturated; the games also
# all live in memory at once, so this trades RAM for speed.
PARALLEL_GAMES = 64

# Save a numbered checkpoint every this many games. Independent of
# PARALLEL_GAMES on purpose: the batch size is a speed knob, how often you
# want a restore point is a separate question. Each file is ~4 MB, so at
# sub-second games a small number here fills the disk with checkpoints
# nobody will ever load.
CHECKPOINT_EVERY = 800

# One log line per game, or just a summary per batch? Per-game lines are
# useful when a run is 100 games and unreadable when it is 10,000.
LOG_EVERY_GAME = False

LOG_PATH = "training_log.txt"
GAME_COUNTER_PATH = "game_counter.txt"

if __name__ == "__main__":
    import time

    # How many games were already played across all previous runs, so
    # checkpoint numbers (and the log) keep counting up instead of
    # restarting at 1 -- and overwriting last run's checkpoints -- every
    # time this script gets run again.
    if os.path.exists(GAME_COUNTER_PATH):
        with open(GAME_COUNTER_PATH) as f:
            games_already_played = int(f.read().strip())
    else:
        games_already_played = 0

    with open(LOG_PATH, "a") as log_file:

        def record(message):
            print(message, flush=True)
            log_file.write(message + "\n")
            log_file.flush()

        played_this_run = 0
        since_checkpoint = 0
        run_started = time.perf_counter()

        while played_this_run < NUM_GAMES:
            batch_size = min(PARALLEL_GAMES, NUM_GAMES - played_this_run)
            first_game_num = games_already_played + played_this_run + 1

            started = time.perf_counter()
            batch = run_self_play_batch(batch_size, SIMULATIONS_PER_MOVE)
            elapsed = time.perf_counter() - started

            training_data = []
            wins = {1: 0, -1: 0, 0: 0}
            for game in batch:
                played_this_run += 1
                wins[game.winner] += 1
                training_data.extend(game.training_data)
                if LOG_EVERY_GAME:
                    winner_name = {1: "black", -1: "white", 0: "draw"}[game.winner]
                    record(f"game {games_already_played + played_this_run} "
                           f"(this run: {played_this_run}/{NUM_GAMES}): "
                           f"winner={winner_name}, positions={len(game.training_data)}")

            total_played = games_already_played + played_this_run
            record(f"games {first_game_num}-{total_played} "
                   f"(this run: {played_this_run}/{NUM_GAMES}): "
                   f"{elapsed:.1f}s ({elapsed / batch_size:.3f}s per game), "
                   f"black={wins[1]} white={wins[-1]} draw={wins[0]}, "
                   f"{len(training_data)} positions")

            with open(GAME_COUNTER_PATH, "w") as counter_file:
                counter_file.write(str(total_played))

            train_on_positions(training_data)

            # Checkpoint on a game count, not on batch boundaries, so
            # PARALLEL_GAMES can be tuned for speed without changing how
            # often restore points appear.
            since_checkpoint += batch_size
            if since_checkpoint >= CHECKPOINT_EVERY or played_this_run >= NUM_GAMES:
                since_checkpoint = 0
                checkpoint_path = f"checkpoint_game_{total_played}.keras"
                model.save(checkpoint_path)
                # Refresh final_model.keras too: that is the file a restart
                # loads, so without this a crashed run would resume from
                # the *previous* run's weights and silently throw away
                # everything trained since.
                model.save(MODEL_PATH)
                record(f"  -> saved {checkpoint_path} (and refreshed {MODEL_PATH})")

        total_elapsed = time.perf_counter() - run_started
        model.save(MODEL_PATH)
        record(f"training complete: {NUM_GAMES} games in {total_elapsed:.1f}s "
               f"({total_elapsed / NUM_GAMES:.3f}s per game), saved {MODEL_PATH}")
