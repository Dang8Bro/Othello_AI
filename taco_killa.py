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

# --- Self-play exploration -------------------------------------------
#
# Without these, self-play is fully deterministic: the same start position
# plus the same network plus argmax move selection produces the same game
# every time. Running 64 games in lockstep then yields 64 identical games
# and 64 copies of the same 60 training positions, which is worthless as
# training data no matter how fast it is generated.
#
# Two independent sources of variation fix that, both standard AlphaZero:
#
#   1. Dirichlet noise mixed into the root priors, drawn separately per
#      game, so each game's search is nudged toward different first moves.
#   2. Sampling the played move from the visit distribution instead of
#      taking the argmax, for the opening phase of the game.
#
# Neither applies to `run_mcts`, the path server.py uses -- when actually
# playing a human you want the strongest move, not a noisy one.

# Fraction of the root prior that comes from noise. 0.25 is the AlphaZero
# value; 0 disables noise entirely.
DIRICHLET_WEIGHT = 0.25

# Concentration of that noise. AlphaZero scales it roughly as 10 / (average
# number of legal moves) -- 0.03 for Go's ~360, 0.3 for chess's ~35. Othello
# averages around 10 legal moves, giving ~1.0. Lower values concentrate the
# noise on fewer moves, making individual games diverge harder.
DIRICHLET_ALPHA = 1.0

# How many opening moves are sampled from the visit distribution rather
# than played greedily. Sampling explores; greedy play keeps the endgame
# sharp, so the tail of the game stays representative of real strength.
TEMPERATURE_MOVES = 30


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

def build_model():
    """The dual-head network.

    Value head
    ----------
    The value head used to be a single Dense(1) straight off the flattened
    trunk: 4,097 parameters, 1.2% of the network, and no hidden layer, so
    "am I winning" had to be a *linear* function of the final conv features.
    That is the output MCTS leans on hardest -- it replaced rollouts, so
    every simulation backs up whatever it says -- and it measured at 0.28
    correlation with real outcomes and 67% sign accuracy in the endgame.
    More self-play could not fix that; it was a capacity ceiling, not a data
    one.

    It now reduces the trunk with a 1x1 convolution (cheap, and it keeps the
    board's geometry until the very end) and passes through a hidden layer
    before the tanh. The policy head is unchanged.
    """
    board_input = layers.Input(shape=(8, 8, 2))

    first_layer = layers.Conv2D(activation="relu", filters=64, kernel_size=3, padding="same")(board_input)
    second_layer = layers.Conv2D(activation="relu", filters=64, kernel_size=3, padding="same")(first_layer)
    trunk = layers.Conv2D(activation="relu", filters=64, kernel_size=3, padding="same")(second_layer)

    policy = layers.Dense(64, activation="softmax", name="policy")(layers.Flatten()(trunk))

    value_features = layers.Conv2D(filters=32, kernel_size=1, activation="relu",
                                   name="value_conv")(trunk)
    value_hidden = layers.Dense(128, activation="relu",
                                name="value_hidden")(layers.Flatten()(value_features))
    value = layers.Dense(1, activation="tanh", name="value")(value_hidden)

    built = models.Model(inputs=board_input, outputs=[policy, value])
    built.compile(optimizer=tf.keras.optimizers.Adam(),
                  metrics={
                      "policy": "accuracy",
                      "value": "mae",
                  },
                  loss={
                      "policy": tf.keras.losses.CategoricalCrossentropy(),
                      "value": tf.keras.losses.MeanSquaredError()
                  }

    )
    return built


def has_current_architecture(candidate):
    """Whether a loaded model is this file's architecture.

    Saved weights from the old single-Dense value head load without
    complaint -- same inputs, same output shapes -- and would silently keep
    training the network we just replaced. Checking for a layer that only
    the new head has is what stops that.
    """
    return "value_hidden" in {layer.name for layer in candidate.layers}


# If a previously trained model exists, keep training it instead of starting
# over from random weights -- unless it predates the current architecture,
# in which case its weights cannot be carried over.
architecture_changed = False
if os.path.exists(MODEL_PATH):
    _existing = tf.keras.models.load_model(MODEL_PATH)
    if has_current_architecture(_existing):
        model = _existing
        print(f"loaded existing model from {MODEL_PATH}, continuing training")
    else:
        architecture_changed = True
        model = build_model()
        print(f"{MODEL_PATH} has the previous architecture (single-layer value "
              f"head); its weights cannot be reused, starting fresh")
else:
    model = build_model()
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

# One compiled function per model, keyed by identity. A single-slot cache
# would be fine for training, which only ever uses one network -- but the
# server offers difficulty levels backed by different checkpoints, and
# alternating between two models would then retrace on every single call,
# which costs far more than the search itself. The model object is kept in
# the value so its id() cannot be recycled onto a different object.
_compiled_fns = {}


def predict(board_tensor, net=None):
    """Run a batch of boards through the network, returning (policy, value)
    as numpy arrays of shape (N, 64) and (N, 1).

    Returns numpy rather than tf.Tensors so the search tree keeps doing
    plain Python arithmetic -- building tiny tensor ops per node would be
    slower than what this replaced.
    """
    net = model if net is None else net

    entry = _compiled_fns.get(id(net))
    if entry is None or entry[0] is not net:
        entry = (net, tf.function(
            lambda x: net(x, training=False),
            input_signature=_PREDICT_SIGNATURE,
        ))
        _compiled_fns[id(net)] = entry

    policy_output, value_output = entry[1](board_tensor)
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

    def sample_move(self, rng):
        """A move drawn in proportion to child visit counts.

        This is temperature 1 in AlphaZero's terms. Using it for the
        opening moves is what makes two self-play games from the same
        position diverge -- argmax alone would replay one fixed game
        forever, however good the search is.
        """
        moves = list(self.children)
        counts = np.array([self.children[move].visit_count for move in moves],
                          dtype=np.float64)
        total = counts.sum()
        if total <= 0:
            # No simulation reached a child (possible only at absurdly low
            # simulation counts); fall back to picking uniformly.
            return moves[int(rng.integers(len(moves)))]
        return moves[int(rng.choice(len(moves), p=counts / total))]

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
    """Give `node` one child per legal move, seeded with its policy prior.

    The priors are renormalized to sum to 1 across the legal moves. The
    network's softmax runs over all 64 squares, most of which are illegal
    here, and whatever mass it spends on them is simply dropped when we
    read off the legal entries. Without renormalizing, the priors on a
    node's children sum to whatever happened to be left over -- maybe 0.9
    in one position and 0.2 in another.

    That matters because prior multiplies the exploration term in PUCT, so
    the effective value of c_puct would silently swing by several times
    from position to position, for a reason that has nothing to do with the
    position: how confidently the network avoided illegal squares. After
    normalizing, c_puct means the same thing everywhere.
    """
    moves = node.game.legal_moves()
    node.expanded = True
    if not moves:
        return  # only reachable at a finished game, which never gets here

    priors = [float(policy[move_to_index(*move)]) for move in moves]
    total = sum(priors)
    if total > 0.0:
        priors = [prior / total for prior in priors]
    else:
        # An untrained (or very confused) network can put essentially all
        # its mass on illegal squares. Uniform is the honest prior then,
        # and it keeps the search exploring instead of dividing by zero.
        priors = [1.0 / len(moves)] * len(moves)

    for move, prior in zip(moves, priors):
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


def evaluate_leaves(leaves, net=None):
    """Network-evaluate a list of leaves in one batched call.

    This is the whole point of the lockstep driver: one call for N leaves
    costs barely more than one call for a single leaf, because most of the
    cost is per-call, not per-board.

    `net` selects which network to ask; None means the module-level model
    that training uses. The server passes a specific checkpoint here to
    back a difficulty level.
    """
    if not leaves:
        return
    batch = np.stack([board_planes(leaf.game.board, leaf.player) for leaf in leaves])
    policies, values = predict(batch, net)
    for leaf, policy, value in zip(leaves, policies, values):
        expand(leaf, policy)
        backup(leaf, float(value[0]))


def new_root(game):
    """A fresh, unexpanded root node holding a copy of `game`."""
    root = Node(game.current_player, 0.0, None)
    root.game = game.copy()
    return root


def analyse_positions(positions, num_simulations, net=None):
    """Search many independent positions at once, returning a root each.

    Game review needs one search per position of a finished game -- sixty
    of them -- and doing that one at a time would mean sixty times the
    per-call network overhead for no reason. Unlike the simulations inside
    a single search, these positions do not depend on each other at all:
    they are already known, and none of their results feed the others. So
    they can be advanced in lockstep and share one batched network call per
    round, the same trick self-play uses across games.

    Positions that are already over consume their simulations without ever
    reaching the network, since their true result is known.
    """
    roots = [new_root(position) for position in positions]
    remaining = list(range(len(roots)))
    left = [num_simulations] * len(roots)

    while remaining:
        owners = []
        leaves = []
        for index in remaining:
            leaf = select_leaf(roots[index])
            if leaf is None:
                left[index] -= 1
            else:
                owners.append(index)
                leaves.append(leaf)

        evaluate_leaves(leaves, net)
        for index in owners:
            left[index] -= 1

        remaining = [index for index in remaining if left[index] > 0]

    return roots


def child_value_for(parent, child):
    """A child's mean value expressed from the parent's point of view.

    Child values are stored from the child's own player's perspective, and
    a forced pass means that is not always the opposite of the parent's --
    so the sign flip is decided by comparing players, never by depth.
    """
    value = child.mean_value()
    return -value if child.player != parent.player else value


def run_mcts(game, num_simulations, net=None):
    """Search one position and return the root. Single-game convenience
    wrapper -- the browser opponent in server.py uses this.

    Deliberately free of the self-play exploration noise: when someone is
    actually playing this bot, they should face its honest best move.

    Self-play should use run_self_play_batch instead, which shares one
    network call across many games rather than making a batch of one.
    """
    root = new_root(game)
    for _ in range(num_simulations):
        leaf = select_leaf(root)
        if leaf is not None:
            evaluate_leaves([leaf], net)
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

    def __init__(self, simulations_per_move, seed=None):
        self.game = OthelloGame()
        self.simulations_per_move = simulations_per_move
        self.root = new_root(self.game)
        self.simulations_done = 0
        self.history = []
        self.finished = False
        self.training_data = None
        self.winner = None
        self.moves_played = 0
        self.noise_applied = False
        # Each game needs its OWN random stream. Sharing one generator
        # across the batch would still decorrelate the games, but seeding
        # per game keeps a run reproducible when a seed is supplied, and
        # makes it obvious that no two games are drawing the same noise.
        self.rng = np.random.default_rng(seed)

    def _apply_root_noise(self):
        """Mix Dirichlet noise into the root's priors, once per move.

        Has to happen after the root is expanded, because before that it
        has no children to perturb -- the first simulation of each move is
        what creates them. Re-applied every move, including to a root
        inherited through tree reuse, whose priors were set when it was an
        ordinary interior node and carry no noise.
        """
        root = self.root
        if self.noise_applied or not root.children or DIRICHLET_WEIGHT <= 0.0:
            return
        moves = list(root.children)
        noise = self.rng.dirichlet([DIRICHLET_ALPHA] * len(moves))
        for move, sample in zip(moves, noise):
            child = root.children[move]
            child.prior = ((1.0 - DIRICHLET_WEIGHT) * child.prior
                           + DIRICHLET_WEIGHT * float(sample))
        self.noise_applied = True

    def request_leaf(self):
        """Start one simulation; return the leaf needing the network, or
        None if this simulation completed without it."""
        self._apply_root_noise()
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
        """Commit a move and re-root the tree on it.

        Opening moves are sampled from the visit distribution and later
        ones played greedily. The recorded policy target is the full visit
        distribution either way -- sampling changes which line this game
        explores, not what the search believed about the position.
        """
        root = self.root
        if self.moves_played < TEMPERATURE_MOVES:
            move = root.sample_move(self.rng)
        else:
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
        self.moves_played += 1
        self.noise_applied = False   # fresh noise for the next move

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


def run_self_play_batch(num_games, simulations_per_move, progress=None, seed=None):
    """Play `num_games` games concurrently, batching their network calls.

    Every pass of the loop advances every unfinished game by exactly one
    simulation, so the leaves collected in one pass are independent by
    construction -- they come from different games, which know nothing
    about each other. That is what makes them safe to evaluate together.

    The games are only *mechanically* independent, though -- identical
    positions, identical network and greedy move choice would still make
    them play the same game. The exploration in SelfPlayGame is what makes
    them actually differ, and each one is given its own random stream here.
    Passing `seed` makes a whole batch reproducible without making its
    games identical to each other.

    Returns the finished SelfPlayGame objects, in their original order.
    """
    seeds = np.random.SeedSequence(seed).spawn(num_games)
    games = [SelfPlayGame(simulations_per_move, seed=child_seed)
             for child_seed in seeds]
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


# ----------------------------------------------------------------------
# Symmetry and the replay buffer.
# ----------------------------------------------------------------------

# Othello is symmetric under the 8 transformations of the square: four
# rotations, each optionally mirrored. A position and its policy target
# transform together, and the outcome is untouched -- so every game played
# is worth eight training examples, for no extra self-play at all.
NUM_SYMMETRIES = 8


def apply_symmetry(planes, distribution, symmetry):
    """Transform a position and its policy target by one of the 8 symmetries.

    Both are rotated and mirrored the same way, so the square a move points
    at stays the same square. Rotating one but not the other would quietly
    teach the network to play mirror-image moves.
    """
    rotations = symmetry % 4
    board = np.rot90(planes, rotations, axes=(0, 1))
    policy = np.rot90(distribution.reshape(BOARD_SIZE, BOARD_SIZE), rotations)
    if symmetry >= 4:
        board = board[:, ::-1, :]
        policy = policy[:, ::-1]
    # rot90 and reverse slicing both return views with awkward strides;
    # copying now keeps the per-sample assembly below cheap.
    return (np.ascontiguousarray(board),
            np.ascontiguousarray(policy).reshape(NUM_SQUARES))


class ReplayBuffer:
    """The last N positions of self-play, oldest overwritten first.

    Training used to fit on one batch of games and then throw it away, so
    every position was seen exactly once and nothing stopped the network
    quietly forgetting what two batches ago had taught it. Sampling from a
    window instead means each position keeps being revisited while it is
    still recent, and each gradient step sees a mix of ages rather than 64
    games that all look alike.

    Deliberately a list with an index rather than a deque: deque random
    access is O(n), and sampling thousands of positions per step out of a
    hundred thousand would cost more than the training does.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.items = []
        self.cursor = 0

    def add(self, item):
        if len(self.items) < self.capacity:
            self.items.append(item)
        else:
            self.items[self.cursor] = item
            self.cursor = (self.cursor + 1) % self.capacity

    def extend(self, items):
        for item in items:
            self.add(item)

    def __len__(self):
        return len(self.items)


def sample_training_arrays(buffer, sample_size, rng):
    """Draw `sample_size` positions from the buffer, each under a random
    symmetry, as arrays ready for fit().

    The symmetry is applied at sampling time rather than at storage time on
    purpose: storing all eight copies would multiply the buffer's memory by
    eight for information that costs almost nothing to regenerate.
    """
    size = min(sample_size, len(buffer))
    indices = rng.choice(len(buffer), size=size, replace=False)
    symmetries = rng.integers(NUM_SYMMETRIES, size=size)

    boards = np.empty((size, BOARD_SIZE, BOARD_SIZE, 2), dtype=np.float32)
    policies = np.empty((size, NUM_SQUARES), dtype=np.float32)
    values = np.empty(size, dtype=np.float32)

    for slot, (index, symmetry) in enumerate(zip(indices, symmetries)):
        planes, distribution, value = buffer.items[index]
        boards[slot], policies[slot] = apply_symmetry(planes, distribution,
                                                      int(symmetry))
        values[slot] = value
    return boards, policies, values


def train_from_replay(buffer, rng, sample_size, batch_size):
    """One training pass over a fresh random sample of the buffer."""
    if len(buffer) == 0:
        return 0
    boards, policies, values = sample_training_arrays(buffer, sample_size, rng)
    model.fit(
        boards,
        {"policy": policies, "value": values},
        epochs=1,
        batch_size=batch_size,
        verbose=0,
    )
    return len(values)


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

NUM_GAMES = 6000
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

# How many games' worth of positions stay available to train on. Roughly
# 60 positions per game, so 2,000 games is ~120k positions and ~90 MB.
# Larger windows are steadier but slower to reflect the network's current
# play; smaller ones chase the latest games and forget faster.
REPLAY_GAMES = 2000
REPLAY_CAPACITY = REPLAY_GAMES * 60

# Positions drawn from the buffer per training step, and the minibatch size
# within that. A self-play batch of 64 games adds ~3,840 positions, so
# sampling 8,192 means each step revisits older positions as well as the
# new ones -- which is the entire point of keeping a buffer.
TRAIN_SAMPLE_SIZE = 8192
TRAIN_BATCH_SIZE = 64

LOG_PATH = "training_log.txt"
GAME_COUNTER_PATH = "game_counter.txt"

# Where the previous architecture's model and counter are moved if this file
# is run after the network shape changed, so a fresh lineage does not
# overwrite a trained one.
RETIRED_MODEL_PATH = "retired_final_model.keras"
RETIRED_COUNTER_PATH = "retired_game_counter.txt"

if __name__ == "__main__":
    import time

    # A changed architecture means a new lineage: the old weights cannot be
    # carried over, so the old model and its game count are moved aside
    # rather than silently overwritten by the first checkpoint below.
    if architecture_changed:
        if os.path.exists(MODEL_PATH):
            os.replace(MODEL_PATH, RETIRED_MODEL_PATH)
            print(f"moved the previous model to {RETIRED_MODEL_PATH}", flush=True)
        if os.path.exists(GAME_COUNTER_PATH):
            os.replace(GAME_COUNTER_PATH, RETIRED_COUNTER_PATH)
            print(f"moved the previous game count to {RETIRED_COUNTER_PATH}", flush=True)

    # How many games were already played across all previous runs, so
    # checkpoint numbers (and the log) keep counting up instead of
    # restarting at 1 -- and overwriting last run's checkpoints -- every
    # time this script gets run again.
    if os.path.exists(GAME_COUNTER_PATH):
        with open(GAME_COUNTER_PATH) as f:
            games_already_played = int(f.read().strip())
    else:
        games_already_played = 0

    replay = ReplayBuffer(REPLAY_CAPACITY)
    train_rng = np.random.default_rng()

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

            replay.extend(training_data)
            trained = train_from_replay(replay, train_rng,
                                        TRAIN_SAMPLE_SIZE, TRAIN_BATCH_SIZE)
            record(f"  -> trained on {trained} sampled positions "
                   f"(buffer holds {len(replay)} of {REPLAY_CAPACITY})")

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
