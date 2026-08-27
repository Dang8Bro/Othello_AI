"""Post-game review: an evaluation curve and a verdict on every move.

The idea is chess.com's game review. Replay a finished game, show who was
winning at each point, and label each move -- best, good, mistake, blunder.

How a move is judged
--------------------
Every position of the game is searched independently, and a move's cost is
the drop in evaluation across it:

    loss = (eval of the position before the move, to the mover)
         - (eval of the position after it, converted to the same mover)

Both numbers come from their own full search, which is why this is more
trustworthy than reading the played move's Q value out of the parent's
tree: a move the search barely explored has a near-meaningless Q, but the
position it leads to gets the same scrutiny as any other.

Losses are clamped at zero. A small negative means the position after was
scored slightly better than the position before, which is search noise
rather than a move that improved on best play.

The thresholds below are in value units, where +1 is a certain win for the
side to move and -1 a certain loss. They are judgement calls, not physics,
and they are the first thing to tune if the labels feel wrong.
"""

import math

import numpy as np

from othello import BLACK, OthelloGame
import taco_killa as tk

# Loss thresholds, in value units.
GOOD_LOSS = 0.08
INACCURACY_LOSS = 0.18
MISTAKE_LOSS = 0.35

# "Best" means the search's own top choice, and nothing else. It used to
# also cover any move whose loss was tiny, which sounds reasonable and is
# badly wrong: once one side is clearly winning the evaluation saturates
# near +/-1 and *no* move can lose anything, so a player choosing at random
# in a lost position scored "best" on 25 of 29 moves. Requiring the move to
# actually be the top choice makes the label mean what people read it to
# mean.
#
# A move that was not the top choice but cost nothing is "good", which is
# both true and the common case.

# Positions this one-sided are already decided; the winner cannot gain and
# the loser cannot lose, so nothing either side does there says much about
# how well they played. Such moves are still labelled, but they are left
# out of the accuracy average -- otherwise accuracy mostly measures how
# long the game stayed lopsided.
DECIDED_EVAL = 0.85

# "Brilliant" needs the move to be correct, non-obvious, and *load-bearing*:
# the search must prefer it over the runner-up by this margin, the raw
# policy -- the network's instinct, before any search -- must not have
# ranked it first, and the position must still be live. Without the last
# two conditions it fired on roughly one move in ten, which is not what
# the word means.
BRILLIANT_GAP = 0.30

# Turns a per-move loss into a percentage. exp(-4 * loss) puts a perfect
# move at 100%, the inaccuracy threshold near 49%, and the mistake
# threshold near 25%.
ACCURACY_DECAY = 4.0

CLASSIFICATIONS = ("brilliant", "best", "good", "inaccuracy", "mistake",
                   "blunder", "forced")

PLAYER_NAMES = {1: "black", -1: "white"}


def _to_black(value, player):
    """Re-express a value held from `player`'s view as black's advantage.

    The evaluation bar has to mean one fixed thing all game; values coming
    out of the search always mean "good for whoever is to move".
    """
    return value if player == BLACK else -value


def _ranked_children(root):
    """Root's children, most-visited first."""
    return sorted(root.children.items(),
                  key=lambda item: item[1].visit_count, reverse=True)


def _is_brilliant(root, played_move, was_best, eval_before):
    """Whether a correct move was also one the network would have missed.

    Requires the position to still be live: finding the best move in an
    already-won game is not brilliance, it is bookkeeping.
    """
    if not was_best or abs(eval_before) >= DECIDED_EVAL:
        return False
    ranked = _ranked_children(root)
    if len(ranked) < 2:
        return False   # no alternative to be better than

    best_value = tk.child_value_for(root, ranked[0][1])
    runner_up_value = tk.child_value_for(root, ranked[1][1])
    if best_value - runner_up_value < BRILLIANT_GAP:
        return False

    # Would the raw policy have played it without searching?
    instinct = max(root.children.items(), key=lambda item: item[1].prior)[0]
    return instinct != played_move


def classify(loss, was_best, brilliant, forced):
    # A position with one legal move is not a decision. Counting it as
    # "best" inflated a random player's top-move rate to 48%, because late
    # Othello positions are frequently forced and picking the only move
    # available looked like finding the right one.
    if forced:
        return "forced"
    if brilliant:
        return "brilliant"
    if was_best:
        return "best"
    if loss <= GOOD_LOSS:
        return "good"
    if loss <= INACCURACY_LOSS:
        return "inaccuracy"
    if loss <= MISTAKE_LOSS:
        return "mistake"
    return "blunder"


def move_accuracy(loss):
    return 100.0 * math.exp(-ACCURACY_DECAY * max(0.0, loss))


def rebuild_positions(moves):
    """Every position the game passed through, plus the final one.

    Returns (positions, players). `positions[i]` is the board before
    `moves[i]` was played, and the extra last entry is the finished game.
    """
    positions = []
    players = []
    game = OthelloGame()
    for move in moves:
        positions.append(game.copy())
        players.append(game.current_player)
        game.play(*move)
    positions.append(game.copy())
    players.append(game.current_player)
    return positions, players


def review_game(moves, simulations, net=None):
    """Analyse a finished game. Returns a JSON-ready report.

    Every position is searched in one lockstep pass, so a sixty-move game
    costs about as many batched network calls as a single search does --
    not sixty times as many.
    """
    if not moves:
        return None

    positions, players = rebuild_positions(moves)
    roots = tk.analyse_positions(positions, simulations, net)

    # Each root's value is from its own side-to-move's point of view.
    raw = [root.mean_value() for root in roots]

    steps = []
    totals = {name: {key: 0 for key in CLASSIFICATIONS} for name in ("black", "white")}
    accuracy_sum = {"black": 0.0, "white": 0.0}
    live_count = {"black": 0, "white": 0}
    move_count = {"black": 0, "white": 0}
    best_count = {"black": 0, "white": 0}
    choice_count = {"black": 0, "white": 0}

    for index, move in enumerate(moves):
        mover = players[index]
        root = roots[index]

        before = raw[index]
        # The next position may belong to either player -- a forced pass
        # means the same side moves again -- so convert by comparing.
        after = raw[index + 1] if players[index + 1] == mover else -raw[index + 1]
        loss = max(0.0, before - after)

        best_move = root.select_final_move()
        was_best = (best_move is not None and tuple(move) == tuple(best_move))
        forced = len(root.children) <= 1
        brilliant = _is_brilliant(root, tuple(move), was_best, before)
        label = classify(loss, was_best, brilliant, forced)

        # A position this lopsided cannot be improved or spoiled, so it
        # says nothing about how well the move was chosen.
        live = abs(before) < DECIDED_EVAL and not forced

        name = PLAYER_NAMES[mover]
        totals[name][label] += 1
        move_count[name] += 1
        if not forced:
            choice_count[name] += 1
            if was_best:
                best_count[name] += 1
        if live:
            accuracy_sum[name] += move_accuracy(loss)
            live_count[name] += 1

        steps.append({
            "index": index,
            "move": [int(move[0]), int(move[1])],
            "player": int(mover),
            "player_name": name,
            "best_move": [int(best_move[0]), int(best_move[1])] if best_move else None,
            "was_best": bool(was_best),
            "loss": round(float(loss), 4),
            "classification": label,
            "accuracy": round(move_accuracy(loss), 1),
            "live": bool(live),
            "forced": bool(forced),
            "choices": len(root.children),
            # Board *before* this move, so the UI can show the position the
            # player was actually looking at when they chose.
            "board": positions[index].board.tolist(),
            "eval": round(float(_to_black(before, mover)), 4),
        })

    final_game = positions[-1]
    black_discs, white_discs = final_game.counts()

    return {
        "steps": steps,
        "final": {
            "board": final_game.board.tolist(),
            "eval": round(float(_to_black(raw[-1], players[-1])), 4),
            "counts": {"black": black_discs, "white": white_discs},
            "winner": PLAYER_NAMES.get(final_game.winner(), "draw"),
        },
        "summary": {
            name: {
                "counts": totals[name],
                # Averaged over live positions only, and the count is
                # reported so a mostly-decided game cannot quietly present
                # an accuracy drawn from three moves.
                "accuracy": round(accuracy_sum[name] / live_count[name], 1)
                            if live_count[name] else None,
                "live_moves": live_count[name],
                "moves": move_count[name],
                # How often the search's own top choice was played. Far more
                # discriminating than accuracy, because it does not saturate:
                # a random player lands near 1/(legal moves), a strong one
                # near 90%.
                "best_rate": round(100.0 * best_count[name] / choice_count[name], 1)
                             if choice_count[name] else None,
                "choices": choice_count[name],
            }
            for name in ("black", "white")
        },
        "simulations": simulations,
    }
