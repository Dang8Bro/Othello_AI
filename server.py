"""Flask server for the Othello board: per-visitor games against a ladder
of bot opponents.

The engine owns all the rules; this file owns session state and which
network answers for a given difficulty.

Sessions
--------
Every visitor gets their own game, keyed by a cookie. The previous version
kept one module-level OthelloGame, which meant two people on the site --
or even two browser tabs -- shared a single board and fought over it. Games
live in memory only; restarting the server forgets them, which is fine for
something with no accounts yet.

Difficulty
----------
A level is three knobs: how many MCTS simulations the bot gets, whether it
plays its best move or samples from the search's visit counts, and which
checkpoint answers. Simulations are the honest strength dial -- the same
network searching 900 nodes genuinely outplays itself searching 12. Sampling
is what keeps the weak levels from grinding out the same game every time,
and what makes them lose in varied, human-ish ways. Checkpoints add a third
flavor: an early checkpoint has not yet learned what a corner is worth, so
it is weak in a different way than a strong network on a short search.

A level naming a checkpoint that is not on disk quietly falls back to the
main model, so the ladder still works on a fresh clone with one .keras file.
"""

import os
import secrets
import threading
import time

import numpy as np
import tensorflow as tf
from flask import Flask, g, jsonify, request

from othello import BLACK, WHITE, OthelloGame
import taco_killa
from taco_killa import run_mcts

# Importing taco_killa builds or loads a model as a side effect. Node
# expansion looks up "model" as a global inside taco_killa's own namespace,
# so replacing it here redirects every search that does not name a network
# explicitly. Point this at whichever checkpoint should be the house
# default -- final_model.keras is the latest training output.
CHECKPOINT_PATH = "final_model.keras"
taco_killa.model = tf.keras.models.load_model(CHECKPOINT_PATH)

app = Flask(__name__, static_folder="static", static_url_path="")

PLAYER_NAMES = {BLACK: "black", WHITE: "white"}


# ----------------------------------------------------------------------
# The opponent ladder.
# ----------------------------------------------------------------------

LEVELS = [
    {
        "id": "sprout",
        "name": "Sprout",
        "blurb": "Plays on instinct alone. Has never once wondered what a corner is for.",
        "strength": 1,
        "simulations": 0,          # 0 means "pick a legal move at random"
        "sample": True,
        "checkpoint": None,
    },
    {
        "id": "pebble",
        "name": "Pebble",
        "blurb": "Thinks for a moment, then commits. Beatable, but no longer free.",
        "strength": 2,
        "simulations": 12,
        "sample": True,
        "checkpoint": "checkpoint_game_1032.keras",
    },
    {
        "id": "fern",
        "name": "Fern",
        "blurb": "Enough search to punish a careless edge. Still takes its chances.",
        "strength": 3,
        "simulations": 60,
        "sample": True,
        "checkpoint": "checkpoint_game_2696.keras",
    },
    {
        "id": "willow",
        "name": "Willow",
        "blurb": "No more gambling. Plays the best move it can find, every turn.",
        "strength": 4,
        "simulations": 200,
        "sample": False,
        "checkpoint": None,
    },
    {
        "id": "cedar",
        "name": "Cedar",
        "blurb": "Searches exactly as deep as the games it learned from. Its honest strength.",
        "strength": 5,
        "simulations": 400,
        "sample": False,
        "checkpoint": None,
    },
    {
        "id": "ironwood",
        "name": "Ironwood",
        "blurb": "Twice Cedar's search. Takes a breath before moving, and means it.",
        "strength": 6,
        "simulations": 900,
        "sample": False,
        "checkpoint": None,
    },
]

LEVELS_BY_ID = {level["id"]: level for level in LEVELS}
DEFAULT_LEVEL = "fern"
MAX_STRENGTH = max(level["strength"] for level in LEVELS)

# Checkpoints are loaded once and reused. Without this every AI move would
# reload a 4 MB model from disk; with it, switching opponents is instant
# after the first game against each.
_model_cache = {}
_model_lock = threading.Lock()


def level_network(level):
    """The Keras model backing `level`, or None to use the house default.

    None is meaningful rather than an error: taco_killa's search treats it
    as "use the module-level model", so a level whose checkpoint is missing
    still plays, just at the default network's strength.
    """
    path = level.get("checkpoint")
    if not path or not os.path.exists(path):
        return None
    with _model_lock:
        if path not in _model_cache:
            _model_cache[path] = tf.keras.models.load_model(path)
        return _model_cache[path]


def level_json(level):
    """The public description of a level, for the opponent picker."""
    available = not level["checkpoint"] or os.path.exists(level["checkpoint"])
    return {
        "id": level["id"],
        "name": level["name"],
        "blurb": level["blurb"],
        "strength": level["strength"],
        "max_strength": MAX_STRENGTH,
        "simulations": level["simulations"],
        # True when this level is playing its own checkpoint rather than
        # falling back to the default network.
        "own_network": available and bool(level["checkpoint"]),
    }


def choose_ai_move(game, level):
    """The bot's move for this position at this difficulty."""
    simulations = level["simulations"]
    legal = game.legal_moves()

    if simulations <= 0:
        # No search at all: the bottom rung of the ladder.
        return legal[secrets.randbelow(len(legal))]

    root = run_mcts(game, simulations, net=level_network(level))
    if level["sample"]:
        # Sampling in proportion to visits keeps the lower levels from
        # replaying one fixed game, and makes their mistakes varied.
        return root.sample_move(np.random.default_rng())
    return root.select_final_move()


# ----------------------------------------------------------------------
# Per-visitor sessions.
# ----------------------------------------------------------------------

SESSION_COOKIE = "othello_sid"
SESSION_TTL = 6 * 3600      # seconds of inactivity before a game is dropped
MAX_SESSIONS = 500          # hard ceiling, so an open endpoint cannot grow


class Session:
    """One visitor's game plus the opponent they chose.

    The lock matters because Flask serves requests on threads: a double
    click, or the UI firing an AI move while a human move is still landing,
    would otherwise mutate one board from two threads at once.
    """

    def __init__(self, level_id):
        self.game = OthelloGame()
        self.level_id = level_id
        self.lock = threading.Lock()
        self.touched = time.time()

    @property
    def level(self):
        return LEVELS_BY_ID.get(self.level_id, LEVELS_BY_ID[DEFAULT_LEVEL])


_sessions = {}
_sessions_lock = threading.Lock()


def _evict_stale(now):
    """Drop idle games, then oldest-first if still over the ceiling.

    Called while holding _sessions_lock.
    """
    cutoff = now - SESSION_TTL
    for sid in [sid for sid, s in _sessions.items() if s.touched < cutoff]:
        del _sessions[sid]
    if len(_sessions) > MAX_SESSIONS:
        oldest = sorted(_sessions, key=lambda sid: _sessions[sid].touched)
        for sid in oldest[:len(_sessions) - MAX_SESSIONS]:
            del _sessions[sid]


def current_session():
    """This visitor's session, creating one if the cookie is new."""
    sid = request.cookies.get(SESSION_COOKIE)
    now = time.time()
    with _sessions_lock:
        _evict_stale(now)
        session = _sessions.get(sid) if sid else None
        if session is None:
            sid = secrets.token_urlsafe(18)
            session = Session(DEFAULT_LEVEL)
            _sessions[sid] = session
            # Handed to the browser by attach_session_cookie below.
            g.new_session_id = sid
        session.touched = now
    return session


@app.after_request
def attach_session_cookie(response):
    sid = getattr(g, "new_session_id", None)
    if sid:
        response.set_cookie(SESSION_COOKIE, sid, max_age=SESSION_TTL,
                            samesite="Lax", httponly=True)
    return response


# ----------------------------------------------------------------------
# Serialization.
# ----------------------------------------------------------------------

def state_json(session, last_info=None):
    """Everything the UI needs to render, in one payload."""
    game = session.game
    black, white = game.counts()
    level = session.level
    state = {
        "board": game.board.tolist(),            # 8x8 of -1/0/1
        "current_player": game.current_player,   # 1=black, -1=white
        "current_player_name": PLAYER_NAMES[game.current_player],
        "legal_moves": game.legal_moves() if not game.game_over else [],
        "game_over": game.game_over,
        "counts": {"black": black, "white": white},
        "level": level_json(level),
    }
    if game.game_over:
        state["winner"] = PLAYER_NAMES.get(game.winner(), "draw")
    if last_info is not None:
        state["last_move"] = last_info
    return state


# ----------------------------------------------------------------------
# Routes.
# ----------------------------------------------------------------------

@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.get("/api/levels")
def get_levels():
    session = current_session()
    return jsonify({
        "levels": [level_json(level) for level in LEVELS],
        "current": session.level_id,
    })


@app.get("/api/state")
def get_state():
    session = current_session()
    with session.lock:
        return jsonify(state_json(session))


@app.post("/api/move")
def post_move():
    session = current_session()
    data = request.get_json(force=True, silent=True) or {}
    with session.lock:
        try:
            info = session.game.play(int(data["row"]), int(data["col"]))
        except (ValueError, KeyError, TypeError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(state_json(session, {
            "row": data["row"], "col": data["col"],
            "flipped": info["flipped"], "passed": info["passed"],
        }))


@app.post("/api/new_game")
def new_game():
    session = current_session()
    data = request.get_json(force=True, silent=True) or {}
    requested = data.get("level")
    with session.lock:
        if requested is not None:
            if requested not in LEVELS_BY_ID:
                return jsonify({"error": f"unknown level {requested!r}"}), 400
            session.level_id = requested
        session.game = OthelloGame()
        return jsonify(state_json(session))


@app.post("/api/ai_move")
def ai_move():
    session = current_session()
    with session.lock:
        game = session.game
        if game.game_over:
            return jsonify({"error": "Game is over; no more moves can be played."}), 400
        row, col = choose_ai_move(game, session.level)
        info = game.play(row, col)
        return jsonify(state_json(session, {
            "row": row, "col": col,
            "flipped": info["flipped"], "passed": info["passed"],
        }))


if __name__ == "__main__":
    # host="0.0.0.0" makes this reachable from other devices on the same
    # network (e.g. a phone), not just this machine. debug=False on purpose
    # here -- Flask's debugger can execute code if reached, which is only
    # safe to leave on for a server that's localhost-only. It also avoids
    # the auto-reloader, whose child process used to survive Ctrl+C and
    # keep port 5000 held.
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
