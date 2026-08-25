"""Minimal Flask server exposing the Othello engine to the browser UI.

Deliberately tiny: the engine owns all the rules; this file just serializes
game state to JSON. One global game instance is enough for local
human-vs-human play.

Future AI hook: when the trained bot exists, implement /api/ai_move to run
MCTS and call game.play() with its chosen move. The frontend already routes
non-human turns through that endpoint, so nothing else changes.

Run with:  python server.py   then open http://127.0.0.1:5000
"""

import tensorflow as tf
from flask import Flask, jsonify, request

from othello import BLACK, WHITE, OthelloGame
import taco_killa
from taco_killa import run_mcts

# Importing taco_killa builds a fresh, randomly-initialized model as a side
# effect (that's just what running that file's top-level code does). Node
# .expand() looks up "model" as a global inside taco_killa's own module
# namespace every time it runs -- so replacing taco_killa.model here, from
# outside, is enough to redirect every future search through this trained
# checkpoint instead, with no changes needed inside taco_killa.py itself.
#
# Point this at whichever checkpoint you want to play against --
# checkpoint_game_5.keras, checkpoint_game_50.keras, final_model.keras, etc.
# -- to compare the bot at different stages of training.
CHECKPOINT_PATH = "final_model.keras"
taco_killa.model = tf.keras.models.load_model(CHECKPOINT_PATH)

app = Flask(__name__, static_folder="static", static_url_path="")

game = OthelloGame()

# How many MCTS simulations the bot runs per move. Matched to the training
# default so the bot plays at the depth its targets were generated at.
# Measured at ~0.55s per move on CPU here, which is unnoticeable against a
# click; 800 roughly doubles that and is still comfortable if you want a
# stronger opponent than the one training produced.
AI_SIMULATIONS = 400

PLAYER_NAMES = {BLACK: "black", WHITE: "white"}


def state_json(last_info=None):
    """Everything the UI needs to render, in one payload."""
    black, white = game.counts()
    state = {
        "board": game.board.tolist(),            # 8x8 of -1/0/1
        "current_player": game.current_player,   # 1=black, -1=white
        "current_player_name": PLAYER_NAMES[game.current_player],
        "legal_moves": game.legal_moves() if not game.game_over else [],
        "game_over": game.game_over,
        "counts": {"black": black, "white": white},
    }
    if game.game_over:
        w = game.winner()
        state["winner"] = PLAYER_NAMES.get(w, "draw")
    if last_info is not None:
        state["last_move"] = last_info
    return state


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.get("/api/state")
def get_state():
    return jsonify(state_json())


@app.post("/api/move")
def post_move():
    data = request.get_json(force=True)
    try:
        info = game.play(int(data["row"]), int(data["col"]))
    except (ValueError, KeyError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(state_json({
        "row": data["row"], "col": data["col"],
        "flipped": info["flipped"], "passed": info["passed"],
    }))


@app.post("/api/new_game")
def new_game():
    global game
    game = OthelloGame()
    return jsonify(state_json())


@app.post("/api/ai_move")
def ai_move():
    if game.game_over:
        return jsonify({"error": "Game is over; no more moves can be played."}), 400
    root = run_mcts(game, AI_SIMULATIONS)
    row, col = root.select_final_move()
    info = game.play(row, col)
    return jsonify(state_json({
        "row": row, "col": col,
        "flipped": info["flipped"], "passed": info["passed"],
    }))


if __name__ == "__main__":
    # host="0.0.0.0" makes this reachable from other devices on the same
    # network (e.g. a phone), not just this machine. debug=False on purpose
    # here -- Flask's debugger can execute code if reached, which is only
    # safe to leave on for a server that's localhost-only.
    app.run(debug=False, host="0.0.0.0", port=5000)
