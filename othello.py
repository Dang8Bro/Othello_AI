"""Core Othello (Reversi) game engine.

Pure rules only: board state, legal move generation, flipping, passing,
game-over detection. No AI, no evaluation, no search — this module is the
ground truth that the later MCTS/network code will trust completely.

Board representation
--------------------
An 8x8 numpy array of int8, where:

    EMPTY = 0,  BLACK = +1,  WHITE = -1

Why this encoding:
  * (row, col) indexing keeps the direction-walking code readable — a
    direction is just (dr, dc) added to (r, c), and "off the board" is a
    plain bounds check. A flat list of 64 would need index arithmetic where
    stepping "left" off row 3 silently wraps onto row 2 unless you guard it.
  * Using +1/-1 for the two colors means "the opponent of p" is simply -p,
    and later `board * current_player` gives the canonical
    my-discs-are-positive view an AlphaZero-style network wants as input.
    A numpy int8 array also feeds straight into TensorFlow with no
    conversion step.
  * Bitboards (two uint64s, one per color) are the fast choice for heavy
    MCTS rollouts, but the flip logic becomes shift-and-mask twiddling
    where wrap-around bugs are hard to see. Correctness is the priority
    here, so we take the readable representation; a bitboard engine can be
    validated against this one later if speed becomes the bottleneck.
"""

import numpy as np

BOARD_SIZE = 8

EMPTY = 0
BLACK = 1    # Black moves first, per standard rules.
WHITE = -1

# All 8 compass directions as (row_delta, col_delta).
DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]


def opponent(player):
    """The other player. With the +1/-1 encoding this is just negation."""
    return -player


class OthelloGame:
    """One game of Othello: board + whose turn it is.

    Turn handling (including forced passes) lives inside `play()`, so the
    caller never has to remember to pass manually: after every move the
    engine advances to whichever player can actually move, or declares the
    game over if neither can.
    """

    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        # Standard starting position: 4 center discs, alternating colors.
        # White on d4/e5 -> (3,3),(4,4); Black on e4/d5 -> (3,4),(4,3).
        self.board[3, 3] = WHITE
        self.board[4, 4] = WHITE
        self.board[3, 4] = BLACK
        self.board[4, 3] = BLACK
        self.current_player = BLACK
        self.game_over = False

    # ------------------------------------------------------------------
    # Flip logic — the heart of the rules.
    # ------------------------------------------------------------------

    def flips_for_move(self, row, col, player=None):
        """All discs that would be flipped if `player` played at (row, col).

        Returns a list of (r, c) positions. An empty list means the move is
        illegal (a move MUST flip at least one disc). This one function is
        both the legality test and the flip calculator, so the two can
        never disagree.

        For each of the 8 directions we walk outward from (row, col):
        collect consecutive opponent discs, and if the walk then lands on
        one of `player`'s own discs, everything collected is captured. If
        the walk instead hits an empty square or falls off the board, that
        direction captures nothing.
        """
        if player is None:
            player = self.current_player
        if self.board[row, col] != EMPTY:
            return []  # Can only play on an empty square.

        opp = opponent(player)
        flips = []

        for dr, dc in DIRECTIONS:
            # Walk in this direction, gathering opponent discs.
            candidates = []
            r, c = row + dr, col + dc
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r, c] == opp:
                candidates.append((r, c))
                r, c = r + dr, c + dc
            # The gathered discs only flip if the line is capped by one of
            # our own discs (still on the board, and not empty).
            if candidates and 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE \
                    and self.board[r, c] == player:
                flips.extend(candidates)

        return flips

    def legal_moves(self, player=None):
        """Every legal move for `player` (default: current player),
        as a list of (row, col) tuples."""
        if player is None:
            player = self.current_player
        return [
            (r, c)
            for r in range(BOARD_SIZE)
            for c in range(BOARD_SIZE)
            if self.flips_for_move(r, c, player)
        ]

    def has_any_move(self, player):
        """True if `player` has at least one legal move (early-exit scan)."""
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.flips_for_move(r, c, player):
                    return True
        return False

    # ------------------------------------------------------------------
    # Playing a move.
    # ------------------------------------------------------------------

    def play(self, row, col):
        """Current player plays at (row, col). Raises ValueError if illegal.

        Handles the turn transition automatically:
          * normally the opponent moves next;
          * if the opponent has no legal move, the turn passes straight
            back to the mover (a forced pass);
          * if neither side can move, the game is over.

        Returns an info dict: {"flipped": [(r,c), ...], "passed": bool}
        where "passed" means the opponent had to skip their turn.
        """
        if self.game_over:
            raise ValueError("Game is over; no more moves can be played.")

        flips = self.flips_for_move(row, col)
        if not flips:
            raise ValueError(
                f"Illegal move {(row, col)} for player {self.current_player}: "
                "square occupied or move flips no discs."
            )

        mover = self.current_player
        self.board[row, col] = mover
        for r, c in flips:
            self.board[r, c] = mover

        # Advance the turn, auto-passing if needed.
        passed = False
        opp = opponent(mover)
        if self.has_any_move(opp):
            self.current_player = opp
        elif self.has_any_move(mover):
            passed = True            # Opponent must pass; mover goes again.
        else:
            self.game_over = True    # Nobody can move: game ends.

        return {"flipped": flips, "passed": passed}

    def copy(self):
        """An independent copy of this game.

        Search code needs to explore a move without disturbing the real
        position, so it copies, plays, and throws the copy away. The board
        is copied (not shared) so writes to the clone can't leak back.
        """
        clone = OthelloGame()
        clone.board = self.board.copy()
        clone.current_player = self.current_player
        clone.game_over = self.game_over
        return clone

    # ------------------------------------------------------------------
    # Scoring.
    # ------------------------------------------------------------------

    def counts(self):
        """Disc counts as a (black, white) tuple."""
        black = int(np.count_nonzero(self.board == BLACK))
        white = int(np.count_nonzero(self.board == WHITE))
        return black, white

    def winner(self):
        """BLACK, WHITE, or 0 for a draw. Only meaningful once game_over."""
        black, white = self.counts()
        if black > white:
            return BLACK
        if white > black:
            return WHITE
        return 0


# ----------------------------------------------------------------------
# Helper for tests and debugging: build/print positions as ASCII art.
# ----------------------------------------------------------------------

def board_from_strings(rows):
    """Build a board array from 8 strings of 8 chars: '.'=empty, 'B', 'W'.

    Lets tests state positions visually instead of as coordinate soup.
    """
    assert len(rows) == BOARD_SIZE, "need exactly 8 rows"
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    lookup = {".": EMPTY, "B": BLACK, "W": WHITE}
    for r, row in enumerate(rows):
        assert len(row) == BOARD_SIZE, f"row {r} needs exactly 8 chars"
        for c, ch in enumerate(row):
            board[r, c] = lookup[ch]
    return board


def board_to_strings(board):
    """Inverse of board_from_strings, for readable assertion messages."""
    lookup = {EMPTY: ".", BLACK: "B", WHITE: "W"}
    return ["".join(lookup[int(cell)] for cell in row) for row in board]
