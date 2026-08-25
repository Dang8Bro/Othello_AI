"""Core Othello (Reversi) game engine.

Pure rules only: board state, legal move generation, flipping, passing,
game-over detection. No AI, no evaluation, no search — this module is the
ground truth that the later MCTS/network code will trust completely.

Board representation
--------------------
Externally, `game.board` is an 8x8 numpy array of int8, where:

    EMPTY = 0,  BLACK = +1,  WHITE = -1

Using +1/-1 for the two colors means "the opponent of p" is simply -p, and
`board * current_player` gives the canonical my-discs-are-positive view an
AlphaZero-style network wants as input.

Internally those same 64 squares live in a flat Python list (`_cells`),
indexed 0..63 as `row * 8 + col`, and `board` is a property that builds the
numpy view on demand. The reason is speed. The rules code below reads
individual squares in very tight loops, and a numpy scalar lookup
(`arr[r, c]`) costs roughly an order of magnitude more than a plain list
index, because every access constructs a numpy scalar object. Under MCTS
these scans run hundreds of thousands of times per game, and that gap was
the single largest consumer of self-play time.

The other half of the speedup is RAYS: a precomputed table of the squares
lying in each of the 8 directions from each square. Walking a precomputed
tuple removes the per-step bounds checks and (row, col) arithmetic the
previous version redid on every call.

Bitboards (two uint64s per position) would be faster still, but the flip
logic becomes shift-and-mask twiddling where wrap-around bugs are hard to
see. This representation keeps the rules readable — the direction walks
below are the same algorithm as before, just over a cheaper array — and
the tests in test_othello.py pin the behavior either way.
"""

import numpy as np

BOARD_SIZE = 8
NUM_SQUARES = BOARD_SIZE * BOARD_SIZE

EMPTY = 0
BLACK = 1    # Black moves first, per standard rules.
WHITE = -1

# All 8 compass directions as (row_delta, col_delta).
DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]


def _build_rays():
    """RAYS[square][d] = the square indices walking off `square` in
    direction d, in order, stopping at the edge of the board.

    Precomputing these is what lets the flip walk below skip bounds checks
    entirely: running off the board is just the ray running out of entries.
    """
    table = []
    for square in range(NUM_SQUARES):
        row, col = divmod(square, BOARD_SIZE)
        per_direction = []
        for dr, dc in DIRECTIONS:
            ray = []
            r, c = row + dr, col + dc
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                ray.append(r * BOARD_SIZE + c)
                r, c = r + dr, c + dc
            per_direction.append(tuple(ray))
        table.append(tuple(per_direction))
    return tuple(table)


RAYS = _build_rays()

# Precomputed (row, col) for each flat index, so the hot loops can convert
# back without paying for divmod.
COORDS = tuple(divmod(square, BOARD_SIZE) for square in range(NUM_SQUARES))


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
        self._cells = [EMPTY] * NUM_SQUARES
        # Standard starting position: 4 center discs, alternating colors.
        # White on d4/e5 -> (3,3),(4,4); Black on e4/d5 -> (3,4),(4,3).
        self._cells[3 * BOARD_SIZE + 3] = WHITE
        self._cells[4 * BOARD_SIZE + 4] = WHITE
        self._cells[3 * BOARD_SIZE + 4] = BLACK
        self._cells[4 * BOARD_SIZE + 3] = BLACK
        self.current_player = BLACK
        self.game_over = False

    # ------------------------------------------------------------------
    # The numpy view.
    # ------------------------------------------------------------------

    @property
    def board(self):
        """The position as a fresh 8x8 int8 numpy array.

        Fresh, rather than a view onto internal state, so callers that keep
        the result — the search stores board snapshots inside tree nodes —
        cannot be surprised by it changing underneath them later.
        """
        return np.array(self._cells, dtype=np.int8).reshape(BOARD_SIZE, BOARD_SIZE)

    @board.setter
    def board(self, value):
        """Replace the whole position at once, e.g. from board_from_strings."""
        flat = np.asarray(value, dtype=np.int8).reshape(NUM_SQUARES)
        self._cells = [int(v) for v in flat]

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
        one of the mover's own discs, everything collected is captured. If
        the walk instead hits an empty square or runs off the board, that
        direction captures nothing.
        """
        if player is None:
            player = self.current_player
        squares = self._flip_squares(row * BOARD_SIZE + col, player)
        return [COORDS[square] for square in squares]

    def _flip_squares(self, square, player):
        """flips_for_move's inner loop, in flat 0..63 indices.

        Split out so `play` can stay in flat indices and skip converting
        coordinates it is only going to use as array offsets anyway.
        """
        cells = self._cells
        if cells[square] != EMPTY:
            return []  # Can only play on an empty square.

        opp = -player
        flips = []
        for ray in RAYS[square]:
            # Walk this direction, gathering opponent discs.
            run = []
            for target in ray:
                value = cells[target]
                if value == opp:
                    run.append(target)
                    continue
                # The gathered discs only flip if the line is capped by one
                # of our own discs.
                if run and value == player:
                    flips.extend(run)
                break
            # Exhausting the ray without hitting a capping disc means we
            # ran off the board, which captures nothing.
        return flips

    def _can_play(self, square, player):
        """True if `player` may play `square` (flat index).

        Same walk as _flip_squares, but it returns the moment it proves the
        move legal instead of collecting every captured disc. Move
        generation only needs the yes/no, and skipping the list building is
        most of why generation is cheap.
        """
        cells = self._cells
        if cells[square] != EMPTY:
            return False

        opp = -player
        for ray in RAYS[square]:
            saw_opponent = False
            for target in ray:
                value = cells[target]
                if value == opp:
                    saw_opponent = True
                    continue
                if saw_opponent and value == player:
                    return True
                break
        return False

    def legal_moves(self, player=None):
        """Every legal move for `player` (default: current player),
        as a list of (row, col) tuples."""
        if player is None:
            player = self.current_player
        can_play = self._can_play
        return [COORDS[square] for square in range(NUM_SQUARES)
                if can_play(square, player)]

    def has_any_move(self, player):
        """True if `player` has at least one legal move (early-exit scan)."""
        can_play = self._can_play
        for square in range(NUM_SQUARES):
            if can_play(square, player):
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

        mover = self.current_player
        square = row * BOARD_SIZE + col
        flips = self._flip_squares(square, mover)
        if not flips:
            raise ValueError(
                f"Illegal move {(row, col)} for player {self.current_player}: "
                "square occupied or move flips no discs."
            )

        cells = self._cells
        cells[square] = mover
        for target in flips:
            cells[target] = mover

        # Advance the turn, auto-passing if needed.
        passed = False
        opp = -mover
        if self.has_any_move(opp):
            self.current_player = opp
        elif self.has_any_move(mover):
            passed = True            # Opponent must pass; mover goes again.
        else:
            self.game_over = True    # Nobody can move: game ends.

        return {"flipped": [COORDS[t] for t in flips], "passed": passed}

    def copy(self):
        """An independent copy of this game.

        Search code needs to explore a move without disturbing the real
        position, so it copies, plays, and throws the copy away. The cell
        list is copied (not shared) so writes to the clone can't leak back.

        Built via __new__ rather than OthelloGame() because the constructor
        would lay out the opening position we are about to overwrite, and
        this runs once per expanded search node.
        """
        clone = OthelloGame.__new__(OthelloGame)
        clone._cells = self._cells[:]
        clone.current_player = self.current_player
        clone.game_over = self.game_over
        return clone

    # ------------------------------------------------------------------
    # Scoring.
    # ------------------------------------------------------------------

    def counts(self):
        """Disc counts as a (black, white) tuple."""
        cells = self._cells
        black = 0
        white = 0
        for value in cells:
            if value == BLACK:
                black += 1
            elif value == WHITE:
                white += 1
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
