"""Tests for the Othello engine.

Positions are written as 8-line ASCII pictures ('.'=empty, 'B', 'W') so a
failing test shows you the board, not a pile of coordinates. Coordinates in
assertions are (row, col), 0-indexed from the top-left.
"""

import random
import unittest

import numpy as np

from othello import (
    BLACK, WHITE, EMPTY, BOARD_SIZE,
    OthelloGame, board_from_strings, board_to_strings,
)


def make_game(rows, player):
    """Build a game in an arbitrary position for testing."""
    game = OthelloGame()
    game.board = board_from_strings(rows)
    game.current_player = player
    return game


class TestLegalMoveGeneration(unittest.TestCase):

    def test_starting_position_black(self):
        """Black opens the game with exactly 4 symmetric options."""
        game = OthelloGame()
        self.assertEqual(
            sorted(game.legal_moves()),
            [(2, 3), (3, 2), (4, 5), (5, 4)],
        )

    def test_starting_position_white(self):
        """If it were White's turn, White would have the mirrored 4."""
        game = OthelloGame()
        self.assertEqual(
            sorted(game.legal_moves(WHITE)),
            [(2, 4), (3, 5), (4, 2), (5, 3)],
        )

    def test_midgame_position_multiple_directions(self):
        """A mid-game shape where legality comes from different directions
        for different candidate squares."""
        game = make_game([
            "........",
            "........",
            "....B.B.",
            "....WW..",
            "BWWW....",
            "........",
            "........",
            "........",
        ], BLACK)
        # Worked out by hand, not by re-running the engine's own logic:
        #   (4,4) caps three separate white runs -- the row-4 run against
        #         the black disc at (4,0), the column-4 disc against (2,4),
        #         and the down-left diagonal disc against (2,6);
        #   (4,6) caps the white disc at (3,5) against (2,4) going up-left.
        # Every other white run ends on an empty square instead of a black
        # disc, so nothing else is legal.
        self.assertEqual(sorted(game.legal_moves()), [(4, 4), (4, 6)])
        # The far-end square of a run is the ONLY capture point: (3, 3) sits
        # next to two white discs but the line beyond them is empty.
        self.assertEqual(game.flips_for_move(3, 3, BLACK), [])
        # The busiest square flips in 3 directions at once:
        flips = game.flips_for_move(4, 4, BLACK)
        self.assertEqual(
            sorted(flips),
            [(3, 4), (3, 5), (4, 1), (4, 2), (4, 3)],
        )

    def test_forced_pass_position(self):
        """White is on the board but has zero legal moves; Black has one.

        White's only disc sits at (0,1); every line of Black discs runs off
        the board before reaching a second White disc, so White can cap
        nothing.
        """
        game = make_game([
            ".WBBBBBB",
            "........",
            "........",
            "........",
            "........",
            "........",
            "........",
            "........",
        ], BLACK)
        self.assertEqual(game.legal_moves(WHITE), [])
        self.assertEqual(game.legal_moves(BLACK), [(0, 0)])

    def test_finished_game_position(self):
        """A full board: no empty squares, so neither player can move."""
        game = make_game([
            "BBBBBBBB",
            "BBBBBBBB",
            "BBBBBBBB",
            "BBBBBBBB",
            "BBBBBWWW",
            "WWWWWWWW",
            "WWWWWWWW",
            "WWWWWWWW",
        ], BLACK)
        self.assertEqual(game.legal_moves(BLACK), [])
        self.assertEqual(game.legal_moves(WHITE), [])
        self.assertEqual(game.counts(), (37, 27))
        self.assertEqual(game.winner(), BLACK)

    def test_move_on_occupied_square_is_illegal(self):
        game = OthelloGame()
        self.assertEqual(game.flips_for_move(3, 3, BLACK), [])
        with self.assertRaises(ValueError):
            game.play(3, 3)

    def test_move_that_flips_nothing_is_illegal(self):
        """Empty square, but no capture line -> illegal (0,0 at game start)."""
        game = OthelloGame()
        self.assertEqual(game.flips_for_move(0, 0, BLACK), [])
        with self.assertRaises(ValueError):
            game.play(0, 0)


class TestFlipLogic(unittest.TestCase):

    def test_flip_single_direction(self):
        """Opening move d3: flips exactly the one white disc to its south."""
        game = OthelloGame()
        info = game.play(2, 3)
        self.assertEqual(info["flipped"], [(3, 3)])
        self.assertEqual(board_to_strings(game.board), [
            "........",
            "........",
            "...B....",
            "...BB...",
            "...BW...",
            "........",
            "........",
            "........",
        ])
        self.assertEqual(game.current_player, WHITE)

    def test_flip_multiple_directions_simultaneously(self):
        """One move capturing north, northeast, and west at once."""
        game = make_game([
            "........",
            "........",
            "....B.B.",
            "....WW..",
            "BWWW....",
            "........",
            "........",
            "........",
        ], BLACK)
        info = game.play(4, 4)
        self.assertEqual(
            sorted(info["flipped"]),
            [(3, 4), (3, 5), (4, 1), (4, 2), (4, 3)],
        )
        # Every white disc was captured.
        self.assertEqual(game.counts(), (9, 0))

    def test_flip_to_far_edge_horizontal(self):
        """Capture a full row: play in the last column, anchor in column 0.
        Off-by-one bugs in the walk loop show up exactly here."""
        game = make_game([
            "BWWWWWW.",
            "........",
            "........",
            "........",
            "........",
            "........",
            "........",
            "........",
        ], BLACK)
        info = game.play(0, 7)
        self.assertEqual(
            sorted(info["flipped"]),
            [(0, c) for c in range(1, 7)],
        )
        self.assertEqual(board_to_strings(game.board)[0], "BBBBBBBB")

    def test_flip_to_far_edge_diagonal(self):
        """Capture the full main diagonal, playing into the corner."""
        game = make_game([
            "B.......",
            ".W......",
            "..W.....",
            "...W....",
            "....W...",
            ".....W..",
            "......W.",
            "........",
        ], BLACK)
        info = game.play(7, 7)
        self.assertEqual(
            sorted(info["flipped"]),
            [(i, i) for i in range(1, 7)],
        )
        self.assertEqual(game.counts(), (8, 0))

    def test_gap_before_anchor_does_not_flip(self):
        """A run of W discs must NOT be captured *through* an empty square,
        even though a black anchor sits further down the line."""
        game = make_game([
            "B.WW....",
            "........",
            "........",
            "........",
            "........",
            "........",
            "........",
            "........",
        ], BLACK)
        # Walking west from (0,4): W, W, then EMPTY at (0,1). The black disc
        # at (0,0) is irrelevant -- the line was already broken.
        self.assertEqual(game.flips_for_move(0, 4, BLACK), [])

    def test_run_falling_off_board_does_not_flip(self):
        """A run of W discs reaching the board edge with no anchor behind it
        captures nothing. This is the bounds guard in the walk loop: without
        it, the index would run past column 7."""
        game = make_game([
            ".WWWWWWW",
            "........",
            "........",
            "........",
            "........",
            "........",
            "........",
            "........",
        ], BLACK)
        # Walking east from (0,0) collects 7 white discs, then falls off the
        # right edge without ever finding a black disc -> no capture.
        self.assertEqual(game.flips_for_move(0, 0, BLACK), [])
        self.assertEqual(game.legal_moves(BLACK), [])


class TestPassAndGameOver(unittest.TestCase):

    def test_automatic_pass_returns_turn_to_mover(self):
        """After Black's move White has no reply, so Black moves again
        without the caller doing anything."""
        game = make_game([
            ".WBBBBBB",
            "........",
            "........",
            "........",
            "........",
            "........",
            "........",
            ".WBBBBBB",
        ], BLACK)
        info = game.play(0, 0)          # flips (0,1); White's turn... almost
        self.assertTrue(info["passed"])  # ...but White had to pass
        self.assertEqual(game.current_player, BLACK)
        self.assertFalse(game.game_over)
        self.assertEqual(game.legal_moves(), [(7, 0)])

    def test_game_over_when_neither_side_can_move(self):
        """Continuing the position above: Black's second move wipes out the
        last white disc, so neither side can move and the game ends —
        with empty squares still on the board."""
        game = make_game([
            ".WBBBBBB",
            "........",
            "........",
            "........",
            "........",
            "........",
            "........",
            ".WBBBBBB",
        ], BLACK)
        game.play(0, 0)
        game.play(7, 0)
        self.assertTrue(game.game_over)
        self.assertEqual(game.counts(), (16, 0))
        self.assertEqual(game.winner(), BLACK)
        with self.assertRaises(ValueError):
            game.play(3, 3)  # no moves accepted after the game ends

    def test_draw_detection(self):
        game = make_game([
            "BBBBBBBB",
            "BBBBBBBB",
            "BBBBBBBB",
            "BBBBBBBB",
            "WWWWWWWW",
            "WWWWWWWW",
            "WWWWWWWW",
            "WWWWWWWW",
        ], BLACK)
        self.assertEqual(game.counts(), (32, 32))
        self.assertEqual(game.winner(), 0)


class TestPerft(unittest.TestCase):
    """Cross-check the whole rule set against published node counts.

    "Perft" counts how many distinct move paths exist from the opening to a
    given depth. These totals are well known for Othello, so matching them
    validates move generation and flipping across thousands of real
    positions at once -- far more coverage than hand-written cases. If a
    flip rule were subtly wrong, some branch would gain or lose a legal
    move and the totals would drift.

    Depths 7 (55092) and 8 (390216) also match but are too slow for a
    routine test run; they were verified separately.
    """

    EXPECTED = {1: 4, 2: 12, 3: 56, 4: 244, 5: 1396, 6: 8200}

    def perft(self, game, depth):
        if depth == 0 or game.game_over:
            return 1
        total = 0
        for move in game.legal_moves():
            child = game.copy()
            child.play(*move)
            total += self.perft(child, depth - 1)
        return total

    def test_perft_matches_published_counts(self):
        for depth, expected in sorted(self.EXPECTED.items()):
            with self.subTest(depth=depth):
                self.assertEqual(self.perft(OthelloGame(), depth), expected)


class TestCopy(unittest.TestCase):

    def test_copy_is_independent(self):
        """Mutating a copy must not touch the original -- search code
        depends on this."""
        game = OthelloGame()
        clone = game.copy()
        clone.play(2, 3)
        self.assertEqual(game.counts(), (2, 2))
        self.assertEqual(game.current_player, BLACK)
        self.assertEqual(clone.counts(), (4, 1))


class TestRandomSelfPlay(unittest.TestCase):

    def test_full_random_games_run_to_completion(self):
        """Smoke test: 20 games of uniformly random play must all reach a
        clean game-over state while keeping every invariant intact."""
        for seed in range(20):
            rng = random.Random(seed)
            game = OthelloGame()
            plies = 0
            while not game.game_over:
                moves = game.legal_moves()
                # The engine must never leave the current player stuck:
                # if the game isn't over, the player to move has a move.
                self.assertTrue(moves, f"seed {seed}: stuck with no moves")
                before = sum(game.counts())
                game.play(*rng.choice(moves))
                after = sum(game.counts())
                # Each move adds exactly one disc; flips only recolor.
                self.assertEqual(after, before + 1, f"seed {seed}")
                plies += 1
                self.assertLessEqual(plies, 60, f"seed {seed}: runaway game")

            # Final state sanity: board only holds valid values, and
            # neither player has a legal move left.
            self.assertTrue(np.isin(game.board, [EMPTY, BLACK, WHITE]).all())
            self.assertEqual(game.legal_moves(BLACK), [], f"seed {seed}")
            self.assertEqual(game.legal_moves(WHITE), [], f"seed {seed}")
            black, white = game.counts()
            self.assertLessEqual(black + white, 64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
