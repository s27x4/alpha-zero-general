import numpy as np
import pytest

from otrio.OtrioGame import OtrioGame


def action_idx(size, row, col):
    return size * 9 + row * 3 + col


def test_get_valid_moves_initial_all_ones():
    game = OtrioGame()
    board = game.getInitBoard()
    valid = game.getValidMoves(board, 1)
    assert valid.shape[0] == game.getActionSize()
    assert valid.sum() == game.getActionSize()


def test_get_valid_moves_after_move_becomes_zero():
    game = OtrioGame()
    board = game.getInitBoard()
    action = action_idx(0, 1, 1)
    board, player = game.getNextState(board, 1, action)
    valid = game.getValidMoves(board, player)
    assert valid[action] == 0
    assert valid.sum() == game.getActionSize() - 1


def test_get_game_ended_row_win():
    game = OtrioGame()
    board = game.getInitBoard()
    board[0, 0] = np.array([1, 1, 1])
    assert game.getGameEnded(board, 1) == 1
    assert game.getGameEnded(board, -1) == -1


def test_get_game_ended_tower_win():
    game = OtrioGame()
    board = game.getInitBoard()
    board[:, 0, 0] = 1
    assert game.getGameEnded(board, 1) == 1
    assert game.getGameEnded(board, -1) == -1


def test_get_game_ended_draw():
    game = OtrioGame()
    board = game.getInitBoard()
    board[:3] = np.array([
        [[1, -1, 1],
         [1, -1, -1],
         [-1, 1, -1]],
        [[-1, 1, -1],
         [1, -1, -1],
         [1, -1, 1]],
        [[1, -1, 1],
         [1, -1, 1],
         [-1, 1, -1]]
    ], dtype=np.int8)
    assert game.getGameEnded(board, 1) == 1e-4
    assert game.getGameEnded(board, -1) == 1e-4


def test_reserve_counts_decrement():
    game = OtrioGame()
    board = game.getInitBoard()
    size = 0
    actions = [
        action_idx(size, 0, 0),
        action_idx(size, 0, 1),
        action_idx(size, 0, 2),
    ]
    for act in actions:
        board, _ = game.getNextState(board, 1, act)
    assert board[game.SIZES + size, 0, 0] == 0
    with pytest.raises(AssertionError):
        game.getNextState(board, 1, action_idx(size, 1, 0))


def test_get_game_ended_draw_reserves_empty():
    game = OtrioGame()
    board = game.getInitBoard()
    board[3:] = 0
    assert game.getGameEnded(board, 1) == 1e-4
    assert game.getGameEnded(board, -1) == 1e-4
