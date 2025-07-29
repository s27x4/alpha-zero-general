import numpy as np
from otrio.OtrioGame import OtrioGame


def action_idx(size, row, col):
    return size * 9 + row * 3 + col


def test_reserve_decrements():
    g = OtrioGame()
    b = g.getInitBoard()
    b2, _ = g.getNextState(b, 1, 0)
    assert b2[0, g.SIZES, 0, 0] == g.PIECES_PER_SIZE - 1


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
    board[0, 0, 0] = np.array([1, 1, 1])
    assert game.getGameEnded(board, 1) == 1
    assert game.getGameEnded(board, -1) == -1


def test_get_game_ended_tower_win():
    game = OtrioGame()
    board = game.getInitBoard()
    board[0, :, 0, 0] = 1
    assert game.getGameEnded(board, 1) == 1
    assert game.getGameEnded(board, -1) == -1


def test_get_game_ended_draw():
    game = OtrioGame()
    board = game.getInitBoard()
    pattern = np.array([
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
    board[0, :game.SIZES] = pattern
    board[1, :game.SIZES] = pattern
    assert game.getGameEnded(board, 1) == 1e-4
    assert game.getGameEnded(board, -1) == 1e-4


def test_draw_by_reserve_exhaustion():
    g = OtrioGame()
    b = g.getInitBoard()
    p = 1
    total = g.PIECES_PER_SIZE * g.SIZES * g.COLORS * g.n_players
    for _ in range(total):
        a = np.where(g.getValidMoves(b, p) == 1)[0][0]
        b, p = g.getNextState(b, p, a)
    assert g.getGameEnded(b, p) == 1e-4
