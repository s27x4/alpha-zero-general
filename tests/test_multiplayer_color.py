import numpy as np
from otrio.OtrioGame import OtrioGame


def test_three_player_fixed_colors():
    g = OtrioGame(n_players=3)
    assert g.COLORS == 3
    assert g.player_colors == [[0], [1], [2]]
    b = g.getInitBoard()
    p = 1
    a = np.where(g.getValidMoves(b, p))[0][0]
    b, p = g.getNextState(b, p, a)
    assert g.get_current_color(b, 1) == 0  # 変化しない


def test_four_player_fixed_colors():
    g = OtrioGame(n_players=4)
    assert g.COLORS == 4
    assert g.player_colors == [[0], [1], [2], [3]]
    b = g.getInitBoard()
    p = 1
    a = np.where(g.getValidMoves(b, p))[0][0]
    b, p = g.getNextState(b, p, a)
    assert g.get_current_color(b, 1) == 0
