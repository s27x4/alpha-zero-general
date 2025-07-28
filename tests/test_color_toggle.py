import numpy as np
from otrio.OtrioGame import OtrioGame


def test_two_player_toggle():
    g = OtrioGame()
    b = g.getInitBoard(); p = 1
    a0 = np.where(g.getValidMoves(b, p))[0][0]
    b, p = g.getNextState(b, p, a0)
    assert g.next_color[0] == 1  # トグル済
