import numpy as np
from otrio.OtrioGame import OtrioGame


def test_two_player_toggle():
    g = OtrioGame()
    b = g.getInitBoard(); p = 1
    a0 = np.where(g.getValidMoves(b, p))[0][0]
    b, p = g.getNextState(b, p, a0)
    assert g.get_current_color(b, 1) == 1  # トグル済


def test_get_current_color_canonical():
    g = OtrioGame()
    b = g.getInitBoard(); p = 1
    a0 = np.where(g.getValidMoves(b, p))[0][0]
    b, p = g.getNextState(b, p, a0)
    # p == -1 の状態で正規化
    canon = g.getCanonicalForm(b, p)
    # canonical=True を指定すると色インデックスが補正される
    assert g.get_current_color(canon, p, canonical=True) == 0
