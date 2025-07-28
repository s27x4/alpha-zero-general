import numpy as np
from otrio.OtrioGame import OtrioGame


def test_canonical_form_color_rotation_two_player():
    g = OtrioGame()
    b = g.getInitBoard()
    canon = g.getCanonicalForm(b, -1)
    expected = b.copy()
    expected[:, :g.SIZES] *= -1
    expected = np.roll(expected, -2, axis=0)
    assert np.array_equal(canon, expected)


def test_canonical_form_color_rotation_multi_player():
    g = OtrioGame(n_players=3)
    b = g.getInitBoard()
    canon = g.getCanonicalForm(b, 3)
    expected = b.copy()
    expected[:, :g.SIZES] *= 3
    expected = np.roll(expected, -2, axis=0)
    assert np.array_equal(canon, expected)
