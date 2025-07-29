import numpy as np
import torch
from utils import dotdict
from otrio.OtrioGame import OtrioGame
from otrio.pytorch.OtrioNNet import OtrioNNet


def test_otrio_nnet_forward_shape():
    game = OtrioGame()
    args = dotdict({'num_channels': 8, 'dropout': 0})
    nnet = OtrioNNet(game, args)
    board = torch.tensor(game.getInitBoard(), dtype=torch.float32).unsqueeze(0)
    pi, v = nnet(board)
    assert pi.shape == (1, game.getActionSize())
    assert v.shape == (1, 1)
