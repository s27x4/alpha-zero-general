import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.append('../../')
from utils import *
from NeuralNet import NeuralNet
import pickle
import torch
import torch.optim as optim
from .OtrioNNet import OtrioNNet
from torch.utils.tensorboard import SummaryWriter
class NNetWrapper:
    def __init__(self, game, args):
        from utils import dotdict      # 元のユーティリティ再利用
        self.args = args
        self.board_x, self.board_y = args.board_x, args.board_y
        # --- 追加①: CUDA フラグを補完 ---
        if 'cuda' not in self.args:
            from torch import cuda
            self.args.cuda = cuda.is_available()
        self.nnet = OtrioNNet(game, self.args)
        if self.args.cuda:
            self.nnet.cuda()
        self.writer = SummaryWriter(log_dir='logs/otrio-ai')  # 好きなパス名でOK

    # train/predict/save/load は元の Wrapper をコピペ or インポート
    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters())
        global_step = 0
        for epoch in range(self.args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / self.args.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=self.args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if self.args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                self.writer.add_scalar('loss/total', total_loss.item(), global_step)
                self.writer.add_scalar('loss/policy', l_pi.item(), global_step)
                self.writer.add_scalar('loss/value',  l_v.item(), global_step)
                global_step += 1
        self.writer.flush()

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        if self.args.cuda: board = board.contiguous().cuda()
        board = torch.as_tensor(board, dtype=torch.float32).unsqueeze(0)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)
            pi = torch.exp(pi)  # 一度だけ指数を計算
            pi = pi / pi.sum()
        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi.data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if self.args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'], strict=False)

