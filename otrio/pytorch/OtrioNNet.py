import torch.nn as nn
import torch.nn.functional as F
import torch

class OtrioNNet(nn.Module):
    def __init__(self, game, args):
        super().__init__()
        self.args = args
        self.c, self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        # ▼ 入力チャネル数を 3 に
        self.conv1 = nn.Conv2d(self.c, self.args.num_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(self.args.num_channels, self.args.num_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(self.args.num_channels, self.args.num_channels, 3, padding=1)
        self.conv4 = nn.Conv2d(self.args.num_channels, self.args.num_channels, 3, padding=1)

        self.bn1 = nn.BatchNorm2d(self.args.num_channels)
        self.bn2 = nn.BatchNorm2d(self.args.num_channels)
        self.bn3 = nn.BatchNorm2d(self.args.num_channels)
        self.bn4 = nn.BatchNorm2d(self.args.num_channels)

        self.fc1 = nn.Linear(self.args.num_channels * self.board_x * self.board_y, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        # ▼ ポリシー出力 27、価値出力 1
        self.fc3 = nn.Linear(512, self.action_size)
        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        # s shape: (batch, 3, 3, 3) → (batch, 3, 3, 3)
        s = F.relu(self.bn1(self.conv1(s)))
        s = F.relu(self.bn2(self.conv2(s)))
        s = F.relu(self.bn3(self.conv3(s)))
        s = F.relu(self.bn4(self.conv4(s)))
        s = s.view(-1, self.args.num_channels * self.board_x * self.board_y)

        s = F.relu(self.fc_bn1(self.fc1(s)))
        s = F.dropout(s, p=self.args.dropout, training=self.training)
        s = F.relu(self.fc_bn2(self.fc2(s)))

        pi = self.fc3(s)          # (batch, 27)
        v  = torch.tanh(self.fc4(s))  # (batch, 1)
        return F.log_softmax(pi, dim=1), v
