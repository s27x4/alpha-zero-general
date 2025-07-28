import logging
import os
import subprocess
from datetime import datetime

import coloredlogs

from Coach import Coach
from otrio.OtrioGame import OtrioGame as Game
from otrio.pytorch.NNet import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    # ─── 学習サイクル ─────────────────────
    'numIters': 1000,
    'numEps': 200,          # self‑playゲーム数 ↑
    'tempThreshold': 25,
    'updateThreshold': 0.55,
    'maxlenOfQueue': 300_000,
    'numMCTSSims': 160,     # 探索 4‑6倍
    'arenaCompare': 100,    # 評価試合 ↑
    'cpuct': 1.5,

    # ─── 盤設定 ─────────────────────────
    'board_x': 3,
    'board_y': 3,
    'action_size': 27,

    # ─── ネット＆最適化 ────────────────
    'num_channels': 256,
    'dropout': 0.3,
    'lr': 0.002,            # 初期 LR
    'batch_size': 128,
    'epochs': 5,

    # ─── スケジューラ設定（追記） ───────
    'lr_scheduler': 'cosine',   # wrap して使う
    'lr_min': 1e-4,

    # ─── TensorBoard ───────────────────
    'tb_log_dir': 'logs/otrio-v2',

    # ─── I/O ───────────────────────────
    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('o-trio-v2/temp', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 40,
})




def main():
    # --- ログディレクトリの作成と TensorBoard 起動 ---
    base_log_dir = args.tb_log_dir
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join(base_log_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    args.tb_log_dir = log_dir

    try:
        subprocess.Popen([
            'tensorboard',
            '--logdir', log_dir,
        ])
    except FileNotFoundError:
        log.warning('tensorboard コマンドが見つかりませんでした')

    log.info('Loading %s...', Game.__name__)
    g = Game()

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g,args)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
