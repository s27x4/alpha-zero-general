import logging
import os
import subprocess
from datetime import datetime

import coloredlogs

from Coach import Coach
from otrio.OtrioGame import OtrioGame as Game
from otrio.pytorch.NNet import NNetWrapper as nn
from utils import *
from uuid import uuid4
log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    # ─── 学習サイクル ─────────────────────
    'numIters': 1000,          # 大枠そのまま
    'numEps': 256,             # ←200→256 でデータ多め
    'tempThreshold': 25,
    'updateThreshold': 0.55,   # いったん据え置き
    'maxlenOfQueue': 400_000,  # バッファ拡張して古い自己対戦も保持
    'numMCTSSims': 200,        # ★Optuna 推奨値
    'arenaCompare': 50,        # 100→50 で評価高速化
    'cpuct': 2.2,              # ★Optuna 推奨値

    # ─── 盤設定 ─────────────────────────
    'board_x': 3,
    'board_y': 3,
    'action_size': 27,

    # ─── ネット＆最適化 ────────────────
    'num_channels': 192,       # ★Optuna 推奨値
    'dropout': 0.2,            # 0.3→0.2 少し抑えめ
    'lr': 6.6e-4,              # ★Optuna 推奨値
    'batch_size': 256,         # 128→256 GPU メモリ許すなら倍に
    'epochs': 8,               # 5→8 で 1iter あたり学習深め

    # ─── スケジューラ設定 ───────────────
    'lr_scheduler': 'cosine',  # そのまま
    'lr_min': 1e-5,            # 最低 LR も一段下げておく

    # ─── TensorBoard ───────────────────
    'tb_log_dir': 'logs/otrio-v2',

    # ─── I/O ───────────────────────────
    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('o-trio-v2/temp', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 50,  # 40→50 queue 拡張に合わせる
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
