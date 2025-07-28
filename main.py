import logging

import coloredlogs

from Coach import Coach
from otrio.OtrioGame import OtrioGame as Game
from otrio.pytorch.NNet import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    # ── 学習サイクル ─────────────────────────
    'numIters': 1000,
    'numEps': 100,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200_000,
    'numMCTSSims': 25,
    'arenaCompare': 40,
    'cpuct': 1,

    # ── Otrio 盤設定 ────────────────────────
    'board_x': 3,
    'board_y': 3,
    'action_size': 27,

    # ── ネットワーク＆最適化 ────────────────
    'num_channels': 128,   # ★ ← 追加（64〜512 でお好み）
    'dropout': 0.3,
    'lr': 0.001,
    'batch_size': 64,
    'epochs': 10,

    # TensorBoard ログ出力先
    'tb_log_dir': 'logs/otrio-ai',

    # ── I/O ────────────────────────────────
    'checkpoint': './temp/',
    'load_model': True,
    'load_folder_file': ('temp', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})



def main():
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
