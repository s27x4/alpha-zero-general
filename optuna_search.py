import copy, optuna, os, numpy as np
from otrio.OtrioGame import OtrioGame as Game
from otrio.pytorch.NNet import NNetWrapper as NNet
from Coach import Coach
from utils import *

base_args = dotdict({
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

def objective(trial):
    # ---- ハイパラ提案 ----
    t_args = base_args
    t_args.lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    t_args.numMCTSSims  = trial.suggest_int('sims', 80, 300, step=20)
    t_args.cpuct        = trial.suggest_float('cpuct', 1.0, 2.5)
    t_args.num_channels = trial.suggest_categorical('channels', [128, 192, 256])

    # ---- 1 trial 用に Coach を立ち上げ ----
    game = Game()
    nnet = NNet(game, t_args)
    coach = Coach(game, nnet, t_args)
    coach._trial_id = trial.number           # 例保存用 ID

    win_rate = coach.quick_train_eval(iters=3)
    trial.report(win_rate, step=0)
    # Optuna は minimize がデフォルトなので -win_rate
    return -win_rate

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(),
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
    study.optimize(objective, n_trials=20, timeout=60*60)  # 1 時間上限

    print("Best params:", study.best_params, "  WinRate:", -study.best_value)
    study.trials_dataframe().to_csv("optuna_history.csv", index=False)
