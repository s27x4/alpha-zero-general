# Otrio 用 README

このフォルダには Otrio を AlphaZero 手法で学習させるための実装が入っています。以下に `main.py` での設定例、`otrio_env.yaml` の使い方、学習開始手順を説明します。

## main.py の設定例

`main.py` には学習に必要な各種パラメータを `args` 辞書として定義しています。デフォルトでは Otrio の学習が行える設定になっていますが、主な項目は次のとおりです。

```python
args = dotdict({
    'numIters': 1000,    # 学習イテレーション数
    'numEps': 100,       # 1 イテレーションあたりの自己対戦数
    'numMCTSSims': 25,   # MCTS のシミュレーション回数
    'board_x': 3,        # 盤の列数
    'board_y': 3,        # 盤の行数
    'num_channels': 128, # NN のチャネル数
    'checkpoint': './temp/',  # 重み保存先
    'load_model': True,       # 過去モデルのロード有無
})
```

必要に応じて値を変更し、学習条件を調整してください。

## otrio_env.yaml の使い方

Otrio 用の Conda 環境設定ファイル `otrio_env.yaml` がリポジトリ直下にあります。Python 3.11 ベースで必要なライブラリをインストールできます。以下のコマンドで環境を作成します。

```bash
conda env create -f otrio_env.yaml
```

作成後は次のように環境を有効化します。

```bash
conda activate otrioai
```

## 学習開始手順

1. 上記の手順で Conda 環境を構築・有効化します。
2. 本リポジトリのルートディレクトリで以下を実行します。
   
   ```bash
   python main.py
   ```
   
   `main.py` 内のパラメータに従って学習が開始されます。進捗やログは標準出力および `temp/` 以下に保存されます。

以上が Otrio の基本的な実行方法です。パラメータを変更し、さまざまな条件で学習を試してみてください。
