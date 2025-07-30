# AlphaZero Otrio 版

このリポジトリは [suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general) をフォークし、ボードゲーム **Otrio** を学習対象にするための実装を追加したものです。自己対戦型強化学習により Otrio のプレイヤーを訓練できます。Otrio 用のゲームロジックやニューラルネットワークのコードは `otrio/` ディレクトリに配置されています。

基本的な構成はオリジナルと同様で、`Game.py` と `NeuralNet.py` を継承することで他のゲームにも対応可能です。詳細な設定や学習手順は [otrio/README.md](otrio/README.md) を参照してください。

## 使い方

`main.py` を実行すると Otrio の学習が始まります。

```bash
python main.py
```

各種パラメータは `main.py` 内の `args` 辞書で設定できます。

### Docker を利用した環境構築
NVIDIA GPU を使用する場合は [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) を利用すると簡単に環境を構築できます。
以下のスクリプトを実行すると Jupyter コンテナが起動します。
```
./setup_env.sh
```
起動後、別のターミナルで次のコマンドを実行すると学習を開始できます。
```
docker exec -ti pytorch_notebook python main.py
```

### Conda 環境
Python 3.11 を利用した conda 環境 `otrio` 用の `otrio_env.yaml` を追加しました。
次のコマンドで環境を構築できます。
```bash
conda env create -f otrio_env.yaml
```
詳しい使い方や学習手順については [otrio/README.md](otrio/README.md) も参照してください。

### Experiments
以下はオリジナルリポジトリでの Othello 学習例です。6x6 盤で 80 イテレーション、各イテレーション 100 エピソード、1 手あたり 25 回の MCTS を実行した結果、約 3 日で学習が完了しました。学習済みモデルは ```pretrained_models/othello/pytorch/``` にあります。
![alt tag](https://github.com/suragnair/alpha-zero-general/raw/master/pretrained_models/6x6.png)

アルゴリズムの概要は [こちら](https://github.com/suragnair/alpha-zero-general/raw/master/pretrained_models/writeup.pdf) を参照してください。

### Citation

If you found this work useful, feel free to cite it as

```
@misc{thakoor2016learning,
  title={Learning to play othello without human knowledge},
  author={Thakoor, Shantanu and Nair, Surag and Jhunjhunwala, Megha},
  year={2016},
  publisher={Stanford University, Final Project Report}
}
```

### Contributing
While the current code is fairly functional, we could benefit from the following contributions:
* Game logic files for more games that follow the specifications in ```Game.py```, along with their neural networks
* Neural networks in other frameworks
* Pre-trained models for different game configurations
* An asynchronous version of the code- parallel processes for self-play, neural net training and model comparison. 
* Asynchronous MCTS as described in the paper

Some extensions have been implented [here](https://github.com/kevaday/alphazero-general).

### Contributors and Credits
* [Shantanu Thakoor](https://github.com/ShantanuThakoor) and [Megha Jhunjhunwala](https://github.com/jjw-megha) helped with core design and implementation.
* [Shantanu Kumar](https://github.com/SourKream) contributed TensorFlow and Keras models for Othello.
* [Evgeny Tyurin](https://github.com/evg-tyurin) contributed rules and a trained model for TicTacToe.
* [MBoss](https://github.com/1424667164) contributed rules and a model for GoBang.
* [Jernej Habjan](https://github.com/JernejHabjan) contributed RTS game.
* [Adam Lawson](https://github.com/goshawk22) contributed rules and a trained model for 3D TicTacToe.
* [Carlos Aguayo](https://github.com/carlos-aguayo) contributed rules and a trained model for Dots and Boxes along with a [JavaScript implementation](https://github.com/carlos-aguayo/carlos-aguayo.github.io/tree/master/alphazero).
* [Robert Ronan](https://github.com/rlronan) contributed rules for Santorini.
* [Plamen Totev](https://github.com/plamentotev) contributed Go Text Protocol player for Othello.

Note: Chainer and TensorFlow v1 versions have been removed but can be found prior to commit [2ad461c](https://github.com/suragnair/alpha-zero-general/tree/2ad461c393ecf446e76f6694b613e394b8eb652f).
