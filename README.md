# FreeU: Free Lunch in Diffusion U-Net（CVPR24）
## FreeUの説明

追加コストや追加学習なしでStableDiffusionの生成品質を向上させる手法。事前実験に基づいて提案され、さまざまなモデル（SD1.5やSDXL）や拡張手法（DreamBoothなど）で効果を発揮している。

StableDiffusionのデノイズ過程では、高周波成分が生成品質の不安定性の原因となる可能性があり、FreeUはUNetのスキップ接続とバックボーンの特徴マップから来る成分のバランスを調整することでこれを改善する。

具体的には、UNetのデコーダーモジュールにおいて、高周波成分が過剰に詳細な画像を生成するのを防ぐため、スキップ接続の高周波成分とバックボーンの低周波成分の寄与を再調整します。この手法は推論時に適用され、追加の学習を必要とせず、テキストから画像、画像から画像、テキストから動画などの多様なタスクに対応している。

CVPRに採択された理由として、UNetが用いられるさまざまなモデルで機能する点、学習を必要とせずに精度を上げることができる点が評価されていると考えられる。

## 実装したコードの説明

1. requirements.txtを参考に環境構築
2. `python main.py`で実行
StableDiffusionのFreeUありとなしで画像を生成し、保存

### main.py
プロンプトを入力すると、FreeUを使用する場合と使用しない場合の2つの画像を生成します。
プロンプトやパラメータ等は直書きしました。UNetのbackboneの重みb、skip-connectionの重みs、コードに直書きしてあります。
一番単純な実装です。


### utils.py
論文と著者実装を参考に、可能な限り単純化しました.


Fourier_filter関数
- 入力テンソルに対してフーリエ変換を行い、高周波成分をフィルタリングしてから逆フーリエ変換を行う。

register_free_upblock2d関数

- モデルのアップサンプリングブロックをFreeU用に再定義し、各ブロックに高周波フィルタリングとスケーリングを適用。

register_free_crossattn_upblock2d関数:

- モデルのクロスアテンションアップサンプリングブロックをFreeU用に再定義し、各ブロックに高周波フィルタリングとスケーリングを適用。