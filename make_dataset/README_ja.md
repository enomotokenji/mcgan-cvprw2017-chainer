# 重要ファイル
* `crop_rgb-nir.py`
* `make_training_datalist.py`

# おまけファイル
* `tif2png_rgb-nir.py`
* `feature_space_visualizer.py`

# 作業フロー
1. 多バンドのGeoTIFFデータからクロップされたRGBとNIRのPNGデータに変換 (crop_rgb-nir)
1. `make_training_datalist.py` でファイル名リストを作成
1. `make_clouds.py`で雲画像を生成
1. `color_correct.py`でテストデータを色補正

# 使い方
## `crop_rgb-nir.py`
* tifファイルがあるディレクトリとクロップされたpngが保存されるディレクトリを指定する
* 出力先のディレクトリには`RGB/`と`NIR/`が自動的に作成され，それぞれの中にクロップされたrgb画像とnir画像が保存される
* クロップするサイズを指定．デフォルトは256 x 256 pix
* ファイル名を指定しなければディレクトリ下にあるtifファイル全てに対して並列処理が行われる

```
python crop_rgb-nir.py -i <path to input dir> -o <path to output dir> --filename <filename> -s <size>
```
or
```
python crop_rgb-nir.py -i <path to input dir> -o <path to output dir> -s <size>
```

## `tif2png_rgb-nir.py`
* tifファイルがあるディレクトリとpngが保存されるディレクトリを指定する
* 出力先のディレクトリには`RGB/`と`NIR/`が自動的に作成され，それぞれの中にrgb画像とnir画像が保存される
* ファイル名を指定しなければディレクトリ下にあるtifファイル全てに対して並列処理が行われる

```
python tif2png_rgb-nir.py -i <path to input dir> -o <path to output dir> --filename <filename>
```
or
```
python tif2png_rgb-nir.py -i <path to input dir> -o <path to output dir>
```

## `make_training_datalist.py`
* RGB画像があるディレクトリもしくは`filename_feature.pkl`へのパスを指定する
* 全画像からAlexNetのfc7層の中間特徴量を抽出し，ファイル名とペアにして保存（`filename_feature.pkl`）．
* 中間特徴量に対し，t-SNEをし，2次元に次元削減する．
* 2次元特徴空間をグリッドに分割し，各グリッドから均等にデータを選択．学習用のファイル名リストを作成，保存．

```
python make_training_datalist.py -i <path to input dir or filename_feature.pkl> -o <path to output dir> -n_d <num of training data> -n_g <square of num of grids>
```

## `feature_space_visualizer.py`
* `make_training_datalist.py`によって生成された`filename_feature.pkl`から，2次元特徴空間を可視化．

```
python feature_space_visualizer.py -i <path to `filename_feature.pkl`> -o <path to output file> -n_g <square of num of grids>
```

## `make_clouds.py`
* 疑似雲画像を作成
* `crop_rgb-nir.py`で作成された`RGB/`, `NIR/`と同じ階層に`CLOUD/`を作って、そこに保存することを推奨

```
python make_clouds.py -n <the number of cloud images> -o <path to output dir>
```

## `color_correct.py`
* テストデータの色補正
```
python color_correct.py --in_dir <path to input dir> --out_dir <path to output dir> --grey_world --stretch
```