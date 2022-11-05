# Donut demo 
## File Tree
S3(既に存在していることを想定しているもの)
`{path-s3}`: Topディレクトリ
`{path-s3}/result` 結果の保存。
`{path-s3}/dataset` 学習に用いるデータセットの保存。データセットの形式については以下の通り。


## Versions

### 1.0
- [x] ローカルでのcpu学習実験の実装
- [x] solvargを用いたgpu学習の実装

## Dataset
データセットの準備は以下のスクリプトで行う。
```
python prepare_invoice_dataset.py
```
オプションの指定は以下の通り。
`--path-invoice-tables-s3` : データセット作成の元となるテーブルデータのパス **https://github.com/C-FO/ai-lab/tree/master/invoice　により生成されたテーブルを想定**
`--save-dir-s3` : データセット保存先のs3パス

## Training
学習は以下のスクリプトで行う。
cpuでの実験は以下
```
python train --config config/train_cord_custom_cpu.yaml
```
gpuでの学習はこれ
```
sol -vvv run  -d --root . --cmd 'python sol.py  --train' --num-gpu 
```
config/train_cord_custom.yamlが学習の設定ファイルとして読み込まれる。

##TODO
- [ ] ./config/train_cord_custom.yaml内のnum_workerによるエラー
num_worker > 1 でシェアメモリーが足りないエラーが出る。
- [ ] batch_sizeが少なすぎる**
batch_size >= 2　でOOM しょうがないのか？？
- [ ] validation_step, test_stepの実装
- [ ] MLFlowなどのログツールの実装
- [ ] exp_version関連の管理
- [ ] 学習項目の追加
- [ ] validation_stepに関してサイズを決めるなどの拡張性を持たせる。 
ref:utils.py line37