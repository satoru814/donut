# Donut demo 

## Preparation

### Dataset
データセットの準備は以下のスクリプトで行う。
```
python prepare_invoice_dataset.py
```
### Training
学習は以下のスクリプトで行う。
cpuでの実験は以下
```
python train --config config/train_cord_custom_cpu.yaml
```
gpuでの学習はこれ
```
sol -vvv run  -d --root . --cmd 'python sol.py  --train' --num-gpu 
```
この時train_cord_custom.yamlがcofnfiファイルとして読み込まれる。

##TODO
- ./config/train_cord_custom.yaml内のnum_workerによるエラー
num_worker > 1 でシェアメモリーが足りないエラーが出る。
- batch_sizeが少なすぎる**
batch_size >= 2　でOOM しょうがないのか？？
- validation_step, test_stepの実装
- MLFlowなどのログツールの実装
- exp_version関連の管理