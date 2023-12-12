## Installation
Follow https://github.com/fundamentalvision/BEVFormer/blob/master/docs/install.md to prepare the environment.

## Pre-training
```
./tools/dist_train.sh projects/configs/bevformer/occ_bev_sweep2.py 8
```

## Fine-tuning

change UniScene/BEVFormer/projects/configs/bevformer/bevformer_base.py Line 249: load_from = 'UniScene/work_dirs/occ_bev_sweep2/epoch_24.pth' to the pre-trained model from UniScene

```
cd UniScene/BEVFormer
./tools/dist_train.sh ./projects/configs/bevformer/bevformer_base.py 8
```
## Testing

```
cd UniScene/BEVFormer
./tools/dist_test.sh ./projects/configs/bevformer/bevformer_base.py ./work_dirs/bevformer_base/epoch_24.pth 8
```
