

## Preparing Dataset

1. Organize your folder structure as below：

   Downloading annotations.json from https://drive.google.com/drive/folders/1JObO75iTA2Ge5fa8D3BWC8R7yIG8VhrP?usp=share_link&pli=1

```
Occ-BEV
├── projects/
├── tools/
├── ckpts/
│   ├── r101_dcn_fcos3d_pretrain.pth
├── data/
│   ├── can_bus/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/     
│   │   ├── sweeps/     
│   │   ├── v1.0-trainval/
│   │   ├── gts/
│   │   └── annotations.json
```

2. Generate the binary geometry occupancy labels to gts

```
python tools/creat_binary_occ_labels.py --dataroot ./data/nuscenes/ --save_path ./data/nuscenes/ --num_sweeps 2
```

3. Generate the info files for training and validation:

```
python tools/create_data.py occ --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag occ --version v1.0-trainval --canbus ./data --occ-path ./data/nuscenes
```
