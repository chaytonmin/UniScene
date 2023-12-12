

## Preparing Dataset

1. Organize your folder structure as below：

   Downloading [annotations.json](https://drive.google.com/file/d/1c2rXvO1pel6goEeMqaB8FsXjbDGUPF1a/view?usp=drive_link)

```
UniScene
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

   or download from [gts](https://drive.google.com/file/d/1S0lmvo2XwUJp1Iz8iy8nmOwYWHJ8nI9t/view)

```
python tools/creat_binary_occ_labels.py --dataroot ./data/nuscenes/ --save_path ./data/nuscenes/ --num_sweeps 2
```

3. Generate the info files for training and validation:

```
python tools/create_data.py occ --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag occ --version v1.0-trainval --canbus ./data --occ-path ./data/nuscenes
```
