#!/bin/bash
python infer_dir.py \
  --ckpt ./my_model/TG_Video/epoch_4.pth \
  --img_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/sample/images \
  --out_dir ./inference/sample \
  --img_size 224 \
  --num_classes 2 \
  --vit_name R50-ViT-B_16 \
  --n_skip 3
