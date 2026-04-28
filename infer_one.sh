#!/bin/bash
python infer_one.py \
  --ckpt ./model/TU_Synapse224/TU_pretrain_R50-ViT-B_16_skip3_epo10_bs2_224/epoch_9.pth \
  --img_path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test/images/TN3K_test_0360.jpg \
  --out_path ./inference/TN3K_test_0360.png \
  --img_size 224 \
  --num_classes 2 \
  --vit_name R50-ViT-B_16 \
  --n_skip 3