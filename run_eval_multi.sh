#!/bin/bash

# 修改为你的模型权重路径，比如训练时生成的 epoch_9.pth 或 best_model.pth
CKPT="/mnt/wangbd8/workspace/ThyroidAgent/TransUNet/model/TU_Synapse224/TU_pretrain_R50-ViT-B_16_skip3_epo10_bs2_224/epoch_9.pth"
OUTPUT_DIR="./eval_results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
LOG_PATH="$OUTPUT_DIR/summary.log"

# 示例：在多个测试集上评估

python eval_2d_datasets.py \
  --ckpt "$CKPT" \
  --dataset_name "TN3K" \
  --img_dir "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test/images" \
  --mask_dir "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test/masks" \
  --img_size 224 \
  --num_classes 2 \
  --vit_name "R50-ViT-B_16" \
  --n_skip 3 \
  --output_dir "$OUTPUT_DIR" | tee "$LOG_PATH"

python eval_2d_datasets.py \
  --ckpt "$CKPT" \
  --dataset_name "DDTI" \
  --img_dir "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/DDTI/test/images" \
  --mask_dir "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/DDTI/test/masks" \
  --img_size 224 \
  --num_classes 2 \
  --vit_name "R50-ViT-B_16" \
  --n_skip 3 \
  --output_dir "$OUTPUT_DIR" | tee -a "$LOG_PATH"

python eval_2d_datasets.py \
  --ckpt "$CKPT" \
  --dataset_name "ThyroidXL" \
  --img_dir "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/test/images" \
  --mask_dir "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/test/masks" \
  --img_size 224 \
  --num_classes 2 \
  --vit_name "R50-ViT-B_16" \
  --n_skip 3 \
  --output_dir "$OUTPUT_DIR" | tee -a "$LOG_PATH"

python eval_2d_datasets.py \
  --ckpt "$CKPT" \
  --dataset_name "PKTN" \
  --img_dir "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/PKTN/test/images" \
  --mask_dir "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/PKTN/test/masks" \
  --img_size 224 \
  --num_classes 2 \
  --vit_name "R50-ViT-B_16" \
  --n_skip 3 \
  --output_dir "$OUTPUT_DIR" | tee -a "$LOG_PATH"

python eval_2d_datasets.py \
  --ckpt "$CKPT" \
  --dataset_name "TN5K" \
  --img_dir "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/test/images" \
  --mask_dir "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/test/masks" \
  --img_size 224 \
  --num_classes 2 \
  --vit_name "R50-ViT-B_16" \
  --n_skip 3 \
  --output_dir "$OUTPUT_DIR" | tee -a "$LOG_PATH"