#!/bin/bash

# 修改为你的模型权重路径，比如训练时生成的 epoch_9.pth 或 best_model.pth
CKPT="/mnt/wangbd8/workspace/ThyroidAgent/TransUNet/my_model/Nodule/epoch_49.pth"
OUTPUT_DIR="./eval_results/Nodule/$(date +%Y%m%d_%H%M%S)"
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
