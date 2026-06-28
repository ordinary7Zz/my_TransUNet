# TransUNet 推理与评估运行指南

---

## 1. infer_one.sh — 单张图像推理

```bash
python infer_one.py \
  --ckpt ./model/TU_Synapse224/TU_pretrain_R50-ViT-B_16_skip3_epo10_bs2_224/epoch_9.pth \
  --img_path /path/to/your/image.jpg \
  --out_path ./inference/output_mask.png \
  --img_size 224 \
  --num_classes 2 \
  --vit_name R50-ViT-B_16 \
  --n_skip 3
```

- `--ckpt`：模型权重；`--img_path`：输入图像；`--out_path`：输出掩码
- `--vit_name`、`--n_skip`、`--img_size`、`--num_classes` 需与训练时一致

---

## 2. infer_dir.sh — 目录批量推理

```bash
python infer_dir.py \
  --ckpt ./model/TU_Synapse224/TU_pretrain_R50-ViT-B_16_skip3_epo10_bs2_224/epoch_9.pth \
  --img_dir /path/to/input_images \
  --out_dir ./inference/results \
  --img_size 224 \
  --num_classes 2 \
  --vit_name R50-ViT-B_16 \
  --n_skip 3
```

- `--img_dir`：图像目录；`--out_dir`：输出目录（掩码文件名与输入一致）
- 输出掩码为原图大小，单通道 PNG，前景=255

---

## 3. run_eval_multi.sh — 多数据集评估

```bash
python eval_2d_datasets.py \
  --ckpt /path/to/epoch_9.pth \
  --dataset_name "TN3K" \
  --img_dir /path/to/test/images \
  --mask_dir /path/to/test/masks \
  --img_size 224 \
  --num_classes 2 \
  --vit_name "R50-ViT-B_16" \
  --n_skip 3 \
  --output_dir ./eval_results/
```

- `--ckpt`：模型权重；`--dataset_name`：数据集名称；`--img_dir`/`--mask_dir`：图像与标注目录
- 图像和标注通过文件名 stem 自动匹配（支持 png/jpg/jpeg）
- 结果保存为 `<dataset_name>_metrics.json`，包含 Dice 和 HD95 指标及 95% 置信区间
- 可复制多个 `python eval_2d_datasets.py ...` 块评估多个数据集
