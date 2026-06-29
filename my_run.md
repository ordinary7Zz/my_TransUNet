# TransUNet 运行指南

---

## 0. 训练模型

### 下载预训练 ViT 权重

```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/R50-ViT-B_16.npz
mkdir -p ../model/vit_checkpoint/imagenet21k
mv R50-ViT-B_16.npz ../model/vit_checkpoint/imagenet21k/
```

### 准备数据

如果你的数据是 PNG/JPG 图像 + 标注掩码，需要先转为 `.npz` 格式：

**① 用 `prepare_npz.py` 将图像转为 npz**

修改脚本中三个路径，然后运行：

```python
img_dir  = '/path/to/raw/train/images'    # 原始图像目录
mask_dir = '/path/to/raw/train/masks'     # 标注目录（文件名与图像一致）
out_dir  = './datasets/dataset_4/train'   # npz 输出目录
```

每张图像生成一个 `.npz`，含 `image`（float32 灰度）和 `label`（int32 二值掩码 0/1）。

**② 用 `make_train_list.py` 生成文件列表**

```python
npz_dir = './datasets/dataset_4/train'
list_dir = './datasets/dataset_4'
```

生成 `train.txt`，每行一个文件名（无扩展名）。

最终目录结构：
```
./datasets/dataset_4/
├── train/
│   ├── case0001.npz
│   ├── case0002.npz
│   └── ...
└── train.txt
```

数据集可放在任意路径，训练时通过 `--root_path` 和 `--list_dir` 指定。

### 运行训练

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset Synapse \
  --root_path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/TGVideo_PNG/train/npz/npz_img \
  --list_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/TGVideo_PNG/train/npz \
  --vit_name R50-ViT-B_16 \
  --img_size 224 \
  --num_classes 2 \
  --batch_size 24 \
  --max_epochs 50 \
  --base_lr 0.0001 \
  --n_skip 3 \
  --seed 42 \
  --output_dir ./my_model/TG_Video

CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset Synapse \
  --root_path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Superimposed_experiment/dataset_4/train/npz/npz_img \
  --list_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Superimposed_experiment/dataset_4/train/npz \
  --vit_name R50-ViT-B_16 \
  --img_size 224 \
  --num_classes 2 \
  --batch_size 24 \
  --max_epochs 50 \
  --base_lr 0.0001 \
  --n_skip 3 \
  --seed 42 \
  --output_dir ./my_model/Nodule
```

关键参数：显存不足可降低 `--batch_size`（如 12 或 6），并等比降低 `--base_lr`。

> **权重保存路径**：`--output_dir` 自定义权重目录；不指定则自动拼成 `../model/TU_Synapse224/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/`。权重文件如 `epoch_149.pth`。

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
