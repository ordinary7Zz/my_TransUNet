import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# 1. 修改成你自己的路径
img_dir = '/mnt/wangbd8/workspace/DataSets/ThyroidAgent/TGVideo_PNG/train/image'
mask_dir = '/mnt/wangbd8/workspace/DataSets/ThyroidAgent/TGVideo_PNG/train/mask'

# 2. 输出到项目期望路径（相对 TransUNet）
out_dir = '/mnt/wangbd8/workspace/DataSets/ThyroidAgent/TGVideo_PNG/train/npz/npz_img'
os.makedirs(out_dir, exist_ok=True)

# 3. 收集所有图像文件名
valid_ext = ('.png', '.jpg', '.jpeg')
img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(valid_ext)])

def find_same_stem_file(directory: str, stem: str, exts=valid_ext) -> str:
    candidates = []
    for ext in exts:
        p = os.path.join(directory, stem + ext)
        if os.path.exists(p):
            candidates.append(p)
        else:
            p_upper = os.path.join(directory, stem + ext.upper())
            if os.path.exists(p_upper):
                candidates.append(p_upper)
    if len(candidates) == 0:
        raise FileNotFoundError(f"Mask not found for stem='{stem}' in '{directory}'. Expected one of: {', '.join([stem + e for e in exts])}")
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple masks found for stem='{stem}': {candidates}. Keep exactly one.")
    return candidates[0]

for fname in tqdm(img_files, desc="Converting"):
    stem = os.path.splitext(fname)[0]          # e.g. 'case0001_slice000'
    img_path = os.path.join(img_dir, fname)
    mask_path = find_same_stem_file(mask_dir, stem, valid_ext)

    # 读图像，转为单通道
    img = Image.open(img_path).convert('L')    # 'L' = 灰度
    img_np = np.array(img, dtype=np.float32)

    # 读标签（通常已经是单通道整型 mask）
    mask = Image.open(mask_path)
    mask_np = np.array(mask, dtype=np.int32)
    # 你的标签当前是 0 / 255，二分类时要映射为 0 / 1
    mask_np = (mask_np > 0).astype(np.int32)

    # 如果需要，可以这里做归一化 / 尺寸统一
    # 例如：img_np = (img_np - img_np.mean()) / (img_np.std() + 1e-8)

    # 保存为 npz
    out_path = os.path.join(out_dir, stem + '.npz')
    np.savez(out_path, image=img_np, label=mask_np)

print(f"Done. {len(img_files)} npz files saved to {out_dir}")