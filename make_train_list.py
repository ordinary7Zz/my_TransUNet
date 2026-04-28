import os

npz_dir = r'/mnt/wangbd8/workspace/ThyroidAgent/TransUNet/datasets/dataset_4/train'  # 改成你实际的npz目录
list_dir = r'/mnt/wangbd8/workspace/ThyroidAgent/TransUNet/datasets/dataset_4'
os.makedirs(list_dir, exist_ok=True)

npz_files = sorted([f for f in os.listdir(npz_dir) if f.endswith('.npz')])
stems = [os.path.splitext(f)[0] for f in npz_files]

with open(os.path.join(list_dir, 'train.txt'), 'w') as f:
    for s in stems:
        f.write(s + '\n')