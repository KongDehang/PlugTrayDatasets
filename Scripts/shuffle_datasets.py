import os
import random

# /////////////////////////////***打乱数据集图片排列顺序***/////////////////////////////////////////
# 获取图片和标签的路径
image_dir = r'E:\ADataset\all_labeled'
label_dir = r'E:\ADataset\all_labeled\labels'

# 获取所有的图片和标签文件名（不包括扩展名）
image_files = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith('.jpg')]
label_files = [os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith('.txt')]

# 确保图片和标签一一对应
assert set(image_files) == set(label_files)

# 生成一个新的随机序列
random.shuffle(image_files)

# 为每个文件分配一个新的名字
for i, old_name in enumerate(image_files):
    # new_name = str(i)
    new_name = 'img' + str(i)
    # 重命名图片文件
    os.rename(os.path.join(image_dir, old_name + '.jpg'), os.path.join(image_dir, new_name + '.jpg'))
    # 重命名对应的标签文件
    os.rename(os.path.join(label_dir, old_name + '.txt'), os.path.join(label_dir, new_name + '.txt'))
