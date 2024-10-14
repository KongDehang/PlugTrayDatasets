# -*- coding: utf-8 -*-
"""
将数据集划分为训练集，验证集，测试集
"""
import os
import random
import shutil


# 创建保存数据的文件夹
def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


def split_data(img_dir, label_dir):
    random.seed(1)  # 随机种子
    # 1.确定原图片数据集路径
    datasetimg_dir = img_dir
    # 确定原label数据集路径
    datasetlabel_dir = label_dir

    # 2.确定数据集划分后保存的路径
    split_dir = os.path.join(".", "dataset")
    train_dir = os.path.join(split_dir, "train")
    valid_dir = os.path.join(split_dir, "valid")
    test_dir = os.path.join(split_dir, "test")
    dir_list = [train_dir, valid_dir, test_dir]
    image_label = ['images', 'labels']

    for i in range(len(dir_list)):
        for j in range(len(image_label)):
            makedir(os.path.join(dir_list[i], image_label[j]))

    # 3.确定将数据集划分为训练集，验证集，测试集的比例
    train_pct = 0.8
    valid_pct = 0.1
    test_pct = 0.1
    # 4.划分
    imgs = os.listdir(datasetimg_dir)  # 展示目标文件夹下所有的文件名
    imgs = list(filter(lambda x: x.endswith('.jpg'), imgs))  # 取到所有以.png结尾的文件，如果改了图片格式，这里需要修改
    random.shuffle(imgs)  # 乱序路径
    img_count = len(imgs)  # 计算图片数量
    train_point = int(img_count * train_pct)  # 0:train_pct
    valid_point = int(img_count * (train_pct + valid_pct))  # train_pct:valid_pct
    for i in range(img_count):
        if i < train_point:  # 保存0-train_point的图片到训练集
            out_dir = os.path.join(train_dir, 'images')
            label_out_dir = os.path.join(train_dir, 'labels')

        elif i < valid_point:  # 保存train_point-valid_point的图片到验证集
            out_dir = os.path.join(valid_dir, 'images')
            label_out_dir = os.path.join(valid_dir, 'labels')
        else:  # 保存test_point-结束的图片到测试集
            out_dir = os.path.join(test_dir, 'images')
            label_out_dir = os.path.join(test_dir, 'labels')

        target_path = os.path.join(out_dir, imgs[i])  # 指定目标保存路径
        src_path = os.path.join(datasetimg_dir, imgs[i])  # 指定目标原图像路径
        label_target_path = os.path.join(label_out_dir, imgs[i][0:-4] + '.txt')
        label_src_path = os.path.join(datasetlabel_dir, imgs[i][0:-4] + '.txt')
        shutil.copy(src_path, target_path)  # 复制图片
        shutil.copy(label_src_path, label_target_path)  # 复制txt

    print('train:{}, valid:{}, test:{}'.format(train_point, valid_point - train_point,
                                               img_count - valid_point))


if __name__ == "__main__":
    img_dir = r'E:\ADataset\all_labeled\images'
    label_dir = r'E:\ADataset\all_labeled\labels'
    split_data(img_dir, label_dir)
