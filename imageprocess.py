import os
import cv2
import time
import numpy as np
import itertools as it


def cv_show(name, image):
    # 最近邻插值法缩放
    # 缩放到原来的四分之一
    img_resized = cv2.resize(image, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)  # INTER_NEAREST

    # 显示如下处理后的图像
    cv2.imshow(name, img_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cv_dered(image):
    """
    De-red the cut images（去除模板匹配裁剪后，单个穴盘格的红色框线）
    :param image: target
    :return:
    """
    # print(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (image[i, j, 2] / (float(sum(image[i, j, :]))) > 0.50 or image[i, j, 0] >= 220):
                image[i, j, :] = 255


def cv_recut(image, thresh=10):
    """
    Re-cut the anchor image（去除模板匹配裁剪后，单个穴盘格的红色框线）
    :param image: target
    :return:
    """
    # print(image.shape)
    x, y, w, h = 0, 0, image.shape[0], image.shape[1]  # 初始尺寸
    diff = int((image.shape[0] - image.shape[1]) / 2)  # 长宽差/2

    # if abs(diff) > 10:
    #     print("长宽比太大，请检查模板！")
    #     return image

    if diff > 0:
        x = thresh + abs(diff)
        y = thresh
        w = image.shape[1] - 2 * thresh
        h = image.shape[1] - 2 * thresh
    else:
        x = thresh
        y = thresh + abs(diff)
        w = image.shape[0] - 2 * thresh
        h = image.shape[0] - 2 * thresh
    region = image[x:x + w, y:y + h]
    return region


def cv_improcess(image, index=0):
    """
    Morphological Processing
    形态学处理
    :param image: target image
    :param index: default：0,
                  closed1：1,
                  closed2：2,
                  opened1：3,
                  opened2：4,
                  gradient：5,
                  tophat：6,
                  blackhat：7,
                  bitwiseXor -blackhat：8,
                  bitwiseXor -tophat：9
    :return: processed image
    """

    processed_image = image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 定义矩形结构元素

    if index == 0:
        # B, G, R = cv2.split(img_test2)  # 获取红色通道 （此处为灰度图像，所以不用此功能）
        # img = R
        _, RedThresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)  # 设定红色通道阈值160（阈值影响开闭运算效果）

        # OpenCV定义的结构矩形元素
        eroded = cv2.erode(RedThresh, kernel)  # 腐蚀图像
        processed_image = cv2.dilate(eroded, kernel)  # 膨胀图像
    elif index == 1:
        processed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)  # 闭运算1
    elif index == 2:
        processed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=3)  # 闭运算2
    elif index == 3:
        processed_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)  # 开运算1
    elif index == 4:
        processed_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=3)  # 开运算2
    elif index == 5:
        processed_image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)  # 梯度
    elif index == 6:
        processed_image = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)  # 顶帽运算
    elif index == 7:
        processed_image = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)  # 黒帽运算
    elif index == 8:
        BLACKHAT_img = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)  # 黒帽运算
        processed_image = cv2.bitwise_xor(image, BLACKHAT_img)
    elif index == 9:
        TOPHAT_img = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)  # 顶帽运算
        processed_image = cv2.bitwise_xor(image, TOPHAT_img)

    return processed_image


#
# img = cv2.imread("Resources/test/Pic_1.jpg")
# template = cv2.imread("Resources/test/template.jpeg")  # 转换为灰度图片
# gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 读取测试图片并将其转化为灰度图片
# gray2 = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
# h, w = template.shape[:2]  # 匹配#TM_SQ-DIFF匹配方法，归一化的方法更好用
# res = cv2.matchTemplate(gray1, gray2, cv2.TM_SQDIFF)  # 得到极值坐标
# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
# top_left = min_loc
# bottom_right = (top_left[0] + w, top_left[1] + h)  # 画出标记点
# cv2.rectangle(img, top_left, bottom_right, 255, 2)
# cv_show("img", img)


def py_nms(dets, thresh):
    """Pure Python NMS baseline."""
    # x1、y1、x2、y2、以及score赋值
    # （x1、y1）（x2、y2）为box的左上和右下角标
    x1 = dets[:, 0]
    y1 = dets[:, 1]  # 左上角的坐标值

    x2 = dets[:, 2]
    y2 = dets[:, 3]  # 右下角的阈值

    scores = dets[:, 4]
    # 每一个候选框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # order是按照score降序排序的，从大到小
    order = scores.argsort()[::-1]
    # print("order:",order)

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # 计算当前概率最大矩形框与其他矩形框的相交框的坐标，会用到numpy的broadcast机制，得到的是向量
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # 计算相交框的面积,注意矩形框不相交时w或h算出来会是负数，用0代替
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 计算重叠度IOU：重叠面积/（面积1+面积2-重叠面积）
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 找到重叠度不高于阈值的矩形框索引
        inds = np.where(ovr <= thresh)[0]
        # print("inds:",inds)
        # 将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
        order = order[inds + 1]
    return keep


def template(img_gray, template_img, template_threshold):
    """
    img_gray:待检测的灰度图片格式
    template_img:模板小图，也是灰度化了
    template_threshold:模板匹配的置信度
    """

    h, w = template_img.shape[:2]  # 获取模板的高和宽
    res = cv2.matchTemplate(img_gray, template_img, cv2.TM_CCOEFF_NORMED)  # 模板匹配的方式 cv2.TM_CCOEFF_NORMED
    start_time = time.time()
    loc = np.where(res >= template_threshold)  # 大于模板阈值的目标坐标，返回的就是矩阵的行列索引值，其中行坐标为坐标的y值，列坐标为x值
    score = res[
        res >= template_threshold]  # 大于模板阈值的目标置信度cv2.matchTemplate(img_gray, template_img, cv2.TM_CCOEFF_NORMED)[res >= template_threshold]完整写法
    # 将模板数据坐标进行处理成左上角、右下角的格式
    xmin = np.array(loc[1])  # 列坐标为x值
    ymin = np.array(loc[0])  # 横坐标为y值
    xmax = xmin + w
    ymax = ymin + h

    xmin = xmin.reshape(-1, 1)  # 变成n行1列维度
    xmax = xmax.reshape(-1, 1)  # 变成n行1列维度
    ymax = ymax.reshape(-1, 1)  # 变成n行1列维度
    ymin = ymin.reshape(-1, 1)  # 变成n行1列维度
    score = score.reshape(-1, 1)  # 变成n行1列维度

    data_hlist = []
    data_hlist.append(xmin)
    data_hlist.append(ymin)
    data_hlist.append(xmax)
    data_hlist.append(ymax)
    data_hlist.append(score)
    data_hstack = np.hstack(
        data_hlist)  # 将xmin、ymin、xmax、yamx、scores按照列进行拼接       np.hstack():在水平方向上平铺  np.vstack():在竖直方向上堆叠
    thresh = 0.2  # NMS里面的IOU交互比阈值（初始0.3）

    keep_dets = py_nms(data_hstack, thresh)
    print("nms time:", time.time() - start_time)
    dets = data_hstack[keep_dets]
    return dets


def save_result(name, file, img_rgb):
    isExists = os.path.exists("./" + str(name) + '/')
    if not isExists:  # 判断如果文件不存在,则创建
        os.makedirs("./" + str(name) + '/')
        print("./" + str(name) + '/' + "目录创建成功")

    out = str(file.split('.')[0])
    filename = "./" + name + '/%s.jpg' % (out)
    print('已匹配图片：' + filename)
    cv2.imwrite(filename, img_rgb)


def save_cut(name, file, cut, index=""):
    isExists = os.path.exists("./" + str(name) + '/')
    if not isExists:  # 判断如果文件不存在,则创建
        os.makedirs("./" + str(name) + '/')
        print("./" + str(name) + '/' + "目录创建成功")

    out = str(file.split('.')[0])
    if index != "":
        out = index
    filename = "./" + name + '/%s.jpg' % (out)
    print('已裁切图片：' + filename)
    cv2.imwrite(filename, cut)


def template_matching(template_img, image_path):
    file_walk = os.listdir(image_path)
    template_img = cv2.imread(template_img, 0)  # 模板
    cv_show("template_img", template_img)
    template_threshold = 0.17  # 模板置信度

    for file in file_walk:
        file1 = os.path.join(image_path, file)
        img_rgb = cv2.imread(file1)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        cv_show("img_gray", img_gray)

        dets = template(img_gray, template_img, template_threshold)

        index = len(os.listdir("data_processed3/cells"))
        for coord in dets:
            cv2.rectangle(img_rgb, (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3])), (0, 0, 255), 2)
            cut = img_rgb[int(coord[1]):int(coord[3]), int(coord[0]):int(coord[2])]  # 裁切坐标为（y0:y1，x0:x1）
            # cv_dered(cut)
            crop = cv_recut(cut, 12)
            save1 = save_result('data_processed3/save_pics', file, img_rgb)
            save2 = save_cut('data_processed3/pics', file, crop, str(index))
            index += 1


def cut_equally(path, x_q, y_q):
    """
    对图片进行等分裁剪
    :param path: 图片存放路径
    :param x_q: x方向平分倍数
    :param y_q: y方向平分倍数
    :return:
    """
    file_walk = os.listdir(path)

    # 指定保存图片的目录
    save_dir = r"./data_processed3/cells"
    index = len(os.listdir(save_dir))

    for file in file_walk:
        file1 = os.path.join(path, file)
        # 读取图片
        img = cv2.imread(file1)
        # # 裁剪边角
        # crop = cv_recut(img, 12)
        # 获取图片的高和宽
        height, width, _ = img.shape

        # 计算每份的大小
        block_size = int(height / y_q), int(width / x_q)

        for i in range(y_q):
            for j in range(x_q):
                # 计算裁剪的起始和结束位置
                start_y = i * block_size[0]
                end_y = (i + 1) * block_size[0]
                start_x = j * block_size[1]
                end_x = (j + 1) * block_size[1]

                # 使用numpy切片裁剪图片
                block = img[start_y:end_y, start_x:end_x]

                # 保存裁剪后的图片到指定目录
                cv2.imwrite(os.path.join(save_dir, f'{index}.jpg'), block)
                index += 1


if __name__ == "__main__":

    jpg_path = r"E:\ADataset\template_match"  # 图片存放文件夹的路径 E:\ADataset\pictures
    temp_path = r"E:\ADataset\Pic_11.jpg"
    template_matching(temp_path, jpg_path)




    # # jpg_path = "./savepicture0"  # 图片存放文件夹的路径 E:\ADataset\pictures
    # jpg_path = r"./data_processed3/pics"  # 图片存放文件夹的路径 E:\ADataset\pictures
    # file_walk = os.listdir(jpg_path)
    #
    # for file in file_walk:
    #     file1 = os.path.join(jpg_path, file)
    #     img_rgb = cv2.imread(file1)
    #     img_gray = 255 - cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    #     # img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    #     # img_gray = cv2.resize(img_gray, dsize=(1967, 2991), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
    #     # cv_show("img_gray", img_gray)
    #     template_img = cv2.imread(r"./data_processed3/temps/tem1.jpg", 0) # 模板
    #     template_img = cv2.bitwise_not(template_img)
    #     # template_img = cv2.imread(r'E:\ADataset\cell283.jpg', 0)  # 模板
    #     # cv_show("template_img", template_img)
    #     template_threshold = 0.5  # 模板置信度
    #     dets = template(img_gray, template_img, template_threshold)
    #
    #     # index = len(os.listdir('./data_processed3/cells'))
    #     index = 1255
    #     for coord in dets:
    #         cv2.rectangle(img_rgb, (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3])), (0, 0, 255), 2)
    #         cut = img_rgb[int(coord[1]):int(coord[3]), int(coord[0]):int(coord[2])]  # 裁切坐标为（y0:y1，x0:x1）
    #         # ToDo 目标检测算法
    #         # cv_dered(cut)
    #         crop = cv_recut(cut, 6)
    #         save1 = save_result('./data_processed3/save', file, img_rgb)
    #         save2 = save_cut('./data_processed3/cells', file, crop, str(index))
    #         index += 1

print("*******************************************************************************")
print('*                            已完成模板匹配任务                                *')
print("*******************************************************************************")
