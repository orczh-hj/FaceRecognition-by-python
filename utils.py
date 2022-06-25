"""
功能：储存辅助变量、功能
时间：2022年5月16日14:30:49
作者：陈子含
"""
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn import neighbors
import os, joblib, datetime, sys
from glob import glob
import cv2 as cv
import face_recognition as face

"路径"
# 解释器路径
ip_path = os.path.dirname(os.path.realpath(sys.executable))
# 项目路径
p_path = os.path.dirname(os.path.realpath(sys.argv[0]))

"常量"
SCALE = 1.2
MAX_NUM = 100
IMAGE_SIZE = (256, 256)
THRESHOLD = (0.9, 0.6)
assert os.path.exists(p_path + '/data/haarcascade_frontalface_alt2.xml'), \
    './data/haarcascade_frontalface_alt2.xml不存在，请检查！'
face_cascade = cv.CascadeClassifier(p_path + '/data/haarcascade_frontalface_alt2.xml')


def new_id(people):
    """新增人脸时，为该人脸分配id"""
    ids = people.values()
    for i in range(len(ids)):
        if i not in ids:
            return i
    return len(ids)


def putText_ch(img, text, pos, color=(0, 0, 255), text_size=20):
    """
    图像上放置中文
    img: BGR格式的ndarray图像
    text: 中文或英文
    pos: 左上角所在位置
    color: BGR颜色
    text_size: 字体高度（像素）
    -> BGR格式的ndarray图像
    """
    assert isinstance(img, np.ndarray)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    fontStyle = ImageFont.truetype("msyh.ttc", text_size, encoding="utf-8")
    draw.text(pos, text, color, font=fontStyle)
    return np.asarray(img)


def face_train(people):
    """
    对人脸数据库中的数据进行训练
    :param people: 字典，键是人名，值是人名对应的id
    """
    knn = neighbors.KNeighborsClassifier()
    data = []
    for p, l in people.items():
        if os.path.exists(p_path + f'/face_data/{p}/codes.txt'):
            temp = np.loadtxt(p_path + f'/face_data/{p}/codes.txt')
            cods = list(temp)
        else:
            fig_paths = glob(p_path + f'/face_data/{p}/figure/*')
            if not fig_paths:
                shutil.rmtree(p_path + f'/face_data/{p}/')
                continue
            cods = []
            for fig_path in fig_paths:
                print(f'\r{fig_path}', end='')
                fig = Image.open(fig_path)
                fig = np.asarray(fig)[:, :, ::-1]
                faces = face_cascade.detectMultiScale(fig)
                locs = list(map(lambda x: (x[1], x[0] + x[2], x[1] + x[3], x[0]), faces))
                cods += (face.face_encodings(fig, locs, model='small'))
            print('')
            if len(cods) > 0:
                temp = np.stack(cods)
                np.savetxt(p_path + f'/face_data/{p}/codes.txt', temp)
        label = [l] * len(cods)
        data += list(zip(cods, label))
    np.random.shuffle(data)
    X, Y = zip(*data)
    knn.fit(X, Y)
    model_name = 'model_'+datetime.datetime.now().strftime('%y%m%d_%H%M')+'.model'
    knn.name = model_name
    knn.people = people
    joblib.dump(knn, p_path + f'/model/{model_name}')
    return knn


if __name__ == '__main__':
    print('ok')



