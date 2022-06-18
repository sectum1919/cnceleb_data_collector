#encoding=UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np
import os
import re
import sys
import xlrd
import requests
from scipy import misc
import imageio
import tensorflow as tf
from facenet_code import facenet
from facenet_code.align import detect_face

# 超参数，需要几张图片
need = 10

image_size = 160
margin = 32
gpu_memory_fraction = 0.7
model = "/work9/cchen/project/CNCeleb/cnceleb_data_collector/getpoi/facenet_code/20180402-114759"


def check(name, picdir):
    files = os.listdir(picdir + '/' + name)
    image_files = []
    i = 0
    for file in files:
        i += 1
        image_files.append(picdir + '/' + name + '/' + file)
    if i == 10:
        return
    images = load_and_align_data(name, image_files, image_size, margin, gpu_memory_fraction, picdir)
    print("read images")
    with tf.Graph().as_default():

        with tf.Session() as sess:

            # 如果后面loadmodel报错，就把load加到这里
            facenet.load_model(model)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")  # 网络输入
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")  # 输出
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb = sess.run(embeddings, feed_dict=feed_dict)

            nrof_images = emb.shape[0]  # 图片张数

            # Print distance matrix
            matrix = np.zeros((nrof_images, nrof_images))

            for i in range(nrof_images):
                for j in range(nrof_images):
                    dist = np.sqrt(np.sum(np.square(np.subtract(emb[i, :], emb[j, :]))))  # 计算欧式距离
                    matrix[i][j] = dist

            dict = {}
            for i in range(nrof_images - 1):
                dict[str(i)] = np.average(matrix[i])
            out = 0
            list = sorted(dict.items(), key=lambda x: x[1], reverse=True)

            for k in list:
                fn = picdir + '/' + name + '/' + name + '-' + str(k[0]) + '.jpg'
                if os.path.exists(fn):
                    os.remove(fn)
                out += 1
                if nrof_images - out == need:
                    break


def load_and_align_data(name, image_paths, image_size, margin, gpu_memory_fraction, picdir):
    # mtcnn 要用到的3个参数
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    print('Creating networks and loading parameters')
    # 加载mtcnn模型
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    tmp_image_paths = copy.copy(image_paths)

    img_list = []
    i = 0

    filedir = picdir + '/' + name
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    # 遍历图片
    for image in tmp_image_paths:
        print(image)
        try:
            img = misc.imread(os.path.expanduser(image), mode='RGB')
            # img = imageio.imread(os.path.expanduser(image))
        except:
            print("this file is broken, remove ", image)
            image_paths.remove(image)
            os.remove(image)
            continue
        img_size = np.asarray(img.shape)[0:2]
        try:
            bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold,
                                                        factor)  # mtcnn人脸检测返回人脸边框shape(边框数,5)，第二维前4个数是边框坐标，第5个数是score
        except IndexError:
            print("this file is broken, remove ", image)
            image_paths.remove(image)
            os.remove(image)
            continue
        # 如果检测到多于一个人脸
        if len(bounding_boxes) != 1:
            image_paths.remove(image)
            os.remove(image)
            print("can't detect face, remove ", image)
            continue
        os.rename(image, filedir + '/' + name + '-' + str(i) + ".jpg")
        i += 1
        det = np.squeeze(bounding_boxes[0, 0:4])  # 删除维度为1的那一维，即mtcnn返回的边框数那一维（第一维）
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)  # 坐标往下移一点。mtcnn检测出来的只有人脸部分，扩展其范围以包含更多信息
        bb[1] = np.maximum(det[1] - margin / 2, 0)  # 左移
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])  # 上移
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])  # 右移
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]  # 从输入图片中裁剪处人脸部分
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')  # resize为facenet网络输入大小160x160
        prewhitened = facenet.prewhiten(aligned)  # 图片的标准化处理，类似tf.image.per_img_standard()
        img_list.append(prewhitened)
    images = np.stack(img_list)  # 将几张图片堆叠起来
    return images


def download(keyword, page, savedir):
    needurl = 'http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word=' + keyword + "&pn=" + str(page * 30)
    print("\033[94m%s\033[0m" % (keyword))
    try:
        html = requests.get(needurl)
    except:
        print("unknown error")
        html = requests.get(needurl, timeout=10)

    url = re.findall('"objURL":"(.*?)",', html.text, re.S)
    i = 0
    max = 30

    filedir = savedir + '/' + keyword
    if not os.path.exists(filedir):
        os.makedirs(filedir)

    for one in url:
        print('downloading ' + keyword + ': ' + str(i) + r'/' + str(max))
        try:
            pic = requests.get(one, timeout=10)
        except requests.exceptions.ConnectionError:
            print('X error X')
            continue
        except requests.exceptions.ReadTimeout:
            print('X error X')
            continue
        except:
            print('X error X')
            continue

        dir = filedir + '/' + keyword + str(i + page * 60) + '.jpg'
        fp = open(dir, 'wb')
        fp.write(pic.content)
        fp.close()
        i += 1
        if (i > max):
            break


if __name__ == '__main__':
    # 爬虫，需要arg1：excel文件位置，arg2：图片储存位置
    with open(sys.argv[1]) as f:
        lines = f.readlines()
    namelist = [ line.strip() for line in lines ]
    
    for word in namelist:
        download(word, 0, sys.argv[2])
        download(word, 1, sys.argv[2])

    # 数据清洗，需要arg2：图片储存位置
    # Load the model
    #facenet.load_model(model)
    for dirpath, dirnames, filenames in os.walk(sys.argv[2]):
        for dirname in dirnames:
            check(dirname, sys.argv[2])
