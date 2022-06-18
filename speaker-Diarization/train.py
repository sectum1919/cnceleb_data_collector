import tensorflow as tf
from numpy.random import RandomState
import numpy as np
from common import *
import os
import pandas as pd
import cv2

batch_size = 512
epoch = 80000


def process_frame(stringin):
    if stringin == "":
        return
    strcrop = stringin.split(":")
    frame = 0
    frame += int(strcrop[0]) * 25 * 60 * 60
    frame += int(strcrop[1]) * 25 * 60
    frame += int(strcrop[2]) * 25
    frame += int(strcrop[3])
    return frame


def load_array(sync_dir, speaker_dir, label_dir):
    if not os.path.exists(speaker_dir):
        return -1, -1
    with open(sync_dir) as f:
        s_lines = f.readlines()
    with open(speaker_dir) as f:
        w_lines = f.readlines()
    path = sync_dir.replace(config.SyncNet_result_dir, config.video_base_dir).replace(".txt", ".mp4")
    if not os.path.exists(path):
        path = path.replace(".mp4", ".MP4")
    cap = cv2.VideoCapture(path)
    frames_num = int(cap.get(7))
    w_pair = w_lines[0].strip().split('\t')
    w_lines = w_lines[1:]
    length = 25 / float(w_pair[0])
    lap = float(w_pair[1]) * length
    confs = []  # 置信度数组
    for frame in range(frames_num):
        confs.append(-7)  # 没有通过人脸识别的置信度置为-7
    for s_line in s_lines[1:]:
        frame_id = int((s_line.strip().split('\t'))[0])
        confs[frame_id] = float((s_line.strip().split('\t'))[1])

    data = []
    temp = 0
    conf_per_length = []
    n = 0
    for i in range(len(confs)):
        conf = confs[i]
        temp += conf
        if i % (length / 5) == 0 and i != 0:
            conf_per_length.append(temp / (length / 5))
            temp = 0

    for n in range(len(w_lines)):
        now_frame = int(n * (length - lap))
        now_seg_id = int(now_frame / 5)
        speaker_data = float(w_lines[n].strip())
        if (now_seg_id + 5) > len(conf_per_length):
            break
        else:
            for i in range(int(length / 5)):
                if (now_seg_id + i) < len(conf_per_length):
                    data.append([speaker_data, conf_per_length[now_seg_id + i]])


    truelable = pd.read_csv(label_dir, encoding="utf-16-le", sep='\t')
    truelable = truelable[["入点", "出点"]].values
    for row_index in range(len(truelable)):
        for col_index in range(len(truelable[row_index])):
            truelable[row_index][col_index] = process_frame(truelable[row_index][col_index])
    valset = []
    for pair in truelable:
        valset += [i for i in range(pair[0], pair[1])]
    valset = set(valset)

    labels = []
    all_is_1 = True
    for i in range(frames_num):
        if i not in valset:
            all_is_1 = False

        if i % (length / 5) == 0 and i != 0:
            if all_is_1:
                labels.append(1)
            else:
                labels.append(0)
            all_is_1 = True

    labels_result = []
    i = 0
    for n in range(len(data)):
        if len(data) == len(labels_result):
            break
        labels_result.append(labels[i])
        if n % (length / 5) == 0 and n != 0:
            i -= 2
        else:
            i += 1
    return data, labels_result


def load_data():
    sync = config.SyncNet_result_dir
    speaker = config.output_dir
    csv = config.video_base_dir
    POIS = os.listdir(sync)
    train_data = []
    labels = []
    i = 0
    for POI in POIS:
        POI_categories = os.listdir(os.path.join(sync, POI))

        # 遍历文件下所有文件
        for category in POI_categories:
            category_txt = os.path.join(sync, POI, category)
            for root, dirs, files in os.walk(category_txt):
                for file in files:
                    sync_dir = os.path.join(root, file)
                    speaker_dir = sync_dir.replace(sync, speaker)
                    label_dir = sync_dir.replace(sync, csv).replace(".txt", ".csv")
                    data, label = load_array(sync_dir, speaker_dir, label_dir)
                    if data == -1:
                        continue
                    # print(sync_dir)
                    for data_n in data:
                        train_data.append(data_n)
                    for label_n in label:
                        labels.append(label_n)
        i += 1
        if i == 24:
            break
    return train_data, labels


def main():
    with tf.name_scope('graph_logistic') as scope:
        w1 = tf.Variable(tf.random_normal([2, 10], stddev=1, seed=1), name='w1')
        w2 = tf.Variable(tf.random_normal([10, 1], stddev=1, seed=1), name='w2')

        b1 = tf.Variable(tf.random_normal([1, 10], stddev=1, seed=1), name='b1')
        b2 = tf.Variable(tf.random_normal([1, 1], stddev=1, seed=1), name='b2')

        x = tf.placeholder(tf.float32, shape=[None, 2], name='x-input')
        y_target = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

        a = tf.add(tf.matmul(x, w1, name='a'), b1)
        y = tf.add(tf.matmul(tf.tanh(a), w2, name='y'), b2)
        y_pre = tf.sigmoid(y, name='y-pre')

        # 交叉熵
        cross_entropy = - tf.reduce_mean(y_target * tf.log(tf.clip_by_value(y_pre, 1e-10, 1.0)) +
                                         (1 - y_target) * tf.log(tf.clip_by_value((1 - y_pre), 1e-10, 1.0)))

        train_step = tf.train.AdamOptimizer(0.0001).minimize((cross_entropy))

    with tf.Session() as sess:
        writer = tf.summary.FileWriter("log/", sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver(max_to_keep=1)
        X, Y = load_data()
        dataset_size = len(X)
        print(len(X))
        print(len(Y))
        X = np.array(X).reshape(dataset_size, 2)
        Y = np.array(Y).reshape(dataset_size, 1)
        for i in range(epoch + 1):
            start = (i * batch_size) % dataset_size
            end = min(start + batch_size, dataset_size)
            # x_temp = np.array(X[start:end]).reshape(batch_size, 2)
            # y_temp = np.array(Y[start:end]).reshape(batch_size, 1)
            # temp = temp.transpose(0, 2, 1)
            sess.run(train_step, feed_dict={x: X[start:end], y_target: Y[start:end]})

            if i % 1000 == 0:
                # 每隔一段时间计算在所有数据上的损失函数并输出
                total_cross_entropy = sess.run(
                    cross_entropy, feed_dict={x: X, y_target: Y})
                total_w1 = sess.run(w1)
                total_b1 = sess.run(b1)
                total_w2 = sess.run(w2)
                total_b2 = sess.run(b2)
                saver.save(sess, 'ckpt/logistic.ckpt', global_step=i)
                print("After %d training steps(s), cross entropy on all data is %g" % (i, total_cross_entropy))
                print('w1=', total_w1, ',b1=', total_b1)
                print('w2=', total_w2, ',b2=', total_b2)


if __name__ == '__main__':
    main()
