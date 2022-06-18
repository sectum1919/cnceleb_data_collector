# -*-coding:utf-8-*-
import tensorflow as tf
import numpy as np
from common import *
import os
from train import load_array, process_frame
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.python import pywrap_tensorflow


def data_clean(raw_result):
    count = 0
    clean_result = raw_result

    # 若出现前后都超过2个0的孤立的1，则置为0
    for i in range(len(clean_result) - 4):
        if clean_result[i + 2] == 1:
            if clean_result[i] == 0 and clean_result[i + 1] == 0 and clean_result[i + 3] == 0 and clean_result[
                i + 4] == 0:
                clean_result[i + 2] = 0
    # 0的个数出现小于2的则连接起来
    for i in range(len(clean_result) - 1):
        if clean_result[i] == 0:
            count += 1
            if clean_result[i + 1] == 1 and count <= 2:
                for j in range(i - count, i + 1):
                    clean_result[j] = 1
                count = 0
                continue
            if count > 2:
                count = 0

    # 每一段的头尾各删掉一帧
    candidate = []
    while i < len(clean_result):
        if clean_result[i] == 1:
            start = i
            while i < len(clean_result) and clean_result[i] == 1:
                candidate.append(clean_result[i])
                i += 1
            if len(candidate) >= 8:
                clean_result[start] = 0
                clean_result[start + 1] = 0
                clean_result[start + len(candidate) - 1] = 0
                clean_result[start + len(candidate) - 2] = 0
            elif 4 <= len(candidate) <= 8:
                clean_result[start] = 0
                clean_result[start + len(candidate) - 1] = 0
            candidate = []
        else:
            i += 1

    # 再做一遍除去孤立值
    for i in range(len(clean_result) - 3):
        if clean_result[i + 6] == 1:
            if clean_result[i] == 0 and clean_result[i + 1] == 0 and clean_result[i + 2] == 0 and clean_result[
                i + 4] == 0 and clean_result[i + 5] == 0 and clean_result[
                i + 6] == 0:  # and clean_result[i+7] == 0 and clean_result[i+8] == 0:
                clean_result[i + 3] = 0
    return clean_result


def evaluate():
    with tf.Session() as sess:
        sigmoid_graph = tf.train.import_meta_graph(config.graph_dir)
        sigmoid_graph.restore(sess, tf.train.latest_checkpoint(config.model_dir))
        graph = tf.get_default_graph()

        x_input = graph.get_tensor_by_name('graph_logistic/x-input:0')
        y_pre = graph.get_tensor_by_name('graph_logistic/y-pre:0')
        X, Y = load_data()

        dataset_size = len(X)
        X = np.array(X).reshape(dataset_size, 2)
        Y = np.array(Y).reshape(dataset_size, 1)

        output = sess.run(y_pre, feed_dict={x_input: X})
        output = np.squeeze(output)
        Y = np.squeeze(Y)
        output = output.tolist()
        Y = Y.tolist()

        pre_result = []  # 五帧一个值，按照时间顺序排列，无重叠帧
        real_result = []

        n = 0
        for i in range(len(output)):
            if n >= len(pre_result):
                pre_result.append(output[i])
            else:
                pre_result[n] += output[i]
            if i % 5 == 0 and i != 0:
                n -= 2
            else:
                n += 1
        n = 0
        for i in range(len(Y)):
            if n >= len(real_result):
                real_result.append(Y[i])
            else:
                real_result[n] += Y[i]
            if i % 5 == 0 and i != 0:
                n -= 2
            else:
                n += 1

        for i in range(len(real_result)):
            if real_result[i] >= 2:
                real_result[i] = 1
            else:
                real_result[i] = 0

        correct_num = 0
        valid_num = 0
        valid_correct_num = 0
        for i in range(len(pre_result)):
            if pre_result[i] < 1:  # 不要调高！已经验证过高的值在降低recall的同时并没有提高ACC
                pre_result[i] = 0
            else:
                pre_result[i] = 1
        # print(pre_result[1000:1300])
        # pre_result = data_clean(pre_result)
        # print(pre_result[1000:1300])
        for i in range(len(pre_result)):
            if pre_result[i] == real_result[i]:
                correct_num += 1
        accuracy = correct_num / len(pre_result)
        print('\033[96maccuracy = ' + str(accuracy) + '\033[0m')

        for i in range(len(pre_result)):
            if real_result[i] == 1:
                valid_num += 1
                if pre_result[i] == 1:
                    valid_correct_num += 1
        recall = valid_correct_num / valid_num
        print('\033[96mrecall = ' + str(recall) + '\033[0m')


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
        for category in POI_categories:
            if category == 'interview':
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
    return train_data, labels


def visualize():
    X, _ = load_data()
    X = np.squeeze(X)
    speaker = X[:, 0]
    speaker = speaker[:30000]
    sync = X[:, 1]
    sync = sync[:30000]
    fig = plt.figure()
    # 将画图窗口分成1行1列，选择第一块区域作子图
    ax1 = fig.add_subplot(1, 1, 1)
    # 设置标题
    ax1.set_title('Result Analysis')
    # 设置横坐标名称
    ax1.set_xlabel('speaker')
    # 设置纵坐标名称
    ax1.set_ylabel('sync')
    # 画散点图
    ax1.scatter(speaker, sync, s=0.1, c='k', marker='.')
    # 保存
    plt.savefig('speaker_sync.jpg')


def evaluate_1():
    X, Y = load_data()
    X = np.squeeze(X)
    Y = np.squeeze(Y)
    Y = Y.tolist()
    speaker = X[:, 0]
    # print(speaker[1000:1200])
    sync = X[:, 1]
    output = []
    if config.merge_method == 1:
        for i in range(len(speaker)):
            if speaker[i] >= 0.8:
                output.append(1)
            elif 0.5 <= speaker[i] < 0.8:
                output.append(speaker[i])
            else:
                output.append(0)

        for i in range(len(sync)):
            output[i] += float(sync[i] / 10)
    elif config.merge_method == 0:
        for i in range(len(speaker)):
            if speaker[i] > 0.75:
                output.append(1)
            else:
                output.append(0)

        for i in range(len(sync)):
            if sync[i] > 5.1:
                output[i] = 1

    pre_result = []  # 五帧一个值，按照时间顺序排列，无重叠帧
    real_result =[]
    n = 0
    for i in range(len(output)):
        if n >= len(pre_result):
            pre_result.append(output[i])
        else:
            if pre_result[n] != 1:
                pre_result[n] += output[i]
                if config.merge_method == 1:
                    pre_result[n] /= 2
        if i % 5 == 0 and i != 0:
            n -= 2
        else:
            n += 1

    n = 0
    for i in range(len(Y)):
        if n >= len(real_result):
            real_result.append(Y[i])
        else:
            real_result[n] += Y[i]
        if i % 5 == 0 and i != 0:
            n -= 2
        else:
            n += 1

    for i in range(len(real_result)):
        if real_result[i] != 0:
            real_result[i] = 1
        else:
            real_result[i] = 0

    correct_num = 0
    pre_P_num = 0
    valid_num = 0
    valid_correct_num = 0
    if config.merge_method == 0:
        for i in range(len(pre_result)):
            if pre_result[i] < 1:  # 不要调高！已经验证过高的值在降低recall的同时并没有提高ACC
                pre_result[i] = 0
            else:
                pre_result[i] = 1
    elif config.merge_method == 1:
        for i in range(len(pre_result)):
            if pre_result[i] < 1.2:  # 不要调高！已经验证过高的值在降低recall的同时并没有提高ACC
                pre_result[i] = 0
            else:
                pre_result[i] = 1
    pre_result = data_clean(pre_result)

    for i in range(len(pre_result)):
        if pre_result[i] == 1:
            pre_P_num += 1
            if real_result[i] == 1:
                correct_num += 1
    accuracy = correct_num / pre_P_num
    print('\033[96mprecision = ' + str(accuracy) + '\033[0m')
    print(correct_num)
    print(pre_P_num)

    for i in range(len(pre_result)):
        if real_result[i] == 1:
            valid_num += 1
            if pre_result[i] == 1:
                valid_correct_num += 1
    recall = valid_correct_num / valid_num
    print('\033[96mrecall = ' + str(recall) + '\033[0m')
    print(valid_num)


if __name__ == '__main__':
    evaluate_1()
