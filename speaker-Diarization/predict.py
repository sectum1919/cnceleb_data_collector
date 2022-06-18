# -*-coding:utf-8-*-
import tensorflow as tf
import numpy as np
from common import *
import os
import cv2


def predict():
    # print("load model")
    # with tf.Session() as sess:
    #     sigmoid_graph = tf.train.import_meta_graph(config.graph_dir)
    #     sigmoid_graph.restore(sess, tf.train.latest_checkpoint(config.model_dir))
    #     graph = tf.get_default_graph()
    #
    #     x_input = graph.get_tensor_by_name('graph_logistic/x-input:0')
    #     y_pre = graph.get_tensor_by_name('graph_logistic/y-pre:0')
    starting_POI = ''
    print("start at:", starting_POI)
    sync_path = config.SyncNet_result_dir
    speaker_path = config.output_dir
    POIS = os.listdir(sync_path)
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
        POIS = os.listdir(config.video_base_dir)

    if starting_POI != '':
        while POIS[0] != starting_POI:
            print("\033[94mskipping {}\033[0m".format(POIS[0]))
            POIS.pop(0)
        # 从starting_POI的下一个开始
        print("\033[94mskipping {}\033[0m".format(POIS[0]))
        POIS.pop(0) 
    for POI in POIS:
        POI_categories = os.listdir(os.path.join(sync_path, POI))
        # 遍历文件下所有文件
        for category in POI_categories:
            category_txt = os.path.join(sync_path, POI, category)
            out_put_dirname = os.path.join(config.output_dir, POI, category)
            if not os.path.exists(out_put_dirname):
                os.makedirs(out_put_dirname)
            for root, dirs, files in os.walk(category_txt):
                for file in files:
                    sync_dir = os.path.join(root, file)
                    speaker_dir = sync_dir.replace(sync_path, speaker_path)
                    if not os.path.exists(speaker_dir):
                        continue
                    data = load_array(sync_dir, speaker_dir)
                    if data == -1:
                        continue
                    output_dir = sync_dir.replace(sync_path, config.output_dir)
                    print(sync_dir)
                    data = load_array(sync_dir, speaker_dir)
                    data = np.squeeze(data)
                    output = []
                    speaker = data[:, 0]
                    sync = data[:, 1]
                    if config.merge_method == 1:
                        for i in range(len(speaker)):
                            if speaker[i] >= 0.75:
                                output.append(1)
                            elif 0.5 <= speaker[i] < 0.75:
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
                    final_result = []
                    for i in range(len(pre_result)):
                        if pre_result[i] == 1:
                            now_frame = i * 5
                            if len(final_result) >= 1 and (final_result[-1][0] + final_result[-1][1]) >= now_frame - 10:
                                start_frame = final_result[-1][0]
                                slice_length = final_result[-1][1]
                                final_result.pop()
                                slice_length += 5
                            else:
                                slice_length = 5
                                start_frame = i * 5
                            final_result.append([int(start_frame), int(slice_length)])

                    for i in range(len(final_result)):
                        if final_result[i][1] >= 30:
                            final_result[i][0] += 5
                            final_result[i][1] -= 8

                    with open(output_dir, "w") as f:
                        for line in final_result:
                            if line[1] < 20:
                                continue
                            f.write(form_convert(line[0]) + '\t' + form_convert(line[1]) + '\n')


def load_array(sync_dir, speaker_dir):
    if not os.path.exists(speaker_dir):
        return -1
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
    return data


def form_convert(frame_id):
    h = int(frame_id / 90000)
    rest = frame_id % 90000
    m = int(rest / 1500)
    rest = rest % 1500
    s = int(rest / 25)
    lf = rest % 25
    return "{:0>2d}:{:0>2d}:{:0>2d}:{:0>2d}".format(h, m, s, lf)


def data_clean(raw_result):
    count = 0
    clean_result = raw_result

    # 若出现前后都超过4个0的孤立的1，则置为0
    for i in range(len(clean_result) - 8):
        if clean_result[i + 4] == 1:
            if clean_result[i] == 0 and clean_result[i + 1] == 0 and clean_result[i + 2] == 0 and clean_result[
                i + 3] == 0 and clean_result[i + 5] == 0 and clean_result[
                i + 6] == 0 and clean_result[i + 6] == 0 and clean_result[i + 8] == 0:
                clean_result[i + 3] = 0

    # # 0的个数出现小于2的则连接起来
    # for i in range(len(clean_result) - 1):
    #     if clean_result[i] == 0:
    #         count += 1
    #         if clean_result[i + 1] == 1 and count <= 2:
    #             for j in range(i - count, i + 1):
    #                 clean_result[j] = 1
    #             count = 0
    #             continue
    #         if count > 2:
    #             count = 0

    # # 每一段的头尾各删掉一帧
    # candidate = []
    # while i < len(clean_result):
    #     if clean_result[i] == 1:
    #         start = i
    #         while i < len(clean_result) and clean_result[i] == 1:
    #             candidate.append(clean_result[i])
    #             i += 1
    #         if len(candidate) >= 8:
    #             clean_result[start] = 0
    #             clean_result[start + 1] = 0
    #             clean_result[start + len(candidate) - 1] = 0
    #             clean_result[start + len(candidate) - 2] = 0
    #         elif 4 <= len(candidate) <= 8:
    #             clean_result[start] = 0
    #             clean_result[start + len(candidate) - 1] = 0
    #         candidate = []
    #     else:
    #         i += 1

    # # 再做一遍除去孤立值
    # for i in range(len(clean_result) - 6):
    #     if clean_result[i + 3] == 1:
    #         if clean_result[i] == 0 and clean_result[i + 1] == 0 and clean_result[i + 2] == 0 and clean_result[
    #             i + 4] == 0 and clean_result[i + 5] == 0 and clean_result[
    #             i + 6] == 0:  # and clean_result[i+7] == 0 and clean_result[i+8] == 0:
    #             clean_result[i + 3] = 0
    return clean_result


if __name__ == '__main__':
    predict()
    print("complete")
