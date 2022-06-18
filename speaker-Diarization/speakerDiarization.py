# coding:UTF-8
"""A demo script showing how to DIARIZATION ON WAV USING UIS-RNN."""

import numpy as np
import librosa
import sys
from common import *

sys.path.append('ghostvlad')
import toolkits
import model as spkModel
import os
import gc
import subprocess
from scipy.io import wavfile
from tqdm import tqdm

# ===========================================
#        Parse the argument
# ===========================================
import argparse

parser = argparse.ArgumentParser()
# set up training configuration.
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--resume', default=r'ghostvlad/pretrained/weights.h5', type=str)
parser.add_argument('--data_path', default='4persons', type=str)
# set up network configuration.
parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
parser.add_argument('--ghost_cluster', default=2, type=int)
parser.add_argument('--vlad_cluster', default=8, type=int)
parser.add_argument('--bottleneck_dim', default=512, type=int)
parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
# set up learning rate, training loss and optimizer.
parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
parser.add_argument('--test_type', default='normal', choices=['normal', 'hard', 'extend'], type=str)

global args

args = parser.parse_args()


# SAVED_MODEL_NAME = 'pretrained/saved_model.uisrnn_benchmark'

def append2dict(speakerSlice, spk_period):
    key = list(spk_period.keys())[0]
    value = list(spk_period.values())[0]
    timeDict = {}
    timeDict['start'] = int(value[0] + 0.5)
    timeDict['stop'] = int(value[1] + 0.5)
    if (key in speakerSlice):
        speakerSlice[key].append(timeDict)
    else:
        speakerSlice[key] = [timeDict]

    return speakerSlice


def arrangeResult(labels, time_spec_rate):
    lastLabel = labels[0]
    speakerSlice = {}
    j = 0
    for i, label in enumerate(labels):
        if (label == lastLabel):
            continue
        speakerSlice = append2dict(speakerSlice, {lastLabel: (time_spec_rate * j, time_spec_rate * i)})
        j = i
        lastLabel = label
    speakerSlice = append2dict(speakerSlice, {lastLabel: (time_spec_rate * j, time_spec_rate * (len(labels)))})
    return speakerSlice


def frame2time(frame_id):
    frame_id = int(frame_id)
    h = int(frame_id / 90000)
    rest = frame_id % 90000
    m = int(rest / 1500)
    rest = rest % 1500
    s = int(rest / 25)
    lf = rest % 25
    lf = lf * 40
    return "{:0>2d}:{:0>2d}:{:0>2d}.{:0>2d}".format(h, m, s, lf)


def load_wav(audio, sr):
    wav = audio
    wav = wav / max(abs(wav))
    # intervals = librosa.effects.split(wav, top_db=20)
    # wav_output = []
    # for sliced in intervals:
    #     wav_output.extend(wav[sliced[0]:sliced[1]])
    return np.array(wav)
    # return np.array(wav_output), (intervals / sr * 1000).astype(int)


def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
    linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length)  # linear spectrogram
    return linear.T


# 0                   10                  20                  30frame
# |-------------------|-------------------|-------------------|
# |-------------------|
#  |-------------------|
#   |-------------------|
#    |-------------------|
def load_data(video_dir, win_length=400, sr=16000, hop_length=160, n_fft=512, embedding_per_second=0.5,
              overlap_rate=0.5):
    audio_tmp = os.path.join(config.temp_dir, 'audio.wav')
    command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_f32le -ar 16000 %s > %s 2>&1" % (
        video_dir, audio_tmp, os.path.join(config.log_dir, "ffmpeg.log")))
    output = subprocess.call(command, shell=True, stdout=None)
    sample_rate, audio = wavfile.read(audio_tmp)

    wav = load_wav(audio, sr=sr)
    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    spec_mag = mag_T
    spec_len = sr / hop_length / embedding_per_second
    spec_hop_len = spec_len * (1 - overlap_rate)

    cur_slide = 0.0
    utterances_spec = []

    while (True):  # slide window.
        if (cur_slide + spec_len > time):
            break
        spec_mag = mag_T[:, int(cur_slide + 0.5): int(cur_slide + spec_len + 0.5)]

        # preprocessing, subtract mean, divided by time-wise var
        mu = np.mean(spec_mag, 0, keepdims=True)
        std = np.std(spec_mag, 0, keepdims=True)
        spec_mag = (spec_mag - mu) / (std + 1e-5)
        utterances_spec.append(spec_mag)

        cur_slide += spec_hop_len

    return utterances_spec


def load_models(spkModel):
    params = {'dim': (257, None, 1),
              'nfft': 512,
              'spec_len': 250,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': 5994,
              'sampling_rate': 16000,
              'normalize': True,
              }
    network_eval = spkModel.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                                   num_class=params['n_classes'],
                                                   mode='eval', args=args)
    network_eval.load_weights(args.resume, by_name=True)

    return network_eval


def single_recognition(video_path, output_dir, network_eval, embedding_per_second=0.5, overlap_rate=0.5):
    sync_path = video_path.replace(config.video_base_dir, config.SyncNet_result_dir).replace(".mp4", ".txt").replace(
        ".MP4", ".txt")
    if not os.path.exists(sync_path):
        return
    print("single_recognition: ", video_path)
    print("cal specs")
    specs = load_data(video_path, embedding_per_second=embedding_per_second, overlap_rate=overlap_rate)
    print("try to get poi vector")
    poi_vector = get_poi_vector("./temp/audio.wav", sync_path, network_eval, embedding_per_second=0.25, overlap_rate=0.5)
    print("got poi vector")
    if len(poi_vector) == 0:
        print("no poi was selected")
        return
    poi_vector = np.squeeze(poi_vector)
    feat_diff = []
    for spec in tqdm(specs):
        spec = np.expand_dims(np.expand_dims(spec, 0), -1)
        v = network_eval.predict(spec)
        v = np.squeeze(v)
        diff = np.dot(v, poi_vector) / (np.linalg.norm(v) * (np.linalg.norm(poi_vector)))  # 片段与目标的余弦距离
        feat_diff += [diff]

    #### file write ####
    length = 1 / embedding_per_second
    overlop = length * overlap_rate
    with open(output_dir, "w") as f:
        f.write(str(length) + '\t' + str(overlop) + '\n')
        for diff in feat_diff:
            f.write(str(diff) + '\n')


def single_vector(audio_path, network_eval, win_length=400, sr=16000, hop_length=160, n_fft=512,
                  embedding_per_second=2.5, overlap_rate=0.1):
    sample_rate, audio = wavfile.read(audio_path)
    wav = load_wav(audio, sr=sr)

    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    spec_mag = mag_T
    spec_len = sr / hop_length / embedding_per_second
    spec_hop_len = spec_len * (1 - overlap_rate)

    cur_slide = 0.0
    utterances_spec = []

    while (True):  # slide window.
        if (cur_slide + spec_len > time):
            break
        spec_mag = mag_T[:, int(cur_slide + 0.5): int(cur_slide + spec_len + 0.5)]

        # preprocessing, subtract mean, divided by time-wise var
        mu = np.mean(spec_mag, 0, keepdims=True)
        std = np.std(spec_mag, 0, keepdims=True)
        spec_mag = (spec_mag - mu) / (std + 1e-5)
        utterances_spec.append(spec_mag)

        cur_slide += spec_hop_len

    feats = []
    for spec in utterances_spec:
        spec = np.expand_dims(np.expand_dims(spec, 0), -1)
        v = network_eval.predict(spec)
        feats += [v]
    if len(feats) == 0:
        return []
    feat = np.mean(feats, axis=0)  # 这段音频的特征
    return feat


def get_poi_vector(audio_path, syncnet_result_path, network_eval, embedding_per_second=1.0, overlap_rate=0.6):
    audio_tmp = os.path.join(config.temp_dir, 'clip.wav')
    candidate = []
    with open(syncnet_result_path) as f:
        lines = f.readlines()
        n = 1
        avg_conf_per_second = 0
        frames = []
        while True:
            conf_in_second = []
            frames_temp = []
            if n + 100 > len(lines):
                break
            for i in range(n, n+100):
                pair = lines[i].strip().split('\t')
                frames_temp.append(pair[0])
                conf_in_second.append(pair[1])
            avg_conf_temp = np.average(np.array(conf_in_second,dtype='float_'))
            if avg_conf_temp > avg_conf_per_second:
                avg_conf_per_second = avg_conf_temp
                frames = frames_temp
            n += 1
            # if n + 100 > len(lines):
            #     break
        if len(frames) == 0:
            return []
        candidate.append(frame2time(frames[0]))
        candidate.append(frame2time(len(frames)))

    if os.path.exists(audio_tmp):
        os.remove(audio_tmp)
    command = ("sox %s %s trim %s %s" % (audio_path, audio_tmp, candidate[0], candidate[1]))
    cmd_result = subprocess.call(command, shell=True, stdout=None)
    poi_feat = single_vector(audio_tmp, network_eval, embedding_per_second=embedding_per_second,
                                overlap_rate=overlap_rate)

    return poi_feat


def main():
    # gpu configuration
    sess = toolkits.initialize_GPU(args)

    starting_POI = ''
    print("start at:", starting_POI)

    network_eval = load_models(spkModel)
    print("\033[94mall model loaded\033[0m")

    POIS = os.listdir(config.video_base_dir)
    if starting_POI != '':
        while POIS[0] != starting_POI:
            print("\033[94mskipping {}\033[0m".format(POIS[0]))
            POIS.pop(0)

    for POI in POIS:
        print("\033[96mcurrent POI: {}\033[0m".format(POI))
        POI_categories = os.listdir(os.path.join(config.video_base_dir, POI))

        # 遍历文件下所有文件
        for category in POI_categories:
            category_video = os.path.join(config.video_base_dir, POI, category)
            for root, dirs, files in os.walk(category_video):
                for file in files:
                    if file.find('.csv') > 0 or file.find('.txt') > 0 or file.find(".wav") > 0:
                        continue
                    index = file.rfind('.')
                    category_output = root.replace(config.video_base_dir, config.output_dir)
                    if not os.path.exists(category_output):
                        os.makedirs(category_output)
                    single_video_dir = os.path.join(root, file)
                    single_output_dir = os.path.join(category_output, file[:index] + '.txt')
                    print("\033[93mProcessing video: {}\033[0m".format(single_video_dir))
                    single_recognition(single_video_dir, single_output_dir, network_eval, embedding_per_second=1.0,
                                       overlap_rate=0.6)
                    # print("\033[91mUnknown error occurred.\033[0m".format(single_video_dir))
                    gc.collect()
    print("speaker complete")
    sess.close()


if __name__ == '__main__':
    main()
