# -*- encoding: utf-8 -*-
'''
Filename         :generate_splited_json.py
Description      :根据run.sh得到的切分结果，生成待检查的json文件
Time             :2022/06/18 20:49:14
Author           :chenchen
'''
from data_utils import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--metadata-path', required=True, help="absolute path of data/metadata/")
parser.add_argument('--split-path', required=True, help="absolute path of logs/sysp_result/")
parser.add_argument('--data-path', required=True, help="absolute path of data/final_data/")

if __name__ == "__main__":
    args = parser.parse_args()
    metadata_path = args.metadata_path
    split_path    = args.split_path
    data_path     = args.data_path
    generate_final_json(metadata_path, split_path, data_path)