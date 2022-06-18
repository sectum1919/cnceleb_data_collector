# coding:UTF-8
import traceback
from moviepy.editor import *
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import os
import cv2
import shutil
import json
import yaml
from tqdm import tqdm


def transcoding(video_dir, output_dir):
    print(output_dir)
    # if os.path.exists(output_dir):
    #     pass
    cap = cv2.VideoCapture(video_dir)
    if cap.get(cv2.CAP_PROP_FPS) == 25:
        shutil.copyfile(video_dir, output_dir)
        return
    clip = VideoFileClip(video_dir)
    clip.write_videofile(output_dir, verbose=False, fps=25)


if __name__ == '__main__':
    """
    python transcoding.py $input_video_path $input_metadata_path $output_video_path [ $start_seq ]
    必须保证metadata.json存在，且至少应该包含以下字段：
    [
        {
            "savefile": 文件名 ,
            "url": url ,
        }
    ]
    metadata.json保存在 $input_metadata_path 文件夹下，结构如下所示
        $input_metadata_path
        ├── poiname1
        │   ├── entertainment
        │   │   └── metadata.json
        │   ├── interview
        │   │   └── metadata.json
        │   └── vlog
        │       └── metadata.json
        └── poiname2
    """
    argnum = len(sys.argv)
    print(argnum)
    if argnum != 4 and argnum != 5:
        print("args are not valid")
        exit(1)
    top_dir = sys.argv[1]
    json_dir = sys.argv[2]
    output_top_dir = sys.argv[3]
    if not os.path.exists(top_dir):
        print("cannot find origin videos")
        exit(1)
    if not os.path.exists(output_top_dir):
        os.makedirs(output_top_dir)
    names = os.listdir(top_dir)
    i = 0
    
    for name in names[i:]:
        genres = os.listdir(os.path.join(top_dir, name))
        for genre in genres:
            output = os.path.join(output_top_dir, name, genre)
            if not os.path.exists(output):
                os.makedirs(output)
            metadata_json = os.path.join(json_dir, name, genre, 'metadata.json')
            if not os.path.exists(metadata_json):
                continue
            metadata = yaml.safe_load( open(metadata_json))
            new_metadata = []
            i = 1
            for m in tqdm(metadata):
                m['seq'] = 0
                file = os.path.join(top_dir, name, genre, m['savefile']+'.mp4')
                if file.find('.MP4') > 0 or file.find('.mp4') > 0:
                    filename = genre + "-" + str(i) + ".mp4"
                    try:
                        transcoding(file, os.path.join(output, filename))
                        m['seq'] = i
                        new_metadata.append(m)
                        i = i + 1
                    except:
                        traceback.print_stack()
                        continue
                else:
                    continue
            new_metadata_json = os.path.join(json_dir, name, genre, 'metadata_transcoded.json')
            with open(new_metadata_json,'w') as f:
                json.dump(new_metadata, f, ensure_ascii=False, indent=4)
    print("complete")
