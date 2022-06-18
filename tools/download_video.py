#%%
import requests
from pathlib import Path
import os
import json
import subprocess
import time
import re
from tqdm import tqdm
def update_metadata_bilibili(name_kw, genre_kw, metadata_file):
    """获取视频列表，并保存"""

    new_metadata = []
    for p in tqdm(range(1,3,1)):
        url = f'https://api.bilibili.com/x/web-interface/search/all/v2?page={p}&page_size=15&keyword={name_kw}%20{genre_kw}'
        res = requests.get(url)
        for item in res.json()["data"]['result']:
            if item['result_type'] == 'video':
                data = item['data']
                for d in data:
                    if d['type'] == 'video':
                        d['url'] = d['arcurl']
                        d['savefile'] = d['bvid']
                        new_metadata.append(d)

    cur_metadata = []
    already_exists_urls = set()
    if Path(metadata_file).exists():
        with open(metadata_file) as f:
            cur_metadata = json.load(f)
            already_exists_urls = set([v['url'] for v in cur_metadata])

    update_cnt = 0
    for v in new_metadata:
        if v['url'] in already_exists_urls:
            continue
        cur_metadata.append(v)
        already_exists_urls.add(v['url'])
        update_cnt += 1

    with open(metadata_file, "w") as f:
        json.dump(cur_metadata, f, ensure_ascii=False, indent=4)
    print(f"Update {update_cnt} !")

def download_videos(download_dir, metadata_file):
    """
        根据metadata文件，下载视频。
    """
    
    Path(download_dir).mkdir(exist_ok=True, parents=True)

    with open(metadata_file) as f:
        metadata = json.load(f)

    from tqdm import tqdm
    for v in tqdm(metadata):
        savefile = os.path.join(download_dir, f"{v['savefile']}.mp4")
        if os.path.exists(savefile):
            continue
        cmd = f'you-get "{v["url"]}" -o "{download_dir}" -O "{v["savefile"]}" '
        subprocess.call(cmd, shell=True)

if __name__ == "__main__":
    
    rootpath = '/work9/cchen/project/CNCeleb/cnc3_final/demo/part0/data/'
    download_list = [
        {
            "name":'超级小桀',
            "genre_kw":'采访',
            "genre":'interview',
        },
        {
            "name":'超级小桀',
            "genre_kw":'直播',
            "genre":'livebroadcast',
        },
        {
            "name":'寅子',
            "genre_kw":'采访',
            "genre":'interview',
        },
        {
            "name":'寅子',
            "genre_kw":'直播',
            "genre":'livebroadcast',
        },
    ]

    for item in download_list:

        name = item['name']
        genre_kw = item['genre_kw']
        genre = item['genre']

        metadata_path = os.path.join(rootpath, 'metadata', name, genre)
        Path(metadata_path).mkdir(exist_ok=True, parents=True)

        # update_metadata_bilibili(
        #     f'{name}',
        #     f'{genre_kw}',
        #     f'{metadata_path}/metadata.json',
        #     )
        # 人工审查并修改json后再执行下面的下载

        download_videos(
            f'{rootpath}/original-data/{name}/{genre}/', 
            f'{metadata_path}/metadata.json',
            )