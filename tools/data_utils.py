#%%
import json
import os
from pathlib import Path
import shutil
from tqdm import tqdm
import subprocess
#%%
def merge_meta_single(metadata_path, split_path, data_path, name, genre):
    new_meta_f = os.path.join(metadata_path, name, genre, 'final_data.json')
    new_meta = []
    m = os.path.join(metadata_path, name, genre, 'metadata_transcoded.json')
    metadata = json.load(open(m))
    for video in metadata:
        s = os.path.join(split_path, name, genre, f"{genre}-{video['seq']}.txt")
        if not os.path.exists(s):
            continue
        split_list = []
        with open(s) as fp:
            lines = fp.readlines()
        split_list = [ line.strip().split('\t') for line in lines]
        # print(split_list)
        for i in range(0, len(split_list), 1):
            final_video = os.path.join(data_path, name, genre, f"{genre}-{video['seq']}-{i+1}.mp4")
            if not os.path.exists(final_video):
                continue
            new_meta.append(
                {
                    "url": video['url'],
                    "start_time": split_list[i][0],
                    "duration": split_list[i][1],
                    "name": name,
                    "genre": genre,
                    "filename": f"{genre}-{video['seq']}-{i+1}.mp4",
                }
            )
    with open(new_meta_f, 'w') as mf:
        json.dump(new_meta, mf, indent=4, ensure_ascii=False)

def merge_data_part(metadata_path, split_path, data_path):
    name_list = os.listdir(metadata_path)
    for name in name_list:
        for genre in os.listdir(os.path.join(metadata_path, name)):
            merge_meta_single(metadata_path, split_path, data_path, name, genre)

def generate_final_json(metadata_path, split_path, data_path):
    merge_data_part(metadata_path, split_path, data_path)
#%%
# metadata_path = '/work6/cchen/CNCeleb/data/part9/metadata/'
# split_path    = '/work6/cchen/CNCeleb/logs/part9/sysp_result/'
# data_path     = '/work6/cchen/CNCeleb/data/part9/final_data/'
# generate_final_json(metadata_path, split_path, data_path)

# %%
def copy_video_data(
    root_dir, name, 
    src_dir_name='todos', 
    trg_dir_name='checked', 
    src_json_name='todo_data.json', 
    trg_json_name='final_data.json',
    min_duration=1.5,
    max_duration=90,
    ):
    """
    @ discription
        ??????json??????genre????????????mp4???????????????????????????????????????????????????????????????????????????json??????
        ????????????:
            1. ??????genre???????????????mp4???????????????genre
            2. ?????????genre???????????????mp4?????????metadata?????????????????????json?????????
            3. ?????????metadata?????????????????????????????????
    ---------
    @ usage
        root_dir = "/work6/cchen/cnc_data/"
        name = '??????'
        copy_video_data(root_dir, name)
    ---------
    @ params root_dir, name, src_dir_name, trg_dir_name, src_json_name, trg_json_name
        ???????????????????????????trg_dir_name??????????????????????????????
        {root_dir}
            |--{src_dir_name}
            |   `--{name}
            |      `--{genres}
            |          `--{src_json_name}
            `--{trg_dir_name}
                `--{name}
                   `--{genres}
                       `--{src_json_name}
    @ params min_duration 
        ?????????????????????
    @ params max_duration 
        ?????????????????????
    ---------
    @ returns
        no returns
    -------
    """
    new_metadata = {}
    for genre in os.listdir(os.path.join(root_dir, src_dir_name, name)):
        print(genre)
        checked_json = f"{root_dir}/{src_dir_name}/{name}/{genre}/{src_json_name}"
        if not os.path.exists(checked_json):
            print("json file not exits: ", checked_json)
            continue
        new_metadata[genre] = []
        src_dir = f"{root_dir}/{src_dir_name}/{name}/{genre}/"
        dst_dir = f"{root_dir}/{trg_dir_name}/{name}/"
        metadata = json.load(open(checked_json))
        for video in metadata:
            duration_list = [ int(d) for d in video["duration"].split(':') ]
            duration = duration_list[0]*3600 + duration_list[1]*60 + duration_list[2] + duration_list[3]/25.0
            if duration < min_duration or duration > max_duration:
                continue
            src = os.path.join(src_dir, video['filename'])
            Path(os.path.join(dst_dir, video['genre'])).mkdir(exist_ok=True, parents=True)
            dst = os.path.join(dst_dir, video['genre'], video['filename'])
            shutil.copyfile(src, dst)
            new_metadata[genre].append(video)
    for genre, meta in new_metadata.items():
        new_json = os.path.join(root_dir, trg_dir_name, name, genre, trg_json_name)
        json.dump(meta, open(new_json, 'w'))

def merge_json(checked_path, save_path, checked_json_name=None):
    """
    @ discription
    ???${checked_path}???????????????????????????json????????????????????????????????????save_path???
    ---------
    @ params checked_path
        ??????checked_json??????????????????????????????
        checked_path/{pois}/{poi}/{genres}/{checked_json_name}.json
        ??????????????????checked_json_name???????????????????????????????????????json??????
    @ params save_path
        ????????????json???????????????????????????????????????????????????
        save_path/{pois}/{poi}.json
    ---------
    @ returns
        no returns
    -------
    """
    Path(save_path).mkdir(exist_ok=True, parents=True)
    metadata = {}
    for name in os.listdir(checked_path):
        if os.path.isfile( os.path.join(checked_path, name) ):
            continue
        # print(name)
        metadata[name] = []
        for genre in os.listdir( os.path.join(checked_path, name) ):
            genre_path = os.path.join(checked_path, name, genre)
            if os.path.isdir(genre_path):
                # print(genre)
                if checked_json_name is None:
                    fl = os.listdir(genre_path)
                    flist = []
                    for fn in fl:
                        if fn.endswith('.json'):
                            flist.append(fn)
                    fl = flist
                    if len(fl) != 1:
                        print("ERROR: This path contains more than one json, skipping...\n", genre_path, fl)
                        continue
                    json_f = os.path.join(genre_path, fl[0])
                else:
                    json_f = os.path.join(genre_path, checked_json_name)
                print(json_f)
                content = None
                with open( json_f, encoding='UTF-8' ) as f:
                    content = f.read()
                    if content.startswith(u'\ufeff'):
                        content = content.encode('utf8')[3:].decode('utf8')
                metadata[name].extend(json.loads(content))
        with open( os.path.join(save_path, name+'.json'), 'w', encoding='UTF-8' ) as fp:
            json.dump(metadata[name], fp, indent=4, ensure_ascii=False)

def timestamp2second(timestamp):
    time_list = [ int(d) for d in timestamp.split(':') ]
    return time_list[0]*3600 + time_list[1]*60 + time_list[2] + time_list[3]/25.0


genre2int = {
    "advertisement": 1,
    "drama": 2,
    "entertainment": 3,
    "interview": 4,
    "livebroadcast": 5,
    "movie": 6,
    "play": 7,
    "recitation": 8,
    "singing": 9,
    "speech": 10,
    "vlog": 11,
}

def video_hash(a):
    # genre > origin_genre > video_seq > clip_seq
    a_genre = genre2int[a['genre']]
    a_origenre = genre2int[ a['filename'].split('-')[0].lower().replace('_', '') ]
    a_vs = int(a['filename'].split('-')[1])
    a_cs = int(a['filename'].split('-')[2].split('.')[0])
    print(a_genre, a_origenre, a_vs, a_cs)
    istr = str(a_genre).zfill(2) + str(a_origenre).zfill(2) + str(a_vs).zfill(2) + str(a_cs).zfill(4)
    print(istr)
    return int(istr)

def sort_data(metadata, min_duration=1.5, max_duration=90):
    """
    @ discription
        ??????????????????metadata????????????????????????????????????????????????????????????????????????????????????
    ---------
    @ params metadata
        ???json.load?????????metada.json???????????????dic??????
    @ params min_duration 
        ?????????????????????
    @ params max_duration 
        ?????????????????????
    ---------
    @ returns
        ????????????metadata???dic??????
    -------
    """
    new_metadata = []
    # rename genre & rename url & remove duration unmatch
    for video in metadata:
        video["genre"] = video["genre"].replace('_', '').lower()
        video["url"] = video["url"].split('?')[0]
        duration = timestamp2second(video["duration"])
        if duration < min_duration or duration > max_duration:
            continue
        new_metadata.append(video)
    # print(new_metadata)
    new_metadata.sort(key = lambda x : video_hash(x))
    return new_metadata

def rename_data(metadata):
    """
    @ discription
        ????????????????????????metadata????????????????????????????????????????????????????????????
    ---------
    @ params metadata
        sort_data(metadata)????????????
    ---------
    @ returns
        newfn???cnceleb??????????????????metadata???dic??????
    -------
    """
    def video_seq(video):
        return int(video['filename'].split('-')[1])
    def clips_seq(video):
        return int(video['filename'].split('-')[2].split('.')[0])
    new_metadata = []
    first_video = metadata[0]
    last_ge = first_video['genre']
    last_vs = video_seq(first_video)

    last_newfn_vs = 1
    last_newfn_cs = 1

    first_video["newfn"] = f"{first_video['genre']}-01-001"
    new_metadata.append(first_video)
    for video in metadata[1:]:
        ge = video["genre"]
        vs = video_seq(video)
        cs = clips_seq(video)
        if ge != last_ge:
            last_newfn_vs = 1
            last_newfn_cs = 1
        else:
            if vs != last_vs:
                last_newfn_vs += 1
                last_newfn_cs = 1
            else:
                last_newfn_cs += 1
        if last_newfn_cs > 999 or last_newfn_vs > 99:
            break
        video["newfn"] = f"{ge}-{str(last_newfn_vs).zfill(2)}-{str(last_newfn_cs).zfill(3)}"
        new_metadata.append(video)
        last_ge = ge
        last_vs = vs
    return new_metadata

def orginaze_new_name_for_all_poi(
    merged_json_path='/work6/cchen/cnc_data/todos/merged_json',
    sorted_json_path='/work6/cchen/cnc_data/todos/sorted_json',
    renamed_json_path='/work6/cchen/cnc_data/todos/renamed_json',
    min_duration = 1.5,
    max_duration = 90,
):
    """
    @ discription
        ????????????POI??????json??????????????????????????????????????????????????????cnceleb?????????????????????
    ---------
    @ params merged_json_path
        ????????????POI???json?????????????????????????????????
        {merged_json_path}/{POIs}.json
    @ params sorted_json_path
        ???????????????????????????????????????
        {sorted_json_path}/{POIs}.json
    @ params renamed_json_path
        ??????cnceleb??????????????????????????????json????????????????????????POI???json????????????????????????
        {renamed_json_path}/{POIs}.json
    @ params min_duration 
        ?????????????????????
    @ params max_duration 
        ?????????????????????
    ---------
    @ returns
        no returns
    -------
    """
    
    Path(sorted_json_path).mkdir(exist_ok=True, parents=True)
    Path(renamed_json_path).mkdir(exist_ok=True, parents=True)
    for file in os.listdir(merged_json_path):
        if file.endswith('.json'):
            data = json.load( open(os.path.join(merged_json_path, file), encoding='utf-8') )
            sorted_data = sort_data(data, min_duration=min_duration, max_duration=max_duration)
            with open(os.path.join(sorted_json_path, file), 'w', encoding='utf-8') as fp:
                json.dump(sorted_data, fp, ensure_ascii=False, indent=4)
            new_metadata = rename_data(sorted_data)
            with open(os.path.join(renamed_json_path, file), 'w', encoding='utf-8') as fp:
                json.dump(new_metadata, fp, ensure_ascii=False, indent=4)

def generata_id_list(
    startid,
    renamed_json_path='/work6/cchen/cnc_data/todos/renamed_json',
    trg_save_path='/work6/cchen/cnc_data/checked/',
):
    """
    @ discription
        ?????????POI???????????????speakerid???????????????????????????????????????
    ---------
    @ params startid
        ???startid????????????
    @ params renamed_json_path
        ??????cnceleb??????????????????????????????json????????????????????????POI???json????????????????????????
        {renamed_json_path}/{POIs}.json
    @ params trg_save_path
        ??????id_list?????????
    ---------
    @ returns
        no returns
    -------
    """
    
    id = startid
    poi_list = {}
    content = []
    for poi in os.listdir(renamed_json_path):
        if poi.endswith('.json'):
            name = poi[:-5]
            print(name, 'id'+str(id))
            poi_list[name] = 'id'+str(id)
            content.append(f"{poi_list[name]} {name}\n")
            Path(os.path.join(trg_save_path, poi_list[name])).mkdir(exist_ok=True, parents=True)
            id += 1
    with open(os.path.join(trg_save_path,'id_list.txt'), 'w', encoding='utf-8') as fp:
        fp.writelines(content)

def cp_videos_to_tmpdir(
    src_save_path='/work6/cchen/cnc_data/todos/',
    tmp_save_path='/work6/cchen/cnc_data/checked/',
):
    """
    @ discription
        ??????????????????????????????????????????????????????SID??????????????????
    ---------
    @ params src_save_path
        ???????????????????????????????????????
        {src_save_path}/{names}/{genres}/videos.mp4
    @ params tmp_save_path
        ?????????????????????????????????????????????
        {tmp_save_path}/{sid}/videos.mp4
    ---------
    @ returns
        no returns
    -------
    """
    
    poi_name_id = {}
    with open(os.path.join(tmp_save_path,'id_list.txt'), encoding='utf-8') as fp:
        for line in fp.readlines():
            line = line.strip()
            poi_name_id[line.split(' ')[1]] = line.split(' ')[0]
    # print(poi_name_id)
    for name, id in poi_name_id.items():
        src_path = os.path.join(src_save_path, name)
        tmp_path = os.path.join(tmp_save_path, id)
        Path(tmp_path).mkdir(exist_ok=True, parents=True)
        for genre in os.listdir(src_path):
            print(name, genre)
            for file in tqdm(os.listdir(os.path.join(src_path, genre))):
                shutil.copyfile( os.path.join(src_path, genre, file), os.path.join(tmp_path, file) )

def rename_videos_all(
    metadta_path='/work6/cchen/cnc_data/todos/renamed_json/',
    tmp_save_path='/work6/cchen/cnc_data/checked/',
    dst_save_path='/work6/cchen/cnc_data/final_video/',
):
    """
    @ discription
        ??????cnceleb????????????????????????????????????
    ---------
    @ params metadta_path
        ?????????metadata???json??????????????????
    @ params tmp_save_path
        ???????????????????????????????????????????????????{tmp_save_path}/{sid}/videos.mp4
    @ params dst_save_path
        (????????????)??????????????????????????????????????????????????????
        {dst_save_path}/{sid}
                        |--videos.mp4
                        `--sid.json
    ---------
    @ returns
        no returns
    -------
    """
    
    poi_name_id = {}
    with open(os.path.join(tmp_save_path, 'id_list.txt'), encoding='utf-8') as fp:
        for line in fp.readlines():
            line = line.strip()
            poi_name_id[line.split(' ')[1]] = line.split(' ')[0]
    # print(poi_name_id)
    for name, id in poi_name_id.items():
        metadata_f = os.path.join(metadta_path, name+'.json')
        src_path = os.path.join(tmp_save_path, id)
        dst_path = os.path.join(dst_save_path, id)
        Path(dst_path).mkdir(exist_ok=True, parents=True)
        metadata = json.load( open(metadata_f, encoding='utf-8') )
        new_metadata = []
        print(name)
        for video in tqdm(metadata):
            shutil.copyfile( os.path.join(src_path, video['filename']), os.path.join(dst_path, video["newfn"]+'.mp4') )
            new_metadata.append( {
                    "url"       : video["url"],
                    "start_time": video["start_time"],
                    "duration"  : video["duration"],
                    "id"        : id,
                    "name"      : name,
                    "genre"     : video["genre"],
                    "filename"  : video["newfn"],
            } )
        with open( os.path.join(dst_path, f"{id}.json"), 'w', encoding='utf-8') as fp:
            json.dump(new_metadata, fp, ensure_ascii=False, indent=4)


def generate_audios(
    id_list_file='/work6/cchen/cnc_data/checked/id_list.txt',
    video_save_path='/work6/cchen/cnc_data/final_video/',
    audio_save_path='/work6/cchen/cnc_data/final_audio/',
    metadata_save_path='/work6/cchen/cnc_data/final_metadatas/',
):
    """
    @ discription
        ????????????????????????????????????cnceleb??????????????????
    ---------
    @ params video_save_path
        ?????????????????????
    @ params video_save_path
        (????????????)?????????????????????
    @ params metadata_save_path
        (????????????)meta???????????????
    ---------
    @ returns
        no returns
    -------
    """
    Path(metadata_save_path).mkdir(exist_ok=True, parents=True)
    poi_name_id = {}
    with open(id_list_file, encoding='utf-8') as fp:
        for line in fp.readlines():
            line = line.strip()
            poi_name_id[line.split(' ')[1]] = line.split(' ')[0]
            
    for name, id in poi_name_id.items():
        src_path = os.path.join(video_save_path, id)
        dst_path = os.path.join(audio_save_path, id)
        Path(dst_path).mkdir(exist_ok=True, parents=True)
        metadata_f = os.path.join(src_path, id+'.json')
        metadata = json.load( open(metadata_f, encoding='utf-8') )
        print(name)
        subprocess.call(f"mv {metadata_f} {os.path.join(metadata_save_path, id+'.json')}", shell=True)
        for video in tqdm(metadata):
            video_fn = os.path.join(src_path, video['filename']+'.mp4')
            audio_fn = os.path.join(dst_path, video['filename']+'.wav')
            cmd = f"ffmpeg -v quiet -i {video_fn} -f wav -b:a 256k -ar 16000 -ac 1 -acodec pcm_s16le -strict -2 -y {audio_fn}"
            ret = subprocess.call(cmd, shell=True)
            if ret != 0:
                cmd =      f"ffmpeg -i {video_fn} -f wav -b:a 256k -ar 16000 -ac 1 -acodec pcm_s16le -strict -2 -y {audio_fn}"
                ret = subprocess.call(cmd, shell=True)
                exit()


# orginaze_new_name_for_all_poi()
# rename_videos_all()
# generate_audios()
# merge_json()


def get_statistics(final_metadata_save_path, final_audio_save_path, temp_result_path):

    genre_list = [
        "advertisement",
        "drama"        ,
        "entertainment",
        "interview"    ,
        "livebroadcast",
        "movie"        ,
        "play"         ,
        "recitation"   ,
        "singing"      ,
        "speech"       ,
        "vlog"         ,
    ]
    genre_duration = {}
    genre_json = {}
    genre_speaker = {}
    for genre in genre_list:
        genre_duration[genre] = 0.0
        genre_json[genre] = []
        genre_speaker[genre] = []
    poi_data = {}
    poi_json = {}
    for file in os.listdir(final_metadata_save_path):
        if file.endswith('.json'):
            poi_json[file.split('.')[0]] = json.load( open(os.path.join(final_metadata_save_path, file), encoding='utf-8') )
            
    audios = []
    repeat_audios = []
    all_duration = 0.0

    for poi, metadata in poi_json.items():
        poi_data[poi] = {}
        for video in metadata:
            genre_json[video["genre"]].append(video)
            genre_duration[video["genre"]] += timestamp2second(video["duration"])
            all_duration += timestamp2second(video["duration"])
            genre_speaker[video["genre"]].append(poi)
            audio_file = os.path.join(final_audio_save_path, poi, video['filename']+'.wav')
            if not os.path.exists(audio_file):
                print(audio_file)
            if audio_file in audios:
                print(audio_file)
                print(video)
                repeat_audios.append(video)
            audios.append(audio_file)

    # print(len(audios))
    # print(len(set(audios)))
    if not len(audios) == len(set(audios)):
        print("Some thing wrong happened!")
        print("Some audio file repeated, information write into ", os.path.join(temp_result_path, 'repeat.json'))
        json.dump(repeat_audios, open(os.path.join(temp_result_path, 'repeat.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

    print('-'*50)
    print(
        "genre".ljust(15, ' '),
        '|',
        "spk".rjust(5, ' '),
        '|',
        "utt".rjust(7, ' '),
        '|',
        'duration'.rjust(12, ' '),
    )
    print('-'*50)
    for genre in genre_list:
        print(
            genre.ljust(15, ' '), 
            '|',
            str(len(set(genre_speaker[genre]))).rjust(5,' '),  
            '|',
            str(len(genre_json[genre])).rjust(7,' '),  
            '|',
            format(genre_duration[genre]/3600.0, ".6f").rjust(12, ' '),
        )
    print('-'*50)
    print(
        "total".ljust(15, ' '),
        '|',
        str(len(poi_data)).rjust(5, ' '),
        '|',
        str(len(set(audios))).rjust(7, ' '),
        '|',
        format(all_duration/3600.0, ".6f").rjust(12, ' '),
    )
    print('-'*50)