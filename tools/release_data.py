# -*- encoding: utf-8 -*-
'''
Filename         :release_data.py
Description      :处理已校验回收的数据至可发布状态
Time             :2022/06/13 14:02:37
Author           :chenchen
'''
from data_utils import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--work-dir', required=True, help="root path of all video & json data, also the root path of all subfolders generated")
parser.add_argument('--video-folder', required=True, help="videos folder name in work_dir")
parser.add_argument('--json-folder', required=True, help="json files folder name in work_dir")
parser.add_argument('--startid', type=int, required=True, help="start from #")
parser.add_argument('--min-duration', type=float, default=1.5, help="delete video when its duration less than #")
parser.add_argument('--max-duration', type=float, default=90, help="delete video when its duration more than #")


if __name__ == '__main__':
    args = parser.parse_args()
    work_dir = args.work_dir
    video_folder = args.video_folder
    json_folder  = args.json_folder
    startid      = args.startid
    min_duration = args.min_duration
    max_duration = args.max_duration

    merged_json_path = 'merged_json' if not json_folder == 'merged_json' else 'merged_json_generated'
    sorted_json_path = 'sorted_json' if not json_folder == 'sorted_json' else 'sorted_json_generated'
    renamed_json_path = 'renamed_json' if not json_folder == 'renamed_json' else 'renamed_json_generated'

    video_tmp_save_path = 'tmp_videos' if not video_folder == 'tmp_videos' else 'tmp_videos_generated'
    video_final_save_path = 'final_video' if not video_folder == 'final_video' else 'final_video_generated'
    audio_final_save_path = 'final_audio' if not video_folder == 'final_audio' else 'final_audio_generated'
    json_final_save_path = 'final_metadata' if not video_folder == 'final_metadata' else 'final_metadata_generated'


    print("************************************************")
    print("1. Merge jsonfiles of all pois")
    print("************************************************")
    merge_json(
        checked_path = os.path.join(work_dir, json_folder), 
        save_path    = os.path.join(work_dir, merged_json_path),
        checked_json_name = 'checked.json',
    )
        
    print("************************************************")
    print("2. Rename videos name to standard format")
    print("************************************************")
    orginaze_new_name_for_all_poi(
        merged_json_path  = os.path.join(work_dir, merged_json_path),
        sorted_json_path  = os.path.join(work_dir, sorted_json_path),
        renamed_json_path = os.path.join(work_dir, renamed_json_path),
        min_duration = min_duration,
        max_duration = max_duration,
    )

    print("************************************************")
    print("3. Generate sid for each poi")
    print("************************************************")
    generata_id_list(
        startid           = startid,
        renamed_json_path = os.path.join(work_dir, renamed_json_path),
        trg_save_path     = os.path.join(work_dir, video_tmp_save_path),
    )

    print("************************************************")
    print("4. Copy videos to temp path, this will take a long time")
    print("************************************************")
    cp_videos_to_tmpdir(
        src_save_path = os.path.join(work_dir, video_folder),
        tmp_save_path = os.path.join(work_dir, video_tmp_save_path), 
    )

    print("************************************************")
    print("5. Generate final videos, this will take a long time")
    print("************************************************")
    rename_videos_all(
        metadta_path  = os.path.join(work_dir, renamed_json_path),
        tmp_save_path = os.path.join(work_dir, video_tmp_save_path),
        dst_save_path = os.path.join(work_dir, video_final_save_path),
    )

    print("************************************************")
    print("6. Generate final audios & metadtas, this will take a long time")
    print("************************************************")
    generate_audios(
        id_list_file       = os.path.join(work_dir, video_tmp_save_path, 'id_list.txt'),
        video_save_path    = os.path.join(work_dir, video_final_save_path),
        audio_save_path    = os.path.join(work_dir, audio_final_save_path),
        metadata_save_path = os.path.join(work_dir, json_final_save_path),
    )

    print("************************************************")
    print("7. Data can be released now!")
    print("************************************************")
    print("Video data saved in ", os.path.join(work_dir, video_final_save_path))
    print("Audio data saved in ", os.path.join(work_dir, audio_final_save_path))
    print("Metadata   saved in ", os.path.join(work_dir, json_final_save_path))
    print("Calculating statistics... ", end='\r')
    get_statistics(
        final_metadata_save_path = os.path.join(work_dir, json_final_save_path),
        final_audio_save_path    = os.path.join(work_dir, audio_final_save_path),
        temp_result_path         = work_dir,
    )