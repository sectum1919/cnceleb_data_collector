import json
import os

def get_statistics(final_metadata_save_path, final_audio_save_path):

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
        genre_json = []
        genre_speaker = []
    poi_data = {}
    poi_json = {}
    for file in os.listdir(final_metadata_save_path):
        if file.endswith('.json'):
            poi_json[file.split('.')[0]] = json.load( open(os.path.join(final_metadata_save_path, file), encoding='utf-8') )


    def timestamp2second(timestamp):
        time_list = [ int(d) for d in timestamp.split(':') ]
        return time_list[0]*3600 + time_list[1]*60 + time_list[2] + time_list[3]/25.0

    audios = []
    repeat_audios = []

    for poi, metadata in poi_json.items():
        poi_data[poi] = {}
        for video in metadata:
            genre_json[video["genre"]].append(video)
            genre_duration[video["genre"]] += timestamp2second(video["duration"])
            genre_speaker[video["genre"]].append(poi)
            audio_file = os.path.join(final_audio_save_path, poi, video['filename']+'.wav')
            if not os.path.exists(audio_file):
                print(audio_file)
            if audio_file in audios:
                print(audio_file)
                print(video)
                repeat_audios.append(video)
            audios.append(audio_file)

    print(len(audios))
    print(len(set(audios)))

    # json.dump(repeat_audios, open('/work6/cchen/CNCeleb/repeat.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

    print(
        "genre".center(15, ' '),
        '|',
        "spk".center(5, ' '),
        '|',
        "utt".center(7, ' '),
        '|',
        'duration'.center(10, ' '),
    )
    print('-'*40)
    for genre in genre_list:
        print(
            genre.ljust(15, ' '), 
            '|',
            str(len(set(genre_speaker[genre]))).center(5,' '),  
            '|',
            str(len(genre_json[genre])).center(7,' '),  
            '|',
            format(genre_duration[genre]/3600.0, ".6f").rjust(10, ' '),
        )
