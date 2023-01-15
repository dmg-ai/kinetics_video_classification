import datetime
import os
import re

import pandas as pd
from tqdm import tqdm


def cut_video(sample, path_to_videofolder):
    endpath = os.path.join(path_to_videofolder,'cut_videos/')
    os.makedirs(endpath,exist_ok=True)
    path_to_videofolder = os.path.join(path_to_videofolder, 'video')
    for i in range(len(sample)):
        obj = sample.loc[i]
        if obj['title'] != 'NaN':
            videopath = os.path.join(path_to_videofolder, obj['title'])
            savepath = os.path.join(endpath, obj["title"])

            start = str(datetime.timedelta(seconds=int(obj['time_start'])))
            end = str(datetime.timedelta(seconds=int(obj['time_end'])))
            cmd = f'ffmpeg -ss {start} -to {end} -i "{videopath}" -c copy "{savepath}"'
            os.system(cmd)

if __name__ == "__main__":
    path = '/Users/dmitry/Desktop/cv_itmo/kinetics_video_classification/data/dancing_classes/val'

    for class_csv in tqdm([i for i in os.listdir(path) if i.endswith('.csv')]):
        data = pd.read_csv(os.path.join(path,class_csv))
        data['title'] = data['title'].fillna('NaN')
        class_num = re.findall(r'\d+', class_csv)[0]
        cut_video(data, os.path.join(path,class_num))