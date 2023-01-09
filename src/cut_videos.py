import os
import datetime

import pandas as pd


def cut_video(sample, path_to_videofolder):
    os.makedirs(os.path.join(path_to_videofolder,'cut_videos/videos/'))
    for i in range(len(sample)):
        obj = sample.loc[i]
        videopath = os.path.join(path_to_videofolder, f"{obj['title']}.mp4")
        endpath = os.path.join(path_to_videofolder, f'cut_videos/videos/{obj["title"]}.mp4')

        start = str(datetime.timedelta(seconds=int(obj['time_start'])))
        end = str(datetime.timedelta(seconds=int(obj['time_end'])))
        cmd = f'ffmpeg -ss {start} -to {end} -i "{videopath}" -c copy "{endpath}"'
        os.system(cmd)

if __name__ == "__main__":
    path = '/Users/dmitry/Desktop/cv_itmo/video_classification/data/train'
    dancing = pd.read_csv(os.path.join(path, 'sample_dancing.csv'))
    other = pd.read_csv(os.path.join(path,'sample_other.csv'))

    cut_video(dancing, os.path.join(path,'1/video/'))
    cut_video(other, os.path.join(path, '0/video/'))