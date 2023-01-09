import os

import cv2
from tqdm import tqdm

def read_crop(videopath, save_dir):
    vidcap = cv2.VideoCapture(videopath)
    count = 1
    fcount = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    save_videoframes_path = os.path.join(save_dir, videopath[:-4])

    os.makedirs(save_videoframes_path, exist_ok=True)
    
    k = fcount / 100 # save 100 frames by video or every frame if video contains less than 100
    k = 1 if k < 1 else int(k)

    while True:
        success, image = vidcap.read()
        if success:
            if count % k == 0:
                cv2.imwrite(os.path.join(save_videoframes_path, 'frame%d.jpg' % count), image)
            count+=1
        else:
            break
    return save_videoframes_path

def crop_images(path, cut_folder=None):
    path = os.path.join(path,'video')
    if cut_folder:
        save_dir = os.path.join(path, os.path.join(cut_folder,'images'))
        path = os.path.join(path, os.path.join(cut_folder,'videos'))
    else:
        save_dir = os.path.join(path, 'images')
    videonames = [i for i in os.listdir(path) if '.DS' not in i]
    
    for videoname in tqdm(videonames):
        videopath = os.path.join(path, videoname)
        read_crop(videopath, save_dir)

if __name__ == "__main__":
    path = '/Users/dmitry/Desktop/cv_itmo/video_classification/data/val/0/'
    crop_images(path) # cut_folder='cut_videos'

    path = '/Users/dmitry/Desktop/cv_itmo/video_classification/data/val/1/'
    crop_images(path) # cut_folder='cut_videos'