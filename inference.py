import argparse
import datetime
import os
import shutil

import cv2
import PIL
import torch
import torchvision.transforms as transforms

from src.crop_frames import read_crop
from src.model import *
from src.prediction import predict_label

parser = argparse.ArgumentParser(description='Video classification Inference with CnnResnet model')

parser.add_argument('--model-path', 
    default='experiments/exp1/checkpoints/checkpoint_0.7121046892039259_1ep.pth', type = str,
    help = 'Path to folder with model checkpoint.'
)
parser.add_argument('--videopath', 
    default='inference.mp4', type = str,
    help = 'Path to inference video.'
)
parser.add_argument('--save-frames-path', default = 'inference_frames', type = str,
    help = 'Path to save frames croped from video.'
)
parser.add_argument('--rm-frames', default = True, type = bool,
    help = 'Flag to delete frames at the end or not.'
)
parser.add_argument('--start-time', type = int,
    help = 'Start time to cut video in seconds.'
)
parser.add_argument('--end-time', type = int,
    help = 'End time to cut video in seconds.'
)
parser.add_argument('--duration', type = int,
    help = 'Duration from start time to cut video in seconds.'
)


def cut_video(args):
    videopath = args.videopath
    if args.start_time==0 or args.start_time:
        start = str(datetime.timedelta(seconds=args.start_time))

        if args.end_time:
            end = str(datetime.timedelta(seconds=args.end_time))
        else:
            if args.duration:
                end = str(datetime.timedelta(seconds=args.start_time+args.duration))
            else:
                raise Exception('You must specify --end-time or --duration arguments to cut video.')
        endpath = f'{videopath[:-4]}_cut.mp4'
        cmd = f'ffmpeg -ss {start} -to {end} -i "{videopath}" -c copy "{endpath}"'
        print(cmd)
        os.system(cmd)
        videopath = endpath
    return videopath


if __name__ == "__main__":
    args = parser.parse_args()
    
    model = torch.load(args.model_path, map_location='cpu')

    args.videopath = cut_video(args)

    frames_path = read_crop(videopath=args.videopath, save_dir=args.save_frames_path)

    # transform params
    h, w = 224, 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    test_transforms = transforms.Compose([
            transforms.Resize((h,w)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ])

    timestamps = 4 # take every 4 frames of folder to predict label 

    pred_label = predict_label(model, frames_path, timestamps, test_transforms, inference_mode=True)

    print(f"\n\033[93mPredicted Label\033[0m: \033[92m{pred_label}\033[0m")

    if args.rm_frames:
        shutil.rmtree(args.save_frames_path)