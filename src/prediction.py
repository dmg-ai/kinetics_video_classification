import os
import re

import numpy as np
import torch
from PIL import Image


# functions to sort folder names as frame1, frame2, frame3, ...
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def predict_label(model, folder_path, timestamps, test_transforms, inference_mode=False):
    folder_frames = [os.path.join(folder_path, frame) for frame in os.listdir(folder_path)]
    folder_frames.sort(key=natural_keys)
    steps = len(folder_frames)//4 # if folder for example contains 19 frames we take the first 16 frames

    if not inference_mode:
        true_label = int(folder_frames[0].split('/')[3])

    pred_labels = []
    for i in range(steps): # predict by 4 frames
        folder_batch = folder_frames[i*timestamps:(i+1)*timestamps] # take 4 consecutive frames from folder

        images = []
        for frame_path in folder_batch: # read frames
            image = Image.open(frame_path)
            images.append(image)

        for i, img in enumerate(images): # make transforms
            img = test_transforms(img)
            images[i] = img
        
        images = torch.stack(images)
        images = torch.unsqueeze(images,0) # get tensor of shape (BS,FRAMES_NUM,C,H,W): (1,4,3,224,224)

        # predict
        with torch.no_grad():
            pred = model(images)
        pred_label = pred.argmax(dim=1).item() # get label
        pred_labels.append(pred_label)

    pred_label = max(set(pred_labels), key=pred_labels.count) # get most common label from all pred labels
    
    if inference_mode:
        return pred_label
    else:
        return true_label, pred_label