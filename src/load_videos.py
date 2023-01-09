import os

import pandas as pd
from pytube import YouTube
from tqdm import tqdm


def check_dance(x):
    if 'dancing' in x:
        return 1
    else:
        return 0

def load_video(sample, save_dir):
    save_dir = os.path.join(save_dir, 'video')

    count = 0
    titles = []
    for i in tqdm(range(sample.shape[0])):
        url = sample.iloc[i]['url']
        try:
            yt = YouTube(url)
            yt = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            titles.append(yt.title)
            yt.download(save_dir)
            count += 1
        except:
            titles.append(None)
    sample['title'] = titles
    return sample, count
        
def create_sample(mode='train', sample_size=50, path_to_data = 'data', path_to_save='data'):
    data_json = pd.read_json(os.path.join(path_to_data, f'kinetics700_2020/{mode}.json'))
    data_csv = pd.read_csv(os.path.join(path_to_data, f'kinetics700_2020/{mode}.csv'))

    data_json = data_json.swapaxes("index", "columns").reset_index()

    data_csv = pd.merge(data_csv, data_json[['index','duration','url']], how='inner', left_on='youtube_id', right_on='index',)
    data_csv = data_csv.drop(['split','index'],axis=1)
    
    data_csv['target'] = data_csv['label'].apply(check_dance)

    sample_dancing = data_csv[data_csv['target']==1].sample(sample_size)
    sample_other = data_csv[data_csv['target']==0].sample(sample_size)
    
    save_dir = os.path.join(path_to_save, mode)

    sample_dancing, dancing_count = load_video(sample_dancing, os.path.join(save_dir, '1'))
    sample_other, other_count = load_video(sample_other, os.path.join(save_dir, '0'))

    print(f"\nLoaded video path: {save_dir}\n")
    sample_dancing.to_csv(os.path.join(save_dir, 'sample_dancing.csv'), index=False)
    sample_other.to_csv(os.path.join(save_dir, 'sample_other.csv'), index=False)

    print(f'Loaded {dancing_count} dancing videos from {sample_size}.')
    print(f'Loaded {other_count} other videos from {sample_size}.')

if __name__ == "__main__":
    
    create_sample(mode='val', 
                  sample_size=5,
                  path_to_data='/Users/dmitry/Desktop/cv_itmo/video_classification/data',
                  path_to_save='/Users/dmitry/Desktop/cv_itmo/video_classification/data')