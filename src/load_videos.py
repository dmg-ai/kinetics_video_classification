import os

import pandas as pd
from pytube import YouTube
from tqdm import tqdm

def load_video(sample, save_dir):
    save_dir = os.path.join(save_dir, 'video')

    count = 0
    for i in range(sample.shape[0]):
        url = sample.iloc[i]['url']
        try:
            yt = YouTube(url)
            yt = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            filename = f'{str(i)}.mp4'
            sample.loc[i,'title'] = filename
            yt.download(output_path=save_dir,filename=filename)
            count += 1
        except:
            pass
    return sample, count
        
def create_sample(mode='train', 
                  sample_size=50,
                  class_name = 'dance', 
                  many_classes=False, 
                  path_to_data = 'data', 
                  path_to_save='data'):

    path_to_save = os.path.join(path_to_save, f'{class_name}_classes')

    all_train_data_path = os.path.join(path_to_data, 'train_all.csv')
    if os.path.exists(all_train_data_path):
        data_csv = pd.read_csv(all_train_data_path)
    else:
        data_json = pd.read_json(os.path.join(path_to_data, f'kinetics700_2020/{mode}.json'))
        data_csv = pd.read_csv(os.path.join(path_to_data, f'kinetics700_2020/{mode}.csv'))

        data_json = data_json.swapaxes("index", "columns").reset_index()

        data_csv = pd.merge(data_csv, data_json[['index','duration','url']], how='inner', left_on='youtube_id', right_on='index',)
        data_csv = data_csv.drop(['split','index'],axis=1)
        data_csv.to_csv(all_train_data_path, index=False)        

    check_dance = lambda x: 1 if class_name in x else 0

    data_csv['target'] = data_csv['label'].apply(check_dance)

    if many_classes:
        data_csv = data_csv[data_csv['target']==1]
        data_csv['target'] = data_csv['label'].astype('category').cat.codes

    targets = data_csv['target'].unique()

    video_counts = dict()
    for i, target in tqdm(enumerate(targets)):
        target_sample = data_csv[data_csv['target']==target].sample(sample_size)
        target_sample = target_sample.reset_index().drop('index',axis=1)
        
        save_dir = os.path.join(path_to_save, mode)

        target_sample, video_count = load_video(target_sample, os.path.join(save_dir, str(target)))

        target_sample.to_csv(os.path.join(save_dir, f'dance{target}.csv'), index=False)

        video_counts[target] = video_count
    
    for target in video_counts.keys():
        print(f'\nLoaded {video_counts[target]} "target {target}" videos of {sample_size}.')

if __name__ == "__main__":
    
    create_sample(mode='val', 
                  sample_size=5,
                  class_name='dancing',
                  many_classes=True,
                  path_to_data='/Users/dmitry/Desktop/cv_itmo/kinetics_video_classification/data',
                  path_to_save='/Users/dmitry/Desktop/cv_itmo/kinetics_video_classification/data')