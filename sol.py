"""
Usage:
    sol -vvv run  -d --root . --cmd 'python3.8 sol.py --train' --num-gpu 1
"""

import os
import argparse
import subprocess
from pathlibfs import Path
import awswrangler as wr
import concurrent
from functools import partial


def args_parser():
    parser = argparse.ArgumentParser(description="donut")
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--exp', type=str, default='test')
    parser.add_argument('--checkpoints', type=str, default='20221010-test')
    parser.add_argument('--dataset', type=str, default='20221010_main')
    parser.add_argument('--path-s3', type=str, default="s3://ai-lab-production/chen/donut")
    args = parser.parse_args()
    return args


def download_file_from_s3(filepath_s3, path_local):
    Path(filepath_s3).copy(os.path.join(path_local, Path(filepath_s3).name))


def download_dir_from_s3(path_s3, path_local):
    print('download_start')
    filepaths_s3 = wr.s3.list_objects(path_s3)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for idx, _  in enumerate(executor.map(partial(download_file_from_s3, path_local=path_local), filepaths_s3)):
            if idx%1000 == 0:
                print(idx)
    print(f'download{path_s3}_done!')


def upload_file_to_s3s(checkpoint_path_local, checkpoint_path_s3):
    wr.s3.upload(checkpoint_path_local, checkpoint_path_s3)


def main(args):
    path_s3_images = os.path.join(args.path_s3,  'dataset', args.dataset,'images')
    path_s3_jsonl = os.path.join(args.path_s3, 'dataset', args.dataset, 'metadata.jsonl')
    path_s3_checkpoints = os.path.join(args.path_s3, 'result', args.checkpoints)
    path_s3_result = os.path.join(args.path_s3, 'result', args.exp)
    
    #setup
    os.makedirs('./dataset', exist_ok=True)
    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./result', exist_ok=True)
    Path(path_s3_checkpoints).mkdir(parents=True, exist_ok=True)
    Path(path_s3_result).mkdir(parents=True, exist_ok=True)
    
    #Download checkpoint
    download_dir_from_s3(path_s3_checkpoints, './checkpoints')
    
    #imagesをローカルに保存
    download_dir_from_s3(path_s3_images, './dataset')

    #metadata.jsonlをローカルに保存
    download_dir_from_s3(path_s3_jsonl, '')
    
    
    #Training
    os.system('pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113')
    os.system('pip3 install pytorch-lightning')
    os.system('pip3 install sconf')
    if args.train:
        subprocess.run(['python', 'train.py', '--config', './config/train_cord_custom.yaml', '--exp_version', 'main'])
        subprocess.run(['python', 'eval.py', '--config', './config/train_cord_custom.yaml' '--pretrained_model_name_or_path', './result/train_cord_cumstom/main',\
            '--dataset_name_or_path', 'metadata.jsonl', '--save-path-s3', path_s3_result
            ])
    #Save checkpoint
    for filepath in Path(f'./result/train_cord_custom/main').iterdir():
        print(filepath)
        filepath.copy(os.path.join(path_s3_result, filepath.name))
    #Save validations result
    Path('./result/validations.csv').copy(os.path.join(path_s3_result, 'validations.csv'))
        
if __name__ == '__main__':
    args = args_parser()
    main(args)
