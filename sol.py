"""
Usage:
    sol -vvv run  -d --root . --cmd 'python3.8 sol.py --vqgan' --num-gpu 1
"""

import os
import argparse
import subprocess
from pathlibfs import Path
import awswrangler as wr
import concurrent
from functools import partial


def args_parser():
    parser = argparse.ArgumentParser(description="vqgan")
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--exp', type=str, default='1')
    parser.add_argument('--path-images-s3', type=str, default="s3://ai-lab-production/chen/donut/dataset/20221010-test/images")
    parser.add_argument('--path-jsonl-s3', type=str, default="s3://ai-lab-production/chen/donut/dataset/20221010-test/metadata.jsonl")
    
    parser.add_argument('--path-jsonl-local', type=str, default='.')
    parser.add_argument('--dataset', type=str, default='./dataset/')
    parser.add_argument('--result', type=str, default='./result/')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--max-load', type=int, default=5000) #train num
    parser.add_argument('--path-result-s3', type=str, default='s3://ai-lab-production/chen/donut/result/20221010-test/')
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
    #setup
    ckpt_path = './checkpoints/'
    os.makedirs(args.dataset, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)
    os.system('pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113')
    os.system('pip3 install pytorch-lightning')
    os.system('pip3 install sconf')


    #Download checkpoint
    # if wr.s3.does_object_exist(args.path_result_s3): 
    download_dir_from_s3(args.path_result_s3, args.checkpoints)
    
    # if wr.s3.does_object_exist(ckpt_path_vqgan_s3):
    #     download_file_from_s3(ckpt_path_vqgan_s3, ckpt_path_vqgan)
    # else:
    #     print(f'doesnt exist {ckpt_path_vqgan_s3}')
    
    #imagesをローカルに保存
    download_dir_from_s3(args.path_images_s3, args.dataset)

    #metadata.jsonlをローカルに保存
    download_dir_from_s3(args.path_jsonl_s3, args.path_jsonl_local)
    
    #training_vqganの実行
    if args.train:
        subprocess.run(['python', 'train.py', '--config', './config/train_cord_custom.yaml', '--exp_version', 'test'])
    for filepath in Path('./result/train_cord_custom/test').iterdir():
        print(filepath)
        filepath.copy(os.path.join(args.path_result_s3, filepath.name))

if __name__ == '__main__':
    args = args_parser()
    main(args)
