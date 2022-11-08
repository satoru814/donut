"""
Usage:
    ***

receipt s3url = s3://cfo-upload-tokyo-production/receipt//{cid}/{s3_file_name}
"""
import os
import io
import argparse
import subprocess
import numpy as np
import json
from pathlibfs import Path
from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image
import awswrangler as wr
import concurrent
import pandas as pd
from functools import partial

os.environ["AWS_PROFILE"] = "sso-ai-lab"
os.environ["AWS_WEB_IDENTITY_TOKEN_FILE"] = ""

def args_parser():
    parser = argparse.ArgumentParser(description="prepare invoice  images from csv")
    parser.add_argument('--path-dataset-s3', type=str, default='s3://ai-lab-production/chen/athena/invoice/20221018.csv')
    parser.add_argument('--save-dir-s3', type=str, default='s3://ai-lab-production/chen/donut/dataset/20221010-test/')
    parser.add_argument('--test-dataset', action='store_true', default=False)
    
    # parser.add_argument('--save-images-dir-local', type=str, default='./dataset/images')
    parser.add_argument('--seed', type=str, default=1)
    args = parser.parse_args()
    return args

def get_gt_parse(sample):
    pay_amount_gt = sample['pay_amount']
    pay_date_gt = sample['pay_date']
    issue_daate_gt = sample['issue_date']
    partner_name = sample['partner_name']
    bank_name = sample['bank_name']
    branch_name = sample['branch_name']
    account_number = sample['account_number']
    
    ground_trues_parse = {"invoice" : {"pay_amount":f"{pay_amount_gt}", "pay_date":f"{pay_date_gt}"}}
    return gt_parse


def main(args):
    np.random.seed(args.seed)
    #copy tables to donut dir
    print('copy tables')

    def prepare_sample(sample, train_split=0.8, test_dataset=False):
        row, train_prob = sample
        idx, sample = row
        #prepare jsonl
        dicti = {}
        receipt_id_sp = sample['receipt_id_sp']
        if str(receipt_id_sp) == 'nan': #TODO fix receipt_id_sp == 'nan' case
            return None
        mime_type = sample['mime_type'] #TODO fix columns name miss #TODO fix mime_type == <NAN> case
        pay_amount_gt = sample['pay_amount']
        pay_date_gt = sample['pay_date']
        
        ground_trues_parse = {"invoice" : {"pay_amount":f"{pay_amount_gt}", "pay_date":f"{pay_date_gt}"}}
        split = ['train', 'validation'][1-int(train_prob < train_split)]
        if test_dataset:
            split = 'test'
        meta = {'mime_type':mime_type, 'split':split}
        gt_sentence = json.dumps({"gt_parse": ground_trues_parse, 'meta':meta})
        dicti["file_name"] = os.path.join('./dataset', f'{str(receipt_id_sp)}.png')
        dicti['ground_truth'] = gt_sentence
        
        
        #copy image
        path_s3_in = Path(os.path.join(f's3://cfo-upload-tokyo-production/receipt//{str(sample["company_id"])}/{str(sample["s3_file_path"])}'))
        destname = str(sample['receipt_id_sp'])
        try:
            if mime_type == 'application/pdf':
                image = convert_from_bytes(Path.open(path_s3_in, "rb").read())[0]
            else:
                image = Image.open(Path.open(path_s3_in, "rb").read())
        except:
            return None
        
        path_s3_out = os.path.join(args.save_dir_s3, 'images', destname+'.png')
        byteIO = io.BytesIO()
        image.save(byteIO, format='PNG')
        byteArr = byteIO.getvalue()
        Path.open(Path(path_s3_out),'wb').write(byteArr)
        return dicti
            
    #copy images to donut dir
    print('copy iamges')
    df = wr.s3.read_csv(args.path_dataset_s3)
    train_probs = np.random.rand(len(df))
    jsonl_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor :
        for idx, dicti  in enumerate(executor.map(partial(prepare_sample, test_dataset=args.test_dataset), zip(df.iterrows(), train_probs))) :
            if idx > 1000 :
                break
            if dicti != None:
                jsonl_list.append(dicti)
                
        with open('./metadata.jsonl', 'w') as f:
            for dicti in jsonl_list:
                print(json.dumps(dicti), file=f)
            
        wr.s3.upload('./metadata.jsonl', os.path.join(args.save_dir_s3, 'metadata.jsonl'))


if __name__ == '__main__':
    args = args_parser()
    main(args)