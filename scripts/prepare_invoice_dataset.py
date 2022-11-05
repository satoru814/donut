"""
Usage:
    ***

receipt s3url = s3://cfo-upload-tokyo-production/receipt//{cid}/{s3_file_name}
"""
import os
import argparse
import subprocess
import json
from pathlibfs import Path
from pdf2image import convert_from_path
from PIL import Image
import awswrangler as wr
import concurrent
import pandas as pd
from functools import partial

os.environ["AWS_PROFILE"] = "sso-ai-lab"
os.environ["AWS_WEB_IDENTITY_TOKEN_FILE"] = ""

def args_parser():
    parser = argparse.ArgumentParser(description="prepare invoice  images from csv")
    parser.add_argument('--path-invoice-tables-s3', type=str, default='s3://ai-lab-production/chen/athena/invoice/temp_table_23eddeaccb7a4045b60a3414ff8c5400/')
    parser.add_argument('--save-dir-s3', type=str, default='s3://ai-lab-production/chen/donut/dataset/20221010-test/')
    
    parser.add_argument('--save-images-dir-local', type=str, default='./dataset/images')
    args = parser.parse_args()
    return args

def convert_copy_file(imglist, path_dest_local, path_dest_s3):
    filepath, destname = imglist
    Path(filepath).copy(os.path.join(path_dest_local, destname))
    # image = convert_from_path(os.path.join(path_dest_local, destname))[0]
    try:
        image = convert_from_path(os.path.join(path_dest_local, destname))[0]
    except:
        image = Image.open(os.path.join(path_dest_local, destname))
    image.save(os.path.join(path_dest_local, destname+'.png'))
    Path(os.path.join(path_dest_local, destname+'.png')).copy(os.path.join(path_dest_s3, destname+'.png'))
    
def main(args):
    #copy tables to donut dir
    print('copy tables')
    tables = wr.s3.list_objects(args.path_invoice_tables_s3)
    # for table in tables:
    #     Path(table).copy(os.path.join(args.save_table_dir_s3, Path(table).name))

    def prepare_sample(row):
        idx, sample = row
        #prepare jsonl
        dicti = {}
        receipt_id_sp = sample['receipt_id_sp']
        if str(receipt_id_sp) == 'nan': #TODO fix receipt_id_sp == 'nan' case
            return None
        mime_type = sample['mime_typ'] #TODO fix columns name miss #TODO fix mime_type == <NAN> case
        pay_amount_gt = sample['pay_amount']
        pay_date_gt = sample['pay_date']
        ground_trues_parse = {"invoice" : {"pay_amount":f"{pay_amount_gt}", "pay_date":f"{pay_date_gt}"}}
        gt_sentence = json.dumps({"gt_parse": ground_trues_parse, 'meta':{'mime_type':mime_type}})
        dicti["file_name"] = os.path.join('./dataset', f'{str(receipt_id_sp)}.png')
        dicti['ground_truth'] = gt_sentence
        
        
        #copy image
        filepath = Path(os.path.join(f's3://cfo-upload-tokyo-production/receipt//{str(sample["company_id"])}/{str(sample["s3_file_path"])}'))
        destname = str(sample['receipt_id_sp'])
        Path(filepath).copy(os.path.join(args.save_images_dir_local, destname))
        try:
            if mime_type == 'application/pdf':
                image = convert_from_path(os.path.join(args.save_images_dir_local, destname))[0]
            else:
                image = Image.open(os.path.join(args.save_images_dir_local, destname))
        except:
            return None
        image.save(os.path.join(args.save_images_dir_local, destname+'.png'))
        Path(os.path.join(args.save_images_dir_local, destname+'.png')).copy(os.path.join(args.save_dir_s3, 'images', destname+'.png'))        
        return dicti
            
    #copy images to donut dir
    print('copy iamges')
    df = wr.s3.read_parquet(args.path_invoice_tables_s3, dataset=True)
    unique_idx = []
    unique_receipt_id = {}
    for idx, row in df.iterrows():
        receipt_id = row['receipt_id_sp']
        if unique_receipt_id.get(receipt_id) == None:
            unique_idx.append(idx)
            unique_receipt_id[receipt_id] = 1
    df_unique = df.iloc[unique_idx]
    jsonl_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor :
        for idx, dicti  in enumerate(executor.map(prepare_sample, df_unique.iterrows())) :
            if idx > 1000 :
                break
            if dicti == None:
                pass
            else:
                jsonl_list.append(dicti)
                
        with open('./metadata.jsonl', 'w') as f:
            for dicti in jsonl_list:
                print(json.dumps(dicti), file=f)
            
        wr.s3.upload('./metadata.jsonl', os.path.join(args.save_dir_s3, 'metadata.jsonl'))


if __name__ == '__main__':
    args = args_parser()
    main(args)