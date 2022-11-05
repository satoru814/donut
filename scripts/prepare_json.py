"""
Usage:
    ***
"""
import os
import json
import tempfile
import argparse
import subprocess
import concurrent
import pandas as pd
import awswrangler as wr
from functools import partial
from pathlibfs import Path

os.environ["AWS_PROFILE"] = "sso-ai-lab"
os.environ["AWS_WEB_IDENTITY_TOKEN_FILE"] = ""

def args_parser():
    parser = argparse.ArgumentParser(description="prepare jsonl file for training")
    parser.add_argument('--path-invoice-tables-s3', type=str, default='s3://ai-lab-production/chen/donut/dataset/20221010-test/tables')
    parser.add_argument('--save-jsonl-dir-s3', type=str, default='s3://ai-lab-production/chen/donut/dataset/20221010-test/')
    parser.add_argument('--filename', type=str, default='metadata.jsonl')

    parser.add_argument('--training-data-dir', type=str, default='./dataset')
    parser.add_argument('--no-send-to-s3', action='store_true', default=False)
    args = parser.parse_args()
    return args


def main(args):
    df = wr.s3.read_parquet(args.path_invoice_tables_s3, dataset=True)
    with open('./metadata.jsonl', 'w') as f:
        for i in range(len(df)):
            dicti = {}
            df_i = df.iloc[i]
            receipt_id_sp = df_i['receipt_id_sp']
            if str(receipt_id_sp) == 'nan': #TODO fix receipt_id_sp == 'nan' case
                continue
            mime_type = df_i['mime_typ'] #TODO fix columns name miss #TODO fix mime_type == <NAN> case
            pay_amount_gt = df_i['pay_amount']
            pay_date_gt = df_i['pay_date']
            ground_trues_parse = {"invoice" : {"pay_amount":f"{pay_amount_gt}", "pay_date":f"{pay_date_gt}"}}
            gt_sentence = json.dumps({"gt_parse": ground_trues_parse, 'meta':{'mime_type':mime_type}})
            dicti["file_name"] = os.path.join(args.training_data_dir, str(receipt_id_sp))
            dicti['ground_truth'] = gt_sentence
            print(json.dumps(dicti), file=f)
            if i >= 1000:
                break
    if args.no_send_to_s3:
        pass
    else:
        wr.s3.upload('./metadata.jsonl', os.path.join(args.save_jsonl_dir_s3, args.filename))


if __name__ == '__main__':
    args = args_parser()
    main(args)