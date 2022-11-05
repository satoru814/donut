"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import random
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from sconf import Config

from typing import Any, Dict, List, Tuple, Union
from donut import DonutModel, JSONParseEvaluator, load_json, save_json

#load
def load_image(image_path: str):
    return Image.open(image_path)
    
def test(args, config):
    pretrained_model = DonutModel.from_pretrained(args.pretrained_model_name_or_path)

    if config.gpu:
        pretrained_model.half()
        pretrained_model.to("cuda")
    else:
        pretrained_model.encoder.to(torch.float32)

    pretrained_model.eval()

    # if args.save_path:
    #     os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    predictions = []
    ground_truths = []
    filenames = []
    accs = []

    evaluator = JSONParseEvaluator()
    # dataset = load_dataset(args.dataset_name_or_path, split=args.split)
    dataset = load_json(args.dataset_name_or_path, split=args.split)
    for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
        ground_truth = json.loads(sample["ground_truth"])
        if config.gpu:
            image = load_image(sample['file_name']) 
        else :
            image = load_image(os.path.join('./dataset/images/', Path(sample['file_name']).name)) #load from test data from ./datset/images
            
        if args.task_name == "docvqa":
            output = pretrained_model.inference(
                image=sample["image"],
                prompt=f"<s_{args.task_name}><s_question>{ground_truth['gt_parses'][0]['question'].lower()}</s_question><s_answer>",
            )["predictions"][0]
        else:
            output = pretrained_model.inference(image=image, prompt=f"<s_{args.task_name}>")["predictions"][0]

        if args.task_name == "rvlcdip":
            gt = ground_truth["gt_parse"]
            score = float(output["class"] == gt["class"])
        elif args.task_name == "docvqa":
            # Note: we evaluated the model on the official website.
            # In this script, an exact-match based score will be returned instead
            gt = ground_truth["gt_parses"]
            answers = set([qa_parse["answer"] for qa_parse in gt])
            score = float(output["answer"] in answers)
        else:
            filename = sample['file_name']
            gt = ground_truth["gt_parse"]
            score = evaluator.cal_acc(output, gt)

        accs.append(score)

        predictions.append(output)
        ground_truths.append(gt)
        filenames.append(filename)

    scores = {
        "ted_accuracies": accs,
        "ted_accuracy": np.mean(accs),
        "f1_accuracy": evaluator.cal_f1(predictions, ground_truths),
    }
    print(
        f"Total number of samples: {len(accs)}, Tree Edit Distance (TED) based accuracy score: {scores['ted_accuracy']}, F1 accuracy score: {scores['f1_accuracy']}"
    )

    # if args.save_path:
    scores["predictions"] = predictions
    scores["ground_truths"] = ground_truths
    scores["filenames"] = filenames
        # save_json(args.save_path, scores)
    return scores

def convert_to_df(scores, config):
    predictions = scores['predictions']
    filenames = scores['filenames']
    
    predictions_json = {ky:[] for ky in config.target_items}
    for pred in predictions:
        try:
            # pred_json = json.loads(pred)['invoice']
            pred_json = pred['invoice']
            for item_ky in config.target_items:
                try:
                    item_val = pred_json[item_ky]
                except KeyError:
                    item_val = 'no item'
                predictions_json[item_ky].append(item_val)
                
        #prediction is not json format
        except json.JSONDecodeError:
            for ky in predictions_json.keys():
                predictions_json[ky].append(None)
    df_pred = pd.DataFrame(predictions_json)
    return df_pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default='donut-base-finetuned-docvqa')
    parser.add_argument("--config", type=str, default='config/train_cord_custom.yaml')
    
    parser.add_argument("--dataset_name_or_path", type=str, default='metadata.jsonl')
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--task_name", type=str, default=None)
    args, left_argv = parser.parse_known_args()
    config = Config(args.config)
    random.seed(config.seed)

    if args.task_name is None:
        args.task_name = os.path.basename(args.dataset_name_or_path)
    
    scores = test(args, config)
    df_pred = convert_to_df(scores, config)
    df_pred.to_csv('result/validations.csv', index=False)
    # Path('./result/validations.csv').copy(os.path.join(args.save_path_s3, 'validations.csv'))
    # return df_pred