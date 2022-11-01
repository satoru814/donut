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


from pdf2image import convert_from_path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from threading import Thread

from donut import DonutModel, JSONParseEvaluator, load_json, save_json


def evaluate(args):
    pretrained_model = DonutModel.from_pretrained(args.pretrained_model_name_or_path)

    if torch.cuda.is_available():
        pretrained_model.half()
        pretrained_model.to("cuda")
    else:
        pretrained_model.encoder.to(torch.bfloat16)

    pretrained_model.eval()

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    questions = args.prompt.split('/')

    ground_truths = []
    predictions = [[] for i in range(len(questions))]
    print(len(questions))
    print(questions)
    
    df = pd.read_csv(args.dataset_path)
    print('inferecne start')
    for idx in tqdm(range(len(df)), total=len(df)):
        sample = df.iloc[idx]
        image_path = os.path.join(args.image_path, str(sample["image"]))
        mime_type = sample['mime_type']
        if mime_type == 'application/pdf':
            image = convert_from_path(image_path)[0]
        else:
            image = Image.open(image_path)

        for j in range(len(questions)):
            question = questions[j]
        # for question in questions : 
            output = pretrained_model.inference(
                image=image,
                prompt=f"<s_docvqa><s_question>{question}</s_question><s_answer>",
            )["predictions"][0]

            predictions[j].append(output['answer'])

    for i in range(len(questions)):
        question = questions[i]
        df[f'{question}'] = predictions[i]

    if args.save_path:
        print(args.save_path)
        df.to_csv(args.save_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default='naver-clova-ix/donut-base-finetuned-docvqa')
    parser.add_argument("--task_name", type=str, default='docvqa')
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--image_path", type=str, default='./data/image/')
    parser.add_argument("--save_path", type=str, default='./result/inference_donut.csv')
    
    args, left_argv = parser.parse_known_args()

    if args.task_name is None:
        args.task_name = os.path.basename(args.dataset_name_or_path)

    predictions = evaluate(args)
