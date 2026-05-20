import os
import json
import copy
import math
import argparse
import asyncio
import numpy as np
from typing import Literal
from pydantic import BaseModel
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from datasets import load_dataset
import pandas as pd
from glob import glob
import re
try:
    from .reward_score import default_compute_score
except ImportError:
    from reward_score import default_compute_score

async def compute_single_reward(data_source, response, ground_truth, extra_info):
    """
    Compute reward for one (response, ground-truth) pair.

    * arg_tuple = (gid, response, data_source, ground_truth, extra_info, resp_idx)
    * Returns (gid, detailed_dict, resp_idx)
    """
    try:
        result = await default_compute_score(
            data_source=data_source,
            solution_str=response,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        if isinstance(result, dict):
            detailed = result
            score = detailed.get("score", 0.0)
        else:  # float
            detailed = {"score": float(result)}
            score = float(result)
        detailed["score"] = score
        detailed.update({"data_source": data_source, "response": response, 
                         "ground_truth": ground_truth, "extra_info": extra_info})
        return detailed
    except Exception as e:
        return {"score": 0.0, "error": str(e), 
                "data_source": data_source, "response": response, 
                "ground_truth": ground_truth, "extra_info": extra_info}
            
async def judge_all_responses(qa_df, remove_tags=False, num_workers=None):
    async def bound_func(row):
        async with semaphore:
            if 'answer' in row and not pd.isna(row['answer']):
                ground_truth = row['answer']
            else:
                ground_truth = row['reward_model']['ground_truth']
                
            if 'extra_info' in row and not pd.isna(row['extra_info']):
                extra_info = row['extra_info']
                if 'question' not in extra_info:
                    extra_info['question'] = row['question']
            else:
                extra_info = {'question': row['question']}
                
            if "data_source" not in extra_info:
                extra_info['data_source'] = row.get('data_source', '')
                
            if remove_tags:
                if isinstance(row['system_answer'], str):
                    clean_answer = re.sub(r"<(think|configurator|user_intent|reasoning|planning|reflection|summary)\b[^>]*>[\s\S]*?<\/\1>", 
                                          "", row['system_answer']).strip()
                else:
                    clean_answer = row['system_answer']
                # print(clean_answer)
            else:
                clean_answer = row['system_answer']
                
            data_source = row.get('source', row['general_domain'])
            content = await compute_single_reward(data_source, clean_answer,
                                                  ground_truth, extra_info)
            content['question'] = row['question']
            return content
    
    if num_workers is None:
        num_workers = args.num_workers
    semaphore = asyncio.Semaphore(num_workers)
    total = len(qa_df)
    pbar = tqdm_asyncio(total=total, desc="Judging responses")
    async with semaphore:
        tasks = [bound_func(row) for i, row in qa_df.iterrows()]
        results = await tqdm_asyncio.gather(*tasks)
        pbar.close()
    return results

def load_qa_data(args):
    if args.dataset_path.endswith('.jsonl'):
        with open(args.dataset_path, 'r') as f:
            questions_df = pd.read_json(f, lines=True)
    else:
        # load the question json file
        with open(args.dataset_path, 'r') as f:
            questions_df = pd.read_json(f)
        
    if args.system == 'deerflow':
        
        # The goal is to massage the answers into a format like so: {
        #     "dataset": dataset,
        #     "question": question,
        #     "messages": messages,
        #     "answer": answer,
        #     "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        # }
        data = []
        for file in glob(f"{args.answers_path}/*.jsonl"):
            with open(file, 'r') as f:
                rows = []
                for line in f:
                    rows.append(json.loads(line))
                    
                last_row = rows[-1]
                if last_row.get('node') == 'answerer':
                    system_answer = last_row['content']
                else:
                    system_answer = None
                    
                data.append({
                    "question_deerflow": rows[0]["question"],
                    "messages": rows[1:],
                    "system_answer": system_answer,
                })
        answers_df = pd.DataFrame(data)
        
        # print(answers_df.head())
        # print(questions_df.head())
        
        qa_df = pd.merge(questions_df, answers_df, on='question_deerflow', how='left')
        
    elif args.system == 'direct-api':
        # load the answer json file
        with open(args.answers_path, 'r') as f:
            answers_df = pd.read_json(f, lines=True)
            answers_df.rename(columns={'answer': 'system_answer'}, inplace=True)
        qa_df = pd.merge(questions_df, answers_df, on='question', how='inner')
        # qa_df = pd.merge(questions_df, answers_df, on='question', how='left')
    
    else: 
        raise ValueError(f"Invalid system: {args.system}")
    
    # print(qa_df[['question', 'answer', 'reward_model']].head())
    # for i, row in qa_df.iterrows():
    #     print(row['question'][:50], row.get('answer', 'tes'), str(row.get('reward_model', {}))[:50])
        
    # qa_df = qa_df[qa_df['source'] != 'guru_filtered_v2_codegen']
    qa_df = qa_df.drop_duplicates(subset=['question'])
    # qa_df = qa_df[qa_df['source'] == 'guru_filtered_v2_math']
    
    # print(qa_df.shape)
    return qa_df


def main(args):
    assert args.num_workers > 1, "num_workers must be 2 or greater"
    
    qa_df = load_qa_data(args)

    output_filepath = f"judged_{args.output_name}.json"
    # API will only be called for unjudged responses
    results = asyncio.run(judge_all_responses(qa_df, remove_tags=args.remove_tags))
    
    # cache judge output
    with open(output_filepath, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", type=str, required=True, choices=['deerflow', 'direct-api'], help="System type")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset file")
    parser.add_argument("--answers_path", type=str, required=True, help="Path to answers file for direct-api or directory for deerflow") 
    parser.add_argument("--output_name", type=str, required=True, help="Output name for judged results")
    parser.add_argument("--num_workers", type=int, default=16, help="Async semaphore size. This depends on your rate limit.")
    parser.add_argument("--judge", type=str, default="o4-mini", help="Judge model")
    parser.add_argument("--remove_tags", action="store_true", help="Remove tags from answer")
    args = parser.parse_args()
    main(args)
