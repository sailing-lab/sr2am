"""Build the full SR²AM evaluation set used for the paper's headline Pass@1.

Reproduces the 8219-question repeated test set spanning 11 benchmarks (math, science,
tabular analysis, web information seeking). Each benchmark is repeated to reduce variance
(aime24/aime25 x32; gpqa_diamond/gaia/xbench_deepsearch x4; all others x1).

The set is assembled from three upstream sources (point at local copies via env vars):

  * Guru offline-eval parquets  -> SR2AM_GURU_EVAL_ROOT
      HF dataset LLM360/guru-RL-92k, `offline_eval/` subdir.
  * AFM web benchmarks (json)   -> SR2AM_AFM_BENCHMARKS_ROOT
      GitHub OPPO-PersonalAI/Agent_Foundation_Models, AFM/data/web_agent/test_benchmarks/.
  * xbench DeepSearch (csv)     -> SR2AM_XBENCH_ROOT
      GitHub xbench-ai/xbench-evals (decrypt the shipped CSV per their instructions).

Usage:
    python evaluation/prepare_test_data.py create_test_dataset_full \
        --output_file data/sr2am_test_full.jsonl
"""

import os
import pandas as pd
from tqdm import tqdm
import fire

# Configurable data roots (override via environment variables)
SR2AM_GURU_EVAL_ROOT = os.environ.get(
    "SR2AM_GURU_EVAL_ROOT", os.path.expanduser("~/guru-RL-92k/offline_eval"))
SR2AM_AFM_BENCHMARKS_ROOT = os.environ.get(
    "SR2AM_AFM_BENCHMARKS_ROOT",
    os.path.expanduser("~/Agent_Foundation_Models/AFM/data/web_agent/test_benchmarks"))
SR2AM_XBENCH_ROOT = os.environ.get(
    "SR2AM_XBENCH_ROOT", os.path.expanduser("~/xbench-evals/data"))


def load_aime24(
    input_path: str = "" + SR2AM_GURU_EVAL_ROOT + "/math__aime_repeated_8x_240.parquet"
):
    print('Loading aime24...')
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} samples from {input_path}")
    df = df.drop_duplicates(subset=['prompt'])
    print(f"After dropping duplicates: {len(df)} samples from {input_path}")
    print('Example of aime24:')
    print(df.iloc[0])
    ds = [{
        "id": f"aime24-{i}",
        "question": data["prompt"][0]['content'],
        "reward_model": data["reward_model"],
        "extra_info": data["extra_info"],
        "data_source": data["data_source"],
        "general_domain": 'math__aime24',
    } for i, data in tqdm(df.iterrows(), total=len(df))]
    print(f"Loaded {len(ds)} samples from aime24")
    return ds


def load_aime25(
    input_path: str = "" + SR2AM_GURU_EVAL_ROOT + "/math__aime2025_repeated_8x_240.parquet"
):
    print('Loading aime25...')
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} samples from {input_path}")
    df = df.drop_duplicates(subset=['prompt'])
    print(f"After dropping duplicates: {len(df)} samples from {input_path}")
    print('Example of aime25:')
    print(df.iloc[0])
    ds = [{
        "id": f"aime25-{i}",
        "question": data["prompt"][0]['content'],
        "reward_model": data["reward_model"],
        "extra_info": data["extra_info"],
        "data_source": data["data_source"],
        "general_domain": 'math__aime25',
    } for i, data in tqdm(df.iterrows(), total=len(df))]
    print(f"Loaded {len(ds)} samples from aime25")
    return ds


def load_math500(
    input_path: str = "" + SR2AM_GURU_EVAL_ROOT + "/math__math_500.parquet"
):
    print('Loading math500...')
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} samples from {input_path}")
    df = df.drop_duplicates(subset=['prompt'])
    print(f"After dropping duplicates: {len(df)} samples from {input_path}")
    print('Example of math500:')
    print(df.iloc[0])
    ds = [{
        "id": f"math500-{i}",
        "question": data["prompt"][0]['content'],
        "reward_model": data["reward_model"],
        "extra_info": data["extra_info"],
        "data_source": data["data_source"],
        "general_domain": 'math__math500',
    } for i, data in tqdm(df.iterrows(), total=len(df))]
    print(f"Loaded {len(ds)} samples from math500")
    return ds


def load_gpqa_diamond(
    input_path: str = "" + SR2AM_GURU_EVAL_ROOT + "/stem__gpqa_diamond_198.parquet"
):
    print('Loading gpqa_diamond...')
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} samples from {input_path}")
    df = df.drop_duplicates(subset=['prompt'])
    print(f"After dropping duplicates: {len(df)} samples from {input_path}")
    print('Example of gpqa_diamond:')
    print(df.iloc[0])
    ds = [{
        "id": f"gpqa_diamond-{i}",
        "question": data["prompt"][0]['content'],
        "reward_model": data["reward_model"],
        "extra_info": data["extra_info"],
        "data_source": data["data_source"],
        "general_domain": 'stem__gpqa_diamond',
    } for i, data in tqdm(df.iterrows(), total=len(df))]
    print(f"Loaded {len(ds)} samples from gpqa_diamond")
    return ds


def load_supergpqa(
    input_path: str = "" + SR2AM_GURU_EVAL_ROOT + "/stem__supergpqa_1k.parquet"
):
    print('Loading supergpqa...')
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} samples from {input_path}")
    df = df.drop_duplicates(subset=['prompt'])
    print(f"After dropping duplicates: {len(df)} samples from {input_path}")
    print('Example of supergpqa:')
    print(df.iloc[0])
    ds = [{
        "id": f"supergpqa-{i}",
        "question": data["prompt"][0]['content'],
        "reward_model": data["reward_model"],
        "extra_info": data["extra_info"],
        "data_source": data["data_source"],
        "general_domain": 'stem__supergpqa',
    } for i, data in tqdm(df.iterrows(), total=len(df))]
    print(f"Loaded {len(ds)} samples from supergpqa")
    return ds


def load_finqa(
    input_path: str = "" + SR2AM_GURU_EVAL_ROOT + "/table__finqa_1.1k.parquet"
):
    print('Loading finqa...')
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} samples from {input_path}")
    df = df.drop_duplicates(subset=['prompt'])
    print(f"After dropping duplicates: {len(df)} samples from {input_path}")
    print('Example of finqa:')
    print(df.iloc[0])
    ds = [{
        "id": f"finqa-{i}",
        "question": data["prompt"][0]['content'],
        "reward_model": data["reward_model"],
        # "extra_info": data["extra_info"],
        "data_source": data["data_source"],
        "general_domain": 'table__finqa',
    } for i, data in tqdm(df.iterrows(), total=len(df))]
    print(f"Loaded {len(ds)} samples from finqa")
    return ds


def load_multihier(
    input_path: str = "" + SR2AM_GURU_EVAL_ROOT + "/table__multihier_336.parquet"
):
    print('Loading multihier...')
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} samples from {input_path}")
    df = df.drop_duplicates(subset=['prompt'])
    print(f"After dropping duplicates: {len(df)} samples from {input_path}")
    print('Example of multihier:')
    print(df.iloc[0])
    ds = [{
        "id": f"multihier-{i}",
        "question": data["prompt"][0]['content'],
        "reward_model": data["reward_model"],
        "extra_info": data["extra_info"],
        "data_source": data["data_source"],
        "general_domain": 'table__multihier',
    } for i, data in tqdm(df.iterrows(), total=len(df))]
    print(f"Loaded {len(ds)} samples from multihier")
    return ds


def load_browsecomp(input_path: str = "" + SR2AM_AFM_BENCHMARKS_ROOT + "/browsecomp.json"):
    print('Loading browsecomp...')
    df = pd.read_json(input_path)
    print('Example of browsecomp:')
    print(df.iloc[0])

    ds = [{
        "id": f"browsecomp-{i}",
        "question": data["question"].strip() + " You should provide your final answer in the format \\boxed{YOUR_ANSWER}.",
        "answer": data["answer"],
        "extra_info": {
            'original_question': data['question'],
            'topic': data['topic'],
        },
        "general_domain": 'web__browsecomp',
    } for i, data in tqdm(df.iterrows(), total=len(df))]
    print(f"Loaded {len(ds)} samples from browsecomp")
    return ds


def load_hle(input_path: str = "" + SR2AM_AFM_BENCHMARKS_ROOT + "/hle_test.json"):
    print('Loading hle...')
    df = pd.read_json(input_path)
    print('Example of hle:')
    print(df.iloc[0])

    ds = [{
        "id": f"hle-{i}-{data['id']}",
        "question": data["question"].strip() + " You should provide your final answer in the format \\boxed{YOUR_ANSWER}.",
        "answer": data["answer"],
        "extra_info": {
            'original_question': data['question'],
            'answer_type': data['answer_type'],
            'rationale': data['rationale'],
            'raw_subject': data['raw_subject'],
            'category': data['category'],
        },
        "general_domain": 'web__hle',
    } for i, data in tqdm(df.iterrows(), total=len(df))]
    print(f"Loaded {len(ds)} samples from hle")
    return ds


def load_gaia(input_path: str = "" + SR2AM_AFM_BENCHMARKS_ROOT + "/gaia_dev_103.json"):
    print('Loading gaia...')
    df = pd.read_json(input_path)
    print('Example of gaia:')
    print(df.iloc[0])

    ds = [{
        "id": f"gaia-{i}-{data['id']}",
        "question": data["question"].strip() + " You should provide your final answer in the format \\boxed{YOUR_ANSWER}.",
        "answer": data["answer"],
        "extra_info": {
            'original_question': data['question'],
            'level': data['Level'],
            'annotator_metadata': data['Annotator_Metadata'],
        },
        "general_domain": 'web__gaia',
    } for i, data in tqdm(df.iterrows(), total=len(df))]
    print(f"Loaded {len(ds)} samples from gaia")
    return ds


def load_xbench_deepsearch(
    data_path: str = "" + SR2AM_XBENCH_ROOT + "/DeepSearch-2505-decrypted.csv",
):
    print('Loading xbench_deepsearch...')
    df = pd.read_csv(data_path)
    ds = [{
        "id": f"xbench-deepsearch-{row['id']}",
        "question": str(row["prompt"]).strip() + " You should provide your final answer in the format \\boxed{YOUR_ANSWER}.",
        "answer": str(row["answer"]),
        "extra_info": {"reference_steps": row.get("reference_steps", "")},
        "general_domain": "web__xbench_deepsearch",
        "qwen3_32b_pass_rate": 0.0,
    } for _, row in tqdm(df.iterrows(), total=len(df))]
    print(f"Loaded {len(ds)} samples from {data_path}")
    return ds


def create_test_dataset_full(
    output_file: str = "data/sr2am_test_full.jsonl",
):
    ds = []
    for dataset in [load_aime24, load_aime25, load_math500,
                    load_gpqa_diamond, load_supergpqa, load_hle,
                    load_finqa, load_multihier,
                    load_browsecomp, load_gaia, load_xbench_deepsearch]:
        ds_dataset = dataset()

        dataset_name = ds_dataset[0]['general_domain']
        if dataset_name in ['math__aime24', 'math__aime25']:
            num_repeats = 32
        elif dataset_name in ['stem__gpqa_diamond', 'web__gaia', 'web__xbench_deepsearch']:
            num_repeats = 4
        else:
            num_repeats = 1

        print(f"Repeating {dataset_name} {num_repeats} times")
        ds_dataset = ds_dataset * num_repeats

        ds.extend(ds_dataset)
    df = pd.DataFrame(ds)
    print('General domain distribution:')
    print(df.general_domain.value_counts())
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    df.to_json(output_file, orient='records', lines=True, force_ascii=False)
    print(f"Created {len(ds)} samples from test dataset and saved to {output_file}")


if __name__ == "__main__":
    fire.Fire()
