import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, required=True)
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--num_reps', type=int, default=3)
args = parser.parse_args()

data_df = pd.read_json(args.input_file, lines=True)
data_df.correct.fillna(0.0, inplace=True)
print(data_df.head())
print('Total number of questions: ', len(data_df))

# compute the pass rate of the data_df
pass_rate = data_df['correct'].mean()
# print(f"Pass rate: {pass_rate * 100:.2f}%")
print(f"Pass rate: {pass_rate:.4f}")

# compute the pass rate of the data_df by dataset
pass_rate_by_dataset = data_df.groupby('dataset')['correct'].mean()
print("Pass rate by dataset:")
# print((pass_rate_by_dataset * 100).round(2).astype(str) + "%")
print(pass_rate_by_dataset.round(4).astype(str))

# Also print the average pass rate across datasets
if not pass_rate_by_dataset.empty:
    avg_pass_rate_by_dataset = pass_rate_by_dataset.mean()
    # print(f"Average pass rate across datasets: {avg_pass_rate_by_dataset * 100:.2f}%")
    print(f"Average pass rate across datasets: {avg_pass_rate_by_dataset:.4f}")
else:
    print("No datasets found for computing average pass rate.")


# Compute the pass@k rate (default k=3, but can be changed via argument)
# For each question (grouped by dataset and question), take the first k answers,
# and if any of those k are correct, count it as pass@k.

import sys

k = 3  # Default value
if hasattr(args, 'k') and args.k is not None:
    k = args.k
else:
    # Also allow --k in sys.argv as a fallback (for backwards compatibility or external scripts)
    for idx, arg in enumerate(sys.argv):
        if arg == '--k' and idx + 1 < len(sys.argv):
            k = int(sys.argv[idx + 1])

print(f"Using pass@{k}")

group_keys = ['dataset', 'question']
grouped = data_df.groupby(group_keys)

pass_at_k_total = 0
num_questions = 0

from collections import defaultdict

# We want to compute pass@k by picking all possible sets of k distinct "reps" for each (dataset, question) group,
# with stride = k (i.e., non-overlapping windows), up to num_reps (default 4).
# For consistency, assume that reps are stored in order (e.g. from rep_idx or just order in file).
# Only consider groups where the total number of rows >= k.
# For example, if num_reps=4 and k=3, we consider windows:
# [0,1,2], [3] is ignored since need full k elements.

num_repeats = getattr(args, "num_reps", 4)

def compute_pass_at_k_windowed(grouped, k, num_repeats):
    total_sets, total_passed = 0, 0
    window_results, num_corrects_per_window = [], []

    for _, df_group in grouped:
        df_group = df_group.sort_values(by="rep_idx" if "rep_idx" in df_group else df_group.index)
        n = min(len(df_group), num_repeats)
        for start in range(0, n, k):
            end = start + k
            if end > n:
                continue
            window = df_group.iloc[start:end]
            is_pass = window['correct'].any()
            total_sets += 1
            if is_pass:
                total_passed += 1
            window_results.append(int(is_pass))
            num_corrects_per_window.append(window['correct'].sum())

    win_pass_rate = total_passed / total_sets if total_sets > 0 else 0.0

    if window_results:
        avg_pass_per_window = sum(window_results) / len(window_results)
        # print(f"[Windowed] Average Pass@{k} across windows: {avg_pass_per_window * 100:.2f}%")
    else:
        print(f"No windows of size k={k} found for Pass@k calculation.")
    return win_pass_rate, total_passed, total_sets

win_pass_rate, total_passed, total_sets = compute_pass_at_k_windowed(grouped, k, num_repeats)
# print(f"Pass@{k} rate: {win_pass_rate * 100:.2f}% ({total_passed}/{total_sets})")
print(f"Pass@{k} rate: {win_pass_rate:.4f} ({total_passed}/{total_sets})")

# By dataset
def compute_pass_at_k_windowed_by_dataset(data_df, k, num_repeats):
    pass_at_k_w_by_dataset = {}
    for dataset, df_dataset in data_df.groupby('dataset'):
        # if dataset == 'web__browsecomp':
        #     print(df_dataset.correct)
        #     print(df_dataset)
        grouped_ds = df_dataset.groupby('question')
        rate, tot, cnt = compute_pass_at_k_windowed(grouped_ds, k, num_repeats)
        pass_at_k_w_by_dataset[dataset] = (rate, tot, cnt)
    print(f"[Windowed] Pass@{k} with windows by dataset (num_repeats={num_repeats}):")
    for dataset, (rate, tot, cnt) in pass_at_k_w_by_dataset.items():
        # print(f"  {dataset}: {rate * 100:.2f}% ({tot}/{cnt})")
        print(f"  {dataset}: {rate:.4f} ({tot}/{cnt})")
        
    if pass_at_k_w_by_dataset:
        avg_pass_rate_by_dataset = sum(rate for rate, _, _ in pass_at_k_w_by_dataset.values()) / len(pass_at_k_w_by_dataset)
        # print(f"Average pass@{k} across datasets: {avg_pass_rate_by_dataset * 100:.2f}%")
        print(f"Average pass@{k} across datasets: {avg_pass_rate_by_dataset:.4f}")
    else:
        print(f"No datasets found for computing average pass@{k} rate.")

compute_pass_at_k_windowed_by_dataset(data_df, k, num_repeats)


# for group, df_group in grouped:
#     df_topk = df_group.head(k)
#     if len(df_topk) < k:
#         print(f"Skipping group with fewer than {k} reps: {len(df_topk)}")
#         continue
#     num_questions += 1
#     if df_topk['correct'].any():
#         pass_at_k_total += 1

# pass_at_k_rate = pass_at_k_total / num_questions if num_questions > 0 else 0.0
# print(f"Pass@{k}: {pass_at_k_rate * 100:.2f}% ({pass_at_k_total}/{num_questions})")

# # Pass@k by dataset
# pass_at_k_by_dataset = {}
# for dataset, df_dataset in data_df.groupby('dataset'):
#     grouped_ds = df_dataset.groupby('question')
#     ds_num_questions = 0
#     ds_pass_at_k = 0
#     for _, df_group in grouped_ds:
#         df_topk = df_group.head(k)
#         if len(df_topk) < k:
#             print(f"Skipping group with fewer than {k} reps: {len(df_topk)}")
#             continue
#         ds_num_questions += 1
#         if df_topk['correct'].any():
#             ds_pass_at_k += 1
#     ds_rate = ds_pass_at_k / ds_num_questions if ds_num_questions > 0 else 0.0
#     pass_at_k_by_dataset[dataset] = (ds_rate, ds_pass_at_k, ds_num_questions)
# print(f"Pass@{k} by dataset:")
# for dataset, (rate, tot, cnt) in pass_at_k_by_dataset.items():
#     print(f"  {dataset}: {rate * 100:.2f}% ({tot}/{cnt})")

# def compute_pass_at_k(files: List[str]) -> None:
#     """
#     Compute pass@k across multiple JSON files.
    
#     Args:
#         files: List of file paths to JSON files
#     """
#     # Dictionary to store all attempts for each question
#     question_attempts = defaultdict(list)
#     # Dictionary to store attempts by data source
#     data_source_attempts = defaultdict(lambda: defaultdict(list))
    
#     # Load all files and collect question attempts
#     for filepath in files:
#         try:
#             data = load_json_file(filepath)
#             for item in data:
#                 if 'question' in item and 'score' in item:
#                     question = item['question']
#                     score = item['score']
#                     data_source = item.get('data_source', 'unknown')
#                     question_attempts[question].append(score)
#                     data_source_attempts[data_source][question].append(score)
#         except Exception as e:
#             print(f"Error loading file {filepath}: {e}")
#             continue
    
#     # Calculate pass@k for each question overall
#     total_questions = len(question_attempts)
#     passed_questions = 0
    
#     for question, scores in question_attempts.items():
#         # Check if any attempt for this question was correct (score > 0 or True)
#         if any(score for score in scores):
#             passed_questions += 1
    
#     # Calculate and print overall results
#     k = len(files)
#     pass_at_k_rate = passed_questions / total_questions if total_questions > 0 else 0
    
#     print(f"=== Overall Pass@{k} Results ===")
#     print(f"Total unique questions: {total_questions}")
#     print(f"Questions passed at K={k}: {passed_questions}")
#     print(f"Pass@{k} rate: {pass_at_k_rate:.4f} ({pass_at_k_rate * 100:.2f}%)")
    
#     # Calculate pass@k by data source
#     print(f"\n=== Pass@{k} Results by Data Source ===")
#     for data_source, source_questions in data_source_attempts.items():
#         source_total = len(source_questions)
#         source_passed = 0
        
#         for question, scores in source_questions.items():
#             if any(score for score in scores):
#                 source_passed += 1
        
#         source_pass_rate = source_passed / source_total if source_total > 0 else 0
#         print(f"{data_source}: {source_passed}/{source_total} = {source_pass_rate:.4f} ({source_pass_rate * 100:.2f}%)")
        
#     # Calculate average pass@k across data sources
#     if data_source_attempts:
#         data_source_rates = []
#         for data_source, source_questions in data_source_attempts.items():
#             source_total = len(source_questions)
#             source_passed = 0
            
#             for question, scores in source_questions.items():
#                 if any(score for score in scores):
#                     source_passed += 1
            
#             source_pass_rate = source_passed / source_total if source_total > 0 else 0
#             data_source_rates.append(source_pass_rate)
        
#         avg_pass_rate = sum(data_source_rates) / len(data_source_rates)
#         print(f"\nAverage Pass@{k} across data sources: {avg_pass_rate:.4f} ({avg_pass_rate * 100:.2f}%)")