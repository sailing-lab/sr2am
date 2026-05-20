import pandas as pd
import json
import argparse

def main(args):
    dfs = []
    for judged_results_path in args.paths:
        with open(judged_results_path, 'r') as f:
            judged_results = json.load(f)
            
        df = pd.DataFrame(judged_results)
        dfs.append(df)
        
    df = pd.concat(dfs)
    if args.exclude_datasets:
        df = df[~df['data_source'].isin(args.exclude_datasets)]
    # print(df.head())
    print(args.paths)
    # print(df.iloc[0])
    
    # row = df.dropna(subset=['response']).query("data_source == 'stem'").sample().iloc[0]
    # print(row)
    # print('-'*100)
    # print('Question: ', row['extra_info']['question'])
    # print('-'*100)
    # print('Response: ', row['response'])
    # print('-'*100)
    # print('Ground truth: ', row['ground_truth'])
    # print('-'*100)
    # # print('Judgement: ', row['content'])
    # # print('-'*100)
    # print('Score: ', row['score'])
    # print('-'*100)
    
    # print((df.score == 1).mean())
    print(df.score.mean())
    score_groups = df.groupby('data_source')
    print("\nScore distribution:")
    score_distribution_df = pd.DataFrame({
        'mean': score_groups.score.mean(),
        'count': score_groups.size()
    })
    print(score_distribution_df)
    print('Average score across data sources: ', round(score_distribution_df['mean'].mean(), 4))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=str, nargs="+", required=True, help="Paths to judged results files")
    parser.add_argument("--exclude-datasets", type=str, nargs="+", required=False, help="Datasets to exclude")
    args = parser.parse_args()
    main(args)