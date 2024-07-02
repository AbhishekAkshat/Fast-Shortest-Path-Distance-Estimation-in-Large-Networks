import dask.dataframe as dd
import pandas as pd
import sys
import os

def summarise(fpath):
    df = dd.read_table(fpath, sep='\t', blocksize='32MB')
    DROP_COLUMNS = ['random partition', \
                    'degree partition', 'closeness partition']
    df = df.drop(columns=DROP_COLUMNS)
    df = df.loc[df['source'] != df['target']]
    to_calculate = [col for col in df if col not in ['source', 'target', 'actual distance']]
    for col in to_calculate:
        df[col] = (df[col].sub(df['actual distance'])).abs().div(df['actual distance'])
    results = df[to_calculate].mean()
    results = results.compute()
    return results



folders = ["1", "10", "25", "50", "75", "100", "125", "150"]
cwd = os.getcwd()

summary = pd.DataFrame(columns=['Dataset', 'strategy', 'n_landmarks', 'mean error'])
for folder in folders:
    path = os.path.join(cwd, folder)
    for file in ("email.tsv", "road.tsv", "web.tsv", "musae.tsv", "twitch.tsv"):  #os.listdir(path):
        try:
            print(file)
            fpath = os.path.join(path, file)
            name = file[:-4]
            results = summarise(fpath)
            for result, strategy in zip(results, results.index):
                print(result, strategy)
                summary = summary.append({"Dataset": name, "strategy":strategy, "n_landmarks":folder, 'mean error': result}, ignore_index=True)
        except:
            continue
        print(summary)
summary.to_csv("summary.csv", index=False)
