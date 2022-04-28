from pathlib import Path
import os
import pandas as pd

files = next(os.walk(Path(".") / "experiment_results"), (None, None, []))[2] 
pass

for f in files:
    df = pd.read_csv(Path(".") / "experiment_results" / f, sep=';', encoding='utf-8', decimal=",")
    stats_path = Path(".") / "experiment_results" / f"{f}_stats"
    descr = df.describe()
    descr.to_csv(stats_path, sep=';', encoding='utf-8', header=True, index=True, decimal=",")
