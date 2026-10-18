import sys; sys.path.insert(0, 'research/factor_alchemy')
import numpy as np, pandas as pd
from causal_worker import process_factor
from concurrent.futures import ProcessPoolExecutor

rng = np.random.default_rng(1)
n = 200000
cols = {f'f{i}': rng.random(n) for i in range(86)}
X_all = pd.DataFrame(cols)
W = rng.random((n, 8)).astype(np.float32)
Y = rng.random(n).astype(np.float32)
fic = {}
tasks = [(f'f{i}', X_all, Y, W, fic, 0.05, 50) for i in range(8)]
print('2-WORKER process_factor GPU test', flush=True)
with ProcessPoolExecutor(max_workers=2) as ex:
    for name, r in ex.map(process_factor, tasks):
        print(name, round(r['ate'], 4), 'sig=', r['significant'], flush=True)
print('ALL DONE', flush=True)
