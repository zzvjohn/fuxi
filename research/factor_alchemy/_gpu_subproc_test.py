import sys; sys.path.insert(0, 'research/factor_alchemy')
import numpy as np, time
from causal_worker import double_ml_ate
from concurrent.futures import ProcessPoolExecutor

def run():
    rng = np.random.default_rng(1)
    n = 200000
    W = rng.random((n, 8)).astype(np.float32)
    Y = rng.random(n).astype(np.float32)
    T = rng.random(n).astype(np.float32)
    print('SUBPROCESS double_ml_ate 200k starting', flush=True)
    t = time.time()
    r = double_ml_ate(T, Y, W, 50)
    print('SUBPROCESS DONE %.1fs' % (time.time() - t), flush=True)
    return r

if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(run)
        r = fut.result()
    print('RESULT', r, flush=True)
