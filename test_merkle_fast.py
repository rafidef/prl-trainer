import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from prl_miner_tpu.merkle_fast import FastMerkleTree
import time
import numpy as np

data = b"0" * (1024 * 1024 * 10) # 10MB
key = b"0" * 32
t0 = time.time()
try:
    tree = FastMerkleTree(data, key)
    print("Success in", time.time() - t0)
except Exception as e:
    print("Exception:", e)
