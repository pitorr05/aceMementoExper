"""Explicit lexical similarity utility; no external embedding service is used.

The upstream helper's hosted embedding dependency is replaced for all-local runs.
This utility is a documented baseline adaptation, not Qwen embedding inference.
"""
from collections import Counter
import re
import numpy as np


def compute_embedding_similarity(a, b):
    def vector(text): return Counter(re.findall(r'\w+',text.lower()))
    av,bv=list(map(vector,a)),list(map(vector,b))
    return np.array([[sum(v[k]*w[k] for k in v)/(sum(x*x for x in v.values())**.5*
                     sum(x*x for x in w.values())**.5 or 1) for w in bv] for v in av])
