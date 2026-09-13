#!/usr/bin/env python3
"""Compare complete teacher-forced logits from the two public-API probes."""
import array
import math
import struct
import sys

def read(path):
    with open(path,'rb') as f:
        vocab=struct.unpack('i',f.read(4))[0]
        values=array.array('f');values.frombytes(f.read())
    assert vocab>0 and len(values)==32*vocab
    assert all(math.isfinite(x) for x in values)
    return vocab,values

n,a=read(sys.argv[1]);m,b=read(sys.argv[2]);assert n==m
max_error=max(abs(x-y) for x,y in zip(a,b))
peak=max(max(abs(x) for x in a),max(abs(x) for x in b))
winners=sum(max(range(n),key=a[i*n:(i+1)*n].__getitem__)==max(range(n),key=b[i*n:(i+1)*n].__getitem__) for i in range(32))
exact=sum(x==y for x,y in zip(a,b))
print(f'Fixed-token comparison: rows=32 vocab={n} exact_values={exact}/{len(a)} matching_argmax={winners}/32 max_abs_error={max_error:.9g} peak={peak:.9g} scaled_error={max_error/max(1,peak):.9g}')
assert winners==32, 'different next-token winners on identical teacher-forced input'
assert max_error<=1e-4*max(1,peak), 'logit error exceeds scaled FP32 reference tolerance'
print('FIXED_TOKEN_LOGITS_PASS')
