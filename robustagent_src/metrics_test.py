import numpy as np
from numpy.random import rand, triangular
import matplotlib.pyplot as plt
import time

A = (20,32,51)
B = (10,11,21)
P = 0.02

E_A = (A[0]+A[1]+A[2]) / 3
E_B = (B[0]+B[1]+B[2]) / 3
STD_A = ((A[0]-A[1])**2 + (A[1]-A[2])**2 + (A[1]-A[2])**2)**0.5 / 6
STD_B = ((B[0]-B[1])**2 + (B[1]-B[2])**2 + (B[1]-B[2])**2)**0.5 / 6

E_AB = E_A + P*E_B
E_AB2 = E_A + P*E_B + STD_A/50 + P*STD_B/50
E_AB3 = E_A

def rnd(tuple, p=None):
    if p != None and rand() > p:
        return 0
    return triangular(tuple[0], tuple[1], tuple[2], 1)[0]

def mean():
    v = list(map(lambda x: rnd(A)+rnd(B, P), range(1)))
    return np.mean(v)

def calc_rs(n, e):
    X = []
    for k in range(2222):
        for j in range(n):
            X.append({'j': j, 'k': k, 'dur':mean(), 'start': 0 if j==0 else X[-1]['end']})
            X[-1]['end'] = X[-1]['start']+X[-1]['dur']
    Z = []
    for i in range(n):
        o = list(filter(lambda u: u['j']==i, X))
        Z.append({'dur': np.mean(list(map(lambda u: u['dur'], o))),'end': np.mean(list(map(lambda u: u['end'], o))),'start': np.mean(list(map(lambda u: u['start'], o)))})

    Y = []
    for _ in range(n):
        Y.append({'dur':e, 'start': 0 if not Y else Y[-1]['end']})
        Y[-1]['end'] = Y[-1]['start']+Y[-1]['dur']




    R = (Y[-1]['end'] - Z[-1]['end'])
    S = (sum(map(lambda i: abs(Y[i]['end']-Z[i]['end']), range(n))))
    print(f'n: {n} R:{R} S:{S}')
    return R,S

#r,s = calc_rs(10, E_AB)
#r,s = calc_rs(10, E_AB2)

R = []
S = []
for i in range(20):
    r,s = calc_rs(10, E_AB)
    R.append(r)
    S.append(s)
print("======= R / S")
print(np.mean(R))
print(np.mean(S))

R = []
S = []
for ii in range(20):
    r,s = calc_rs(10, E_A)
    R.append(r)
    S.append(s)
print("======= R / S")
print(np.mean(R))
print(np.mean(S))