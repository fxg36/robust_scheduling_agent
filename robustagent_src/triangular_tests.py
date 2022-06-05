import numpy as np
from numpy.random import rand, triangular
import matplotlib.pyplot as plt
import time

A = (2,10,25)
B = (2,6,15)
P = 0.4

start = time.time()



E_A = (A[0] + A[1] + A[2] ) / 3
E_B = (B[0] + B[1] + B[2] ) / 3

STD_A = ((A[0]-A[1])**2 + (A[1]-A[2])**2 + (A[1]-A[2])**2)**0.5 / 6
STD_B = ((B[0]-B[1])**2 + (B[1]-B[2])**2 + (B[1]-B[2])**2)**0.5 / 6

C = (A[0], E_A+P*E_B, A[2]+B[2])

E_C = (C[0] + C[1] + C[2] ) / 3
STD_C = ((C[0]-C[1])**2 + (C[1]-C[2])**2 + (C[1]-C[2])**2)**0.5 / 6


def get_triang_rnd(tuple):
    return triangular(tuple[0], tuple[1], tuple[2], 1)[0]


a1 = E_C + STD_C/2
a3 = E_A
a2 = (a1+a3)/2

print(a1)
print(a2)
print(a3)

# diff = E_C-STD_C
# print(E_A)
# print(diff)

V = []
for _ in range(2):
    v = get_triang_rnd(A)
    if rand() <= P:
        v += get_triang_rnd(B)
    V.append(v)

print(f"{min(V)}/{C[0]}")
print(f"{np.mean(V)}/{C[1]}")
print(f"{max(V)}/{C[2]}")

end = time.time() - start
print(end)

# plt.hist(V, bins='auto')  # density=False would make counts
# plt.ylabel('Probability')
# plt.xlabel('Data');
# plt.show()




