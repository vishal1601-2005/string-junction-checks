import numpy as np
from scipy.linalg import eigh
from modes import *
rng = np.random.default_rng(7)
M=0.5
t,s = random_balanced(4, rng)
R = np.full(4,5.0)
K0,M0,T0 = build(t,s,R,M,10)
Sig = np.linalg.eigvalsh(T0)
ar = [analytic_roots_equal(S,M,5.0,1)[0] for S in Sig]
print("analytic lowest roots:", np.round(ar,6))
print("n    | plain-stencil err (3 channels)         | half-bead err")
errs={}
for n in (25,50,100,200,400):
    row=[]
    for hb in (False,True):
        K,Mm,T = build(t,s,R,M,n,half_bead=hb)
        w = np.sqrt(np.abs(eigh(K,Mm,eigvals_only=True,subset_by_index=[0,8])))
        e=[w[np.argmin(abs(w-x))]-x for x in ar]
        row.append(e)
    errs[n]=row
    print(f"{n:4d} | "+" ".join(f"{x:+.3e}" for x in row[0])+" | "+" ".join(f"{x:+.3e}" for x in row[1]))
ns=sorted(errs)
for i in range(len(ns)-1):
    r0=np.array(errs[ns[i]][0])/np.array(errs[ns[i+1]][0]); r1=np.array(errs[ns[i]][1])/np.array(errs[ns[i+1]][1])
    print(f"n {ns[i]}->{ns[i+1]}: err ratio plain {np.round(r0,2)}  halfbead {np.round(r1,2)}")
