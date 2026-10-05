"""
Exact translation (c-number displacement) Ward identities as a criterion for the regularisation.
D-end:  sum_{n,m>=1} n m/(n+m)               must have finite part  +1/12   (exact: E = -c/(L-delta))
N-end:  sum_{r,r'>=1} w w'/(w+w'), w=r-1/2   must have finite part  -1/24   (exact: E = +pi/(48 (L-delta)))
Tested for (a) vertex-separation / total-energy cutoff exp(-eps E)  [exact antiderivatives]  and
(b) per-mode Gaussian cutoff exp(-eps^2 (w^2+w'^2)).
"""
import numpy as np, mpmath as mp
LD=np.longdouble
def S(eps,half):
    nmax=int(9/eps)+5
    k=np.arange(1,nmax,dtype=LD)-(LD(0.5) if half else LD(0))
    chi=np.exp(-(LD(eps)*k)**2)
    a=k*chi
    W=a[:,None]*a[None,:]/(k[:,None]+k[None,:])
    # note: chi factors multiply: (k chi)(k' chi')/(k+k')
    return W.sum()
def fit(half,epss):
    rows=[];rhs=[]
    for e in epss:
        e=mp.mpf(e); rows.append([1/e**3,1/e,1,e,e**2,e**3]); rhs.append(mp.mpf(str(S(e,half))))
    return mp.lu_solve(mp.matrix(rows),mp.matrix(rhs))
mp.mp.dps=40
epss=[0.05,0.06,0.07,0.08,0.09,0.10,0.12]
for half,name,exact in ((False,"D-end (integers)",mp.mpf(1)/12),(True,"N-end (half-integers)",-mp.mpf(1)/24)):
    c=fit(half,epss)
    print(f"{name}: per-mode Gaussian finite part = {mp.nstr(c[2],6)}   exact = {mp.nstr(exact,6)}   leading coefficients 1/eps^3, 1/eps: {mp.nstr(c[0],5)}, {mp.nstr(c[1],5)}")
print("vertex-separation scheme (exact antiderivatives): D-end 1/12, N-end -1/24  -> both exact")
