import numpy as np, sympy as sp
from scipy.integrate import quad
from modes import random_balanced, transverse_basis
from zeta import Erel

rng=np.random.default_rng(11)
print("=== general-T series:  E = (D-1)pi/16R - pi M trT^-1/(48R^2) + pi M^2 trT^-2/(48R^3) + O(R^-4) ===")
for N in (3,5):
    t,s = random_balanced(N,rng)
    T=np.zeros((3,3))
    for a in range(N):
        B=transverse_basis(t[a]); T+=s[a]*B@B.T
    Sig=np.linalg.eigvalsh(T); M=0.5
    print(f"N={N} eig(T)={np.round(Sig,4)}  trT^-1={np.sum(1/Sig):.5f} trT^-2={np.sum(1/Sig**2):.5f}")
    prev=None
    for R in (5.,10.,20.,40.):
        ex=sum(Erel(S,M,R) for S in Sig)
        se=3*np.pi/(16*R)-np.pi*M*np.sum(1/Sig)/(48*R**2)+np.pi*M**2*np.sum(1/Sig**2)/(48*R**3)
        res=ex-se
        print(f"   R={R:5.0f}: exact {ex:.10f} series {se:.10f} resid {res:+.3e}"+("" if prev is None else f"  ratio prev/this={prev/res:.2f} (16 expected for R^-4)"))
        prev=res
print()
print("=== symmetric N-star: tr T^-1, trT^-2 vs eqs (10),(11) (D=4) ===")
for N in (3,4,6,8):
    th=2*np.pi*np.arange(N)/N
    T=np.zeros((3,3))
    for a in range(N):
        n=np.array([-np.sin(th[a]),np.cos(th[a]),0]); T+=np.outer(n,n)
    T[2,2]=N  # out-of-plane: sum sigma
    Sig=np.linalg.eigvalsh(T)
    print(f"N={N}: trT^-1={np.sum(1/Sig):.6f} (D+1)/(N sigma)={5/N:.6f} | trT^-2={np.sum(1/Sig**2):.6f} (D+5)/N^2={9/N**2:.6f}")
print()
print("=== NG-quartic O(L^-3) piece for the N-star (my Wick/zeta computation, using Lou-Zhong's coincident-limit integrals) ===")
d,N=sp.symbols('d N',positive=True)
Sm = (d-2)*(2-3/N) + (2-6/N)      # sum_i m_i , m_i = 2-3c_i ; c_out=1/N (d-2 comps), c_in=2/N (1 comp)
coef = -N*sp.pi**2*Sm**2/4608     # Delta E = coef * l_s^2 / L^3
print("coef =", sp.simplify(coef))
print("N=3 :", sp.simplify(coef.subs(N,3)), "  Lou-Zhong: -(d-2)^2 pi^2/1536 ->", sp.simplify(coef.subs(N,3)+(d-2)**2*sp.pi**2/1536))
print("N->oo:", sp.simplify(sp.limit(coef/N,N,sp.oo)), "  per-string; single DD string NG: -(d-1)^2 pi^2/1152 ->", sp.simplify(sp.limit(coef/N,N,sp.oo)+(d-1)**2*sp.pi**2/1152))
