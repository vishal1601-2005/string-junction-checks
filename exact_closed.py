"""
Exact open-closed identity for the quadratic junction model (new in the revision).

For ANY junction (valence N, tensions s_a, directions t_a, arm lengths L_a, junction mass M) the open-channel partition
function Z = Tr e^{-2 pi R H} equals, exactly,
    ln Z_open = ln Z_ground + sum_a -(d-1) sum_n ln(1-qt_a^n) - sum_{n>=1} ln[det(A_n + M n/R)/det(T + M n/R)],
    A_n = sum_a s_a coth(n L_a/R) P_a,   qt_a = exp(-2 L_a/R),
    ln Z_ground = ln Z_pred(M=0) + sum_i ln S(R Sigma_i/M),   S(x) = Gamma(1+x) e^x / (sqrt(2 pi) x^{x+1/2}).
(R-independent constants are absorbed in the renormalised junction mass; see paper Sec. 8.)
This script checks it against Z_open computed independently from the ROOTS of the characteristic function
(different code path: roots + Theorem 2 integral), and tabulates the first excited-state coefficient
c_a = (d-1) - 2 s_a tr(T^-1 P_a)  [ln Z_open/Z_ground = sum_a c_a qt_a + ...].
"""
import sys; sys.path.insert(0,'.')
import numpy as np
from scipy.special import gammaln
from closed_channel import lnZ_open as lnZ_open_planar, lnZ_pred as lnZ_pred_planar
from closed_channel_general import lnZ_open as lnZ_open_gen, lnZ_pred as lnZ_pred_gen, Pa
from modes import random_balanced

def lnS(x): return gammaln(1+x)+x-0.5*np.log(2*np.pi)-(x+0.5)*np.log(x)

def excited(t,s,L,M,R,d,nmax=80):
    P=Pa(t) if t.shape[1]==3 else None
    dd=t.shape[1]
    Pm=[np.eye(dd)-np.outer(x,x) for x in t]
    T=sum(s[a]*Pm[a] for a in range(len(s))); L=np.array(L,float); tot=0.0
    for n in range(1,nmax):
        A=sum(s[a]/np.tanh(n*L[a]/R)*Pm[a] for a in range(len(s)))
        my=M*n/R*np.eye(dd)
        tot-= np.linalg.slogdet(A+my)[1]-np.linalg.slogdet(T+my)[1]
    qt=np.exp(-2*L/R)
    ref=-(d-1)*sum(np.sum(np.log1p(-q**np.arange(1,nmax))) for q in qt)
    return ref+tot

def ccoef(t,s,d):
    dd=t.shape[1]
    Pm=[np.eye(dd)-np.outer(x,x) for x in t]
    T=sum(s[a]*Pm[a] for a in range(len(s))); Ti=np.linalg.inv(T)
    return np.array([(d-1)-2*s[a]*np.trace(Ti@Pm[a]) for a in range(len(s))])

if __name__=="__main__":
    d=3; R=1.0
    print("=== (A) planar trivalent, equal tensions 120 deg, D=4: roots-based Z_open vs exact identity ===")
    shape=np.array([1.0,1.2360679,1.5811388])
    th=np.array([0,2*np.pi/3,4*np.pi/3]); t3=np.stack([np.cos(th),np.sin(th),0*th],1); s3=np.ones(3)
    for M in (0.0,0.3,0.7):
        for sc in (2.0,3.0):
            L=sc*shape
            lo,_=lnZ_open_planar(L,R,M,d); lp=lnZ_pred_planar(L,R,d)
            if M>0: lp+= (d-2)*lnS(R*3.0/M)+2*lnS(R*1.5/M)
            ex=excited(t3,s3,L,M,R,d)
            print(f"  M={M:.1f} L={np.round(L,2)}  ln(open/ground)={lo-lp:+.4e}  identity={ex:+.4e}  residual={lo-lp-ex:+.1e}")
    print("\n=== (B) generic NON-planar N=4 (random tensions/directions/arms), D=4, roots vs identity ===")
    rng=np.random.default_rng(21); t,s=random_balanced(4,rng)
    Pm=Pa(t); T=sum(s[a]*Pm[a] for a in range(4)); Sig=np.linalg.eigvalsh(T)
    shape4=np.array([1.0,1.17,1.31,1.52])
    for M in (0.0,0.4):
        for sc in (3.0,4.0):
            L=sc*shape4
            lo,off=lnZ_open_gen(t,s,L,R,M); lp,_=lnZ_pred_gen(t,s,L,R)
            if M>0: lp+=sum(lnS(R*S/M) for S in Sig)
            ex=excited(t,s,L,M,R,d)
            print(f"  M={M:.1f} L={np.round(L,2)}  ln(open/ground)={lo-lp:+.4e}  identity={ex:+.4e}  residual={lo-lp-ex:+.1e}")
    print("\n=== (C) first-excitation coefficients c_a = (d-1) - 2 s_a tr(T^-1 P_a) ===")
    print("  generic N=4 junction above:  c_a =",np.round(ccoef(t,s,d),4)," (sum = N(d-1)-2d =",4*(d-1)-2*d,")")
    L=3.0*shape4; ex=excited(t,s,L,0.0,R,d); qt=np.exp(-2*L/R)
    print(f"  check: ln(open/ground) = {ex:.4e}  vs  sum_a c_a qt_a = {np.dot(ccoef(t,s,d),qt):.4e}")
    print("\n=== (D) selection-rule scan: symmetric N-star, c = (d-1) - 2d/N  (vanishes iff N(d-1)=2d) ===")
    for dd in (2,3,4,5,6):
        row=[(N,(dd-1)-2*dd/N) for N in range(3,9)]
        hit=[N for N,c in row if abs(c)<1e-12]
        print(f"  d={dd} (D={dd+1}): c(N=3..8) =",[round(c,3) for _,c in row]," vanishes for N =",hit)
    print("\n  explicit check d=2 (D=3), symmetric N=4 star: first-order term must vanish")
    dd=2; thN=2*np.pi*np.arange(4)/4; tN=np.stack([np.cos(thN),np.sin(thN)],1); sN=np.ones(4)
    print("   c_a =",np.round(ccoef(tN,sN,dd),12))
    for L0 in (2.0,3.0):
        L=L0*np.array([1.0,1.0,1.0,1.0]); ex=excited(tN,sN,L,0.0,R,dd); qt=np.exp(-2*L0/R)
        print(f"   L={L0}: ln(open/ground)={ex:.4e}   qt^2={qt**2:.4e}   ratio to qt^2 = {ex/qt**2:.3f}")
