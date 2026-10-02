import numpy as np
from scipy.integrate import quad

def roots_junction(Sig, M, R, nmax):
    """roots k_n R = x_n of  x sin x - a cos x = 0, a = Sig R / M, x in (n pi, n pi + pi/2), n=0..nmax"""
    a = Sig*R/M
    n = np.arange(nmax+1)
    lo = n*np.pi; hi = n*np.pi+np.pi/2
    sg = (-1.0)**n
    for _ in range(80):
        mid = 0.5*(lo+hi)
        h = sg*(mid*np.sin(mid) - a*np.cos(mid))
        pos = h > 0
        hi = np.where(pos, mid, hi); lo = np.where(pos, lo, mid)
    return 0.5*(lo+hi)/R

def DeltaE(Sig, M, R, eps):
    kmax = 45/eps
    nmax = int(kmax*R/np.pi)+5
    k = roots_junction(Sig, M, R, nmax)
    kD = np.arange(1, nmax+1)*np.pi/R
    f = lambda q: q*np.exp(-eps*q)
    # pair k_n (n>=1) with DD_n ; k_0 unpaired
    return 0.5*(f(k[0]) + np.sum(f(k[1:]) - f(kD)))

def Erel(Sig, M, R):
    def integrand(y):
        r = (M*y-Sig)/(M*y+Sig); u = np.exp(-2*y*R)
        return np.log1p(-r*u) - np.log1p(-u)
    v1,_ = quad(integrand, 0, 1e-3, limit=200)
    v2,_ = quad(integrand, 1e-3, 60/R, limit=400)
    return (v1+v2)/(2*np.pi)

if __name__=="__main__":
    M, Sig = 0.5, 3.0
    Rs = [3.0, 5.0, 8.0, 12.0]
    print("Single channel, M=0.5, Sigma=3.  Mode-sum with smooth cutoff exp(-eps k) vs eq.(6)")
    print("Erel(R) from quadrature:", {R: round(Erel(Sig,M,R),8) for R in Rs})
    print()
    print("eps    DeltaE(R)-DeltaE(R=12) [mode sum]  vs  Erel(R)-Erel(12) [eq.6]")
    for eps in (0.08, 0.04, 0.02, 0.01):
        d12 = DeltaE(Sig,M,12.0,eps); e12 = Erel(Sig,M,12.0)
        row=[]
        for R in Rs[:-1]:
            row.append((DeltaE(Sig,M,R,eps)-d12, Erel(Sig,M,R)-e12))
        print(f"{eps:5.2f}  "+"   ".join(f"R={R:>4.0f}: {a:+.6f} vs {b:+.6f}" for R,(a,b) in zip(Rs[:-1],row)))
    print()
    # absolute offset C(eps) = DeltaE - Erel (should be R independent); look at log-divergence coefficient Sig/(2 pi M)
    print("offset C(eps)=DeltaE-Erel at different R (should be R-independent as eps->0), and slope vs ln eps")
    prev=None
    for eps in (0.08,0.04,0.02,0.01):
        c=[DeltaE(Sig,M,R,eps)-Erel(Sig,M,R) for R in Rs]
        print(f"eps={eps:5.2f}  C(R) = "+" ".join(f"{x:+.6f}" for x in c))
        if prev is not None:
            print(f"          dC/dln(1/eps) ~ {(c[1]-prev)/np.log(2):+.5f}   (predicted Sig/(2 pi M) = {Sig/(2*np.pi*M):+.5f})")
        prev=c[1]
