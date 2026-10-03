"""
Stability of the symmetric N-star against fusion of k arms (Sec. 9; extended in the revision).
First order: fusing the arm subset S into a k-string along u = sum_{j in S} t_j/|.| changes the energy by
        dE = l [ sigma_k - sigma |sum_{j in S} t_j| ] + O(l^2).
The maximum of |sum_S t_j| over |S| = k is attained by k ADJACENT arms: sin(k pi/N)/sin(pi/N).
Hence the star is first-order stable iff  sigma_k >= sigma sin(k pi/N)/sin(pi/N)  for all k  (the sine law saturates all of them).
Second order (k=2): with eps = 2 sigma cos(pi/N) - sigma_2,
        E_star - E_H,min = eps^2 R/sigma * [ 1/(4 sin^2(pi/N)) + 1/(N - 4 sin^2(pi/N)) ]  (eps>0 small),
and at eps=0 the star is a strict local minimum:  dE = sigma sin^2(pi/N) l^2/R > 0.
"""
import itertools, numpy as np
from scipy.optimize import minimize

def best_subset(N,k):
    th=2*np.pi*np.arange(N)/N; t=np.stack([np.cos(th),np.sin(th)],1)
    return max(np.linalg.norm(t[list(S)].sum(0)) for S in itertools.combinations(range(N),k))

def H_min(N,sig2,R=1.0,sigma=1.0):
    th=2*np.pi*np.arange(N)/N; q=R*np.stack([np.cos(th),np.sin(th)],1)
    def E(x):
        V1=x[:2]; Vc=x[2:]
        e=sigma*(np.linalg.norm(q[0]-V1)+np.linalg.norm(q[1]-V1))
        e+=sigma*sum(np.linalg.norm(q[j]-Vc) for j in range(2,N))
        return e+sig2*np.linalg.norm(V1-Vc)
    b=(q[0]+q[1]); b=b/np.linalg.norm(b)
    best=None
    for l in (0.02,0.05,0.1,0.2):
        for s in (0.0,0.02,0.1):
            r=minimize(E,np.concatenate([l*b,-s*b]),method='BFGS',options={'gtol':1e-13,'maxiter':2000})
            if best is None or r.fun<best.fun: best=r
    return best.fun

if __name__=="__main__":
    print("=== (1) max |sum_S t_j| over |S|=k vs sin(k pi/N)/sin(pi/N) ===")
    for N in (4,5,6,7,8):
        print(f"  N={N}: ",[(k,round(best_subset(N,k),6),round(np.sin(k*np.pi/N)/np.sin(np.pi/N),6)) for k in range(2,N-1)])
    print("\n=== (2) Casimir scaling vs sine-law bound sigma_k/sigma, all k ===")
    for N in (4,5,6,8,10):
        row=[]
        for k in range(2,N//2+1):
            cas=k*(N-k)/(N-1); sine=np.sin(k*np.pi/N)/np.sin(np.pi/N)
            row.append((k,round(cas,4),round(sine,4),"UNSTABLE" if cas<sine-1e-12 else "ok"))
        print(f"  N={N}: (k, Casimir, sine) =",row)
    print("\n=== (3) exact H minimisation vs second-order formula (R=sigma=1; star energy N) ===")
    for N in (4,5,6):
        th=np.pi/N; thr=2*np.cos(th)
        for eps in (0.02,0.05,0.1):
            sig2=thr-eps; Emin=H_min(N,sig2)
            pred=eps**2*(1/(4*np.sin(th)**2)+1/(N-4*np.sin(th)**2))
            print(f"  N={N} sigma2={sig2:.4f} (eps={eps}): gain exact={N-Emin:.6e}  second-order formula={pred:.6e}  ratio={(N-Emin)/pred:.4f}")
        Emin0=H_min(N,thr); print(f"  N={N} at threshold sigma2=2cos(pi/N): E_H,min - E_star = {Emin0-N:+.3e}  (>=0: star is the minimum)")
