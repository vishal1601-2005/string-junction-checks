import numpy as np
from scipy.optimize import minimize, brentq
from scipy.integrate import quad
from modes import random_balanced, transverse_basis, Fdet
from zeta import DeltaE, Erel

# ---------- (a) log-divergence coefficient --------------------------------
print("(a) offset C(eps)=DeltaE-Erel ; slope vs ln(1/eps) -> Sigma/(2 pi M)")
M,Sig,R=0.5,3.0,5.0
prev=None
for eps in (0.01,0.005,0.0025,0.00125):
    c=DeltaE(Sig,M,R,eps)-Erel(Sig,M,R)
    if prev is not None: print(f"   eps={eps:.5f} C={c:.6f} slope={(c-prev)/np.log(2):.5f}  target {Sig/(2*np.pi*M):.5f}")
    prev=c

# ---------- (b) general S-matrix -------------------------------------------
print("\n(b) general S-matrix: weighted unitarity, Kirchhoff limit, quantisation det(1+S E)")
rng=np.random.default_rng(5)
N=4; t,s=random_balanced(N,rng); R=np.array([4.0,5.3,6.1,4.7]); M=0.5
B=[transverse_basis(t[a]) for a in range(N)]
L=np.vstack([B[a].T for a in range(N)])           # (2N x 3)
Sg=np.diag(np.repeat(s,2))                         # sigma weights
T=L.T@Sg@L
def S(k,M=M): return 2j*L@np.linalg.inv(M*k*np.eye(3)+1j*T)@L.T@Sg-np.eye(2*N)
for k in (0.1,0.7,3.0):
    Sk=S(k); print(f"   k={k}: ||S^dag Sg S - Sg|| = {np.linalg.norm(Sk.conj().T@Sg@Sk-Sg):.2e}")
Sk=S(0.7,M=1e-9); ev=np.sort(np.linalg.eigvals(Sk).real)
print("   M->0 eigenvalues of S: #(+1) =",int(np.sum(ev>0)),"(expect D-1=3), #(-1) =",int(np.sum(ev<0)),"(expect 2N-3=%d)"%(2*N-3))
print("   M->inf:", np.linalg.norm(S(0.7,M=1e12)+np.eye(2*N)) )
def Ephase(k): return np.diag(np.repeat(np.exp(2j*k*R),2))
def detq(k): return np.linalg.det(np.eye(2*N)+S(k)@Ephase(k))
# roots of det[Mk - sum s cot P]*prod sin^2 on a grid
def Phi(k): return np.prod(np.sin(k*R)**2)*Fdet(k,t,s,R,M)
ks=np.linspace(1e-3,0.8,20001); v=np.array([Phi(k) for k in ks]); idx=np.where(np.sign(v[:-1])*np.sign(v[1:])<0)[0]
roots=[brentq(Phi,ks[i],ks[i+1]) for i in idx][:6]
print("   |det(1+S E)| at roots of Phi:", np.array2string(np.array([abs(detq(r)) for r in roots]),precision=1), "  (typical |det| at generic k:", round(abs(detq(0.1234)),3),")")

# ---------- (c) classical stability of Z_N star under pair fusion ------------
print("\n(c) first-order criterion: star unstable to fusing two adjacent arms iff sigma_2 < 2 sigma cos(pi/N)")
for N_ in (4,5,6,8):
    cas=2*(N_-2)/(N_-1); sin_=np.sin(2*np.pi/N_)/np.sin(np.pi/N_); thr=2*np.cos(np.pi/N_)
    print(f"   N={N_}: Casimir sigma_2/sigma={cas:.4f}  sine-law={sin_:.4f}  threshold 2cos(pi/N)={thr:.4f}")
q=np.array([[1,0],[0,1],[-1,0],[0,-1]],float)
def EH(x,s2):
    V1,V2=x[:2],x[2:]
    return (np.linalg.norm(q[0]-V1)+np.linalg.norm(q[1]-V1)+np.linalg.norm(q[2]-V2)+np.linalg.norm(q[3]-V2)+s2*np.linalg.norm(V1-V2))
print("   N=4 'H' config (pairs (0,1),(2,3) fused), star energy = 4:")
for s2 in (1.30,1.36,np.sqrt(2),1.45,1.60):
    best=min((minimize(EH,np.array([a,a,-a,-a]),args=(s2,),method='Nelder-Mead',options=dict(xatol=1e-10,fatol=1e-12,maxiter=4000)) for a in (0.05,0.2,0.4)),key=lambda r:r.fun)
    print(f"      sigma_2={s2:.4f}: min E_H = {best.fun:.8f}   (<4 => star unstable)  V1={np.round(best.x[:2],4)}")

# ---------- (d) unequal arms, M!=0, scalar channel: mode sum vs integral ---------
print("\n(d) unequal arms, M=0.5, 3 arms sigma=1 (scalar channel): cutoff mode-sum vs integral")
M=0.5; base=np.array([4.0,5.2360679,6.7320508])
def Phi_s(k,Rv):
    s_=np.sin(np.outer(k,Rv)); c_=np.cos(np.outer(k,Rv))
    return M*k*np.prod(s_,1)-(c_[:,0]*s_[:,1]*s_[:,2]+c_[:,1]*s_[:,0]*s_[:,2]+c_[:,2]*s_[:,0]*s_[:,1])
def roots_s(Rv,kmax,h=1e-4):
    out=[]; ks=np.arange(1e-6,kmax,h*1.0)
    for ch in np.array_split(ks,max(1,len(ks)//2_000_000)):
        ch=np.append(ch,ch[-1]+h); v=Phi_s(ch,Rv); i=np.where(np.sign(v[:-1])*np.sign(v[1:])<0)[0]
        a=ch[i].copy(); b=ch[i+1].copy(); fa=v[i]
        for _ in range(60):
            m=0.5*(a+b); fm=Phi_s(m,Rv); same=np.sign(fm)==np.sign(fa); a=np.where(same,m,a); b=np.where(same,b,m); fa=np.where(same,fm,fa)
        out.append(0.5*(a+b))
    return np.concatenate(out)
def DE(Rv,eps):
    kmax=40/eps; r=roots_s(Rv,kmax)
    n_exp=int(sum(np.floor(kmax*Rv/np.pi)))
    # unpaired bulk-subtracted sum
    ref=sum(0.5*np.sum((np.arange(1,int(kmax*Ra/np.pi)+2)*np.pi/Ra)*np.exp(-eps*np.arange(1,int(kmax*Ra/np.pi)+2)*np.pi/Ra)) for Ra in Rv)
    return 0.5*np.sum(r*np.exp(-eps*r))-ref, len(r)-n_exp
def Er_s(Rv):
    f=lambda y: np.log((M*y+np.sum(1/np.tanh(y*Rv)))/(M*y+3.0))/(2*np.pi)
    a,_=quad(f,0,1e-3,limit=200); b,_=quad(f,1e-3,60/Rv.min(),limit=400); return a+b
lams=(1.0,1.5,2.0)
for eps in (0.08,0.04):
    res=[DE(base*l,eps) for l in lams]
    print(f"   eps={eps}: root-count offsets (should be 0 or 1): {[r[1] for r in res]}")
    for l,r in zip(lams[1:],res[1:]):
        print(f"      lambda={l}: [DE(l)-DE(1)] = {r[0]-res[0][0]:+.6f}   [Erel(l)-Erel(1)] = {Er_s(base*l)-Er_s(base):+.6f}")
