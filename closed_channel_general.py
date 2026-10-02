"""
(1) Check of my Gaussian/normalisation step against the exact Bessel propagators (d=2).
(2) General junction (non-planar, N=4, unequal tensions AND unequal arms, D=4): exact open-channel Z from the roots of the
    characteristic function vs the general closed-channel prediction
        Z_s-wave = g Prod_a (pi R/L_a)^{(d-1)/2} e^{(d-1)L_a/(12R)}  R^{-d/2} det(sum_a sigma_a P_a/L_a)^{-1/2},
        g = sqrt(det T)/(2 pi)^{d/2} [ x e^{-2 pi R M} (1 + M tr T^-1/(12R)) ].
"""
import numpy as np
from scipy.integrate import quad, dblquad
from scipy.special import kve
from modes import random_balanced, transverse_basis

# ---------------------------------------------------------------- (1) Bessel / Gaussian check (d=2, l_s=1, R=1)
def bessel_test(L, R=1.0):
    E=2*np.pi*R           # E^c ~ 2 pi R / l_s^2 with l_s=1
    q=np.array([[L*np.cos(a),L*np.sin(a)] for a in (0,2*np.pi/3,4*np.pi/3)])
    def f(y,x):
        r=np.sqrt((q[:,0]-x)**2+(q[:,1]-y)**2)
        return np.prod(E/np.sqrt(np.pi)*kve(0,E*r))*np.exp(-E*(r.sum()-3*L))
    I,_=dblquad(f,-4,4,-4,4,epsabs=1e-14,epsrel=1e-10)
    mine=(np.pi*R/L)**1.5*2*(L/R)/3          # (pi R/L)^{3(d-1)/2} * 2 (l_s^2 L/R)^{d/2}/3^{d/2}, e^{-3EL} stripped
    kz=mine*np.pi                              # factor pi^{d/2} difference in KZ eq.(33)
    return I,mine,kz

# ---------------------------------------------------------------- (2) general junction
def build_K(k,t,s,L,M,B):
    N=len(s); n=len(k)
    dim=2*N+3
    K=np.zeros((n,dim,dim))
    for a in range(N):
        K[:,2*a,2*a]=np.sin(k*L[a]); K[:,2*a+1,2*a+1]=np.sin(k*L[a])
        K[:,2*a:2*a+2,2*N:2*N+3]=-B[a].T[None]
        K[:,2*N:2*N+3,2*a:2*a+2]=-(k*s[a]*np.cos(k*L[a]))[:,None,None]*B[a][None]
    K[:,2*N:,2*N:]+=(M*k**2)[:,None,None]*np.eye(3)[None]
    return K
def roots_general(t,s,L,M,wmax,h=2e-4):
    B=[transverse_basis(t[a]) for a in range(len(s))]
    ks=np.arange(1e-5,wmax,h); out=[]
    for ch in np.array_split(ks,max(1,len(ks)//20000)):
        ch=np.append(ch,ch[-1]+h)
        d_=np.linalg.det(build_K(ch,t,s,L,M,B))/ch**3
        out.append((ch,d_))
    ch=np.concatenate([o[0][:-1] for o in out]); v=np.concatenate([o[1][:-1] for o in out])
    i=np.where(np.sign(v[:-1])*np.sign(v[1:])<0)[0]
    a=ch[i].copy(); b=ch[i+1].copy(); fa=v[i]
    for _ in range(50):
        m=0.5*(a+b); fm=np.linalg.det(build_K(m,t,s,L,M,B))/m**3
        same=np.sign(fm)==np.sign(fa); a=np.where(same,m,a); b=np.where(same,b,m); fa=np.where(same,fm,fa)
    r=0.5*(a+b)
    exp=2*int(np.sum(np.floor(wmax*np.array(L)/np.pi)))
    return r,len(r)-exp
def Pa(t): 
    return [np.eye(3)-np.outer(x,x) for x in t]
def Erel_general(t,s,L,M):
    P=Pa(t); T=sum(s[a]*P[a] for a in range(len(s)))
    def f(y):
        C=sum(s[a]/np.tanh(y*L[a])*P[a] for a in range(len(s)))
        return (np.linalg.slogdet(M*y*np.eye(3)+C)[1]-np.linalg.slogdet(M*y*np.eye(3)+T)[1])/(2*np.pi)
    sm=min(L)
    a,_=quad(f,0,0.05/sm,limit=400); b,_=quad(f,0.05/sm,40/sm,limit=800)
    return a+b
def lnZ_open(t,s,L,R,M,d=3):
    wmax=40.0/(2*np.pi*R)
    r,off=roots_general(t,s,L,M,wmax)
    E=Erel_general(t,s,L,M)-(d-1)*np.pi/24*np.sum(1/np.array(L))
    return -2*np.pi*R*E-np.sum(np.log1p(-np.exp(-2*np.pi*R*r))),off
def lnZ_pred(t,s,L,R,d=3):
    P=Pa(t); T=sum(s[a]*P[a] for a in range(len(s))); L=np.array(L)
    TL=sum(s[a]*P[a]/L[a] for a in range(len(s)))
    g0=np.sqrt(np.linalg.det(T))/(2*np.pi)**(d/2)
    return (np.log(g0)+np.sum((d-1)/2*np.log(np.pi*R/L)+(d-1)*L/(12*R))-d/2*np.log(R)-0.5*np.log(np.linalg.det(TL))), np.trace(np.linalg.inv(T))

if __name__=="__main__":
    print("=== (1) Gaussian-integral normalisation vs exact Bessel propagators (d=2, three arms at 120 deg, R=l_s=1) ===")
    for L in (4.0,8.0,16.0):
        I,mine,kz=bessel_test(L)
        print(f"   L={L:5.1f}:  numeric/mine = {I/mine:.4f}    numeric/(mine*pi) = {I/kz:.4f}   (pi={np.pi:.4f})")
    print("\n=== (2) general junction: N=4 non-planar, unequal tensions, unequal arms, D=4 (d=3), R=1 ===")
    rng=np.random.default_rng(21)
    t,s=random_balanced(4,rng)
    print("   tensions:",np.round(s,4)," |force|=%.1e"%np.linalg.norm((s[:,None]*t).sum(0)))
    shape=np.array([1.0,1.17,1.31,1.52])
    for sc in (3.0,4.0,6.0):
        L=sc*shape
        lz,off=lnZ_open(t,s,L,1.0,0.0)
        lp,trTi=lnZ_pred(t,s,L,1.0)
        sq=np.sum(np.exp(-2*L/1.0))
        print(f"   M=0, L_a={np.round(L,2)}: ln(Z_open/Z_pred) = {lz-lp:+.3e}   sum_a q~_a={sq:.2e}  ratio={(lz-lp)/sq:.2f}   root-count offset {off}")
    print("   O(M): d lnZ/dM  vs  trT^-1/12 =",round(trTi/12,6))
    dM=0.02
    for sc in (4.0,6.0,8.0):
        L=sc*shape
        lp_,_=lnZ_open(t,s,L,1.0,+dM); lm_,_=lnZ_open(t,s,L,1.0,-dM)
        print(f"      L_a={np.round(L,2)}: dlnZ/dM = {(lp_-lm_)/(2*dM):.6f}")
