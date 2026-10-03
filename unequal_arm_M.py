"""
Large-size expansion of the one-loop energy for ARBITRARY arm lengths (new in the revision; closes limitation (2)).

With arm lengths R_a = L r_a (min r_a = 1), the junction-excess energy from Theorem 2 expands as
    E_rel = E0(r)/L + M F1(r)/L^2 + M^2 F2(r)/L^3 + O(L^-4),
  E0 = (1/2pi) int dx ln[det A(x)/det T],           A(x) = sum_a s_a coth(x r_a) P_a
  F1 = (1/2pi) int dx x   [tr A^-1 - tr T^-1]
  F2 = -(1/4pi) int dx x^2 [tr A^-2 - tr T^-2]
Equal arms: F1 = -pi trT^-1/48, F2 = +pi trT^-2/48  (Prop. 1).
Planar trivalent equal tension (120 deg) in d=D-1 dims: tr A^-1 = (d-2)/C1 + 4 C1/(3 C2), C1=sum coth, C2=sum_{a<b} coth_a coth_b
(the summand of Lou-Zhong (3.15)).
"""
import numpy as np
from scipy.integrate import quad

def planar_star(d,N=3):
    th=2*np.pi*np.arange(N)/N
    n=np.stack([np.cos(th),np.sin(th)],1)
    return n

def make_P(t):
    dd=t.shape[1]; return [np.eye(dd)-np.outer(x,x) for x in t]

def Amat(x,r,s,P):
    return sum(s[a]/np.tanh(x*r[a])*P[a] for a in range(len(s)))

def coeffs(r,s,t, order=(0,1,2)):
    r=np.asarray(r,float); P=make_P(t); T=sum(s[a]*P[a] for a in range(len(s))); Ti=np.linalg.inv(T)
    brk=[0,1e-3,1e-1,1,5,20,80]
    def integ(f):
        return sum(quad(f,brk[i],brk[i+1],limit=400,epsabs=1e-13,epsrel=1e-12)[0] for i in range(len(brk)-1))
    out={}
    def f0(x):
        A=Amat(x,r,s,P); return np.linalg.slogdet(A)[1]-np.linalg.slogdet(T)[1]
    def f1(x):
        Ai=np.linalg.inv(Amat(x,r,s,P)); return x*(np.trace(Ai)-np.trace(Ti))
    def f2(x):
        Ai=np.linalg.inv(Amat(x,r,s,P)); return x*x*(np.trace(Ai@Ai)-np.trace(Ti@Ti))
    out['E0']=integ(f0)/(2*np.pi); out['F1']=integ(f1)/(2*np.pi); out['F2']=-integ(f2)/(4*np.pi)
    out['trTi']=np.trace(Ti); out['trTi2']=np.trace(Ti@Ti)
    return out

def Erel_full(r,s,t,M):
    r=np.asarray(r,float); P=make_P(t); T=sum(s[a]*P[a] for a in range(len(s))); dd=t.shape[1]
    brk=[0,1e-3,1e-1,1,5,20,80]
    def f(x):
        A=Amat(x,r,s,P); return (np.linalg.slogdet(M*x*np.eye(dd)+A)[1]-np.linalg.slogdet(M*x*np.eye(dd)+T)[1])
    return sum(quad(f,brk[i],brk[i+1],limit=400,epsabs=1e-13,epsrel=1e-12)[0] for i in range(len(brk)-1))/(2*np.pi)

if __name__=="__main__":
    print("=== (1) equal arms: F1 = -pi trT^-1/48, F2 = +pi trT^-2/48 ===")
    for d in (2,3,4):
        n=planar_star(d); t=np.zeros((3,d)); t[:,:2]=n; s=np.ones(3)
        c=coeffs([1,1,1],s,t)
        print(f"  d={d}: F1={c['F1']:+.8f} (exact {-np.pi*c['trTi']/48:+.8f})  F2={c['F2']:+.8f} (exact {np.pi*c['trTi2']/48:+.8f})")
    print("\n=== (2) series vs full Theorem-2 integral at small M (unequal arms r=(1,1.236,1.581), D=4, N=3) ===")
    d=3; n=planar_star(d); t=np.zeros((3,d)); t[:,:2]=n; s=np.ones(3); r=[1,1.236,1.581]
    c=coeffs(r,s,t); E0=Erel_full(r,s,t,0.0)
    for M in (0.02,0.05,0.1):
        E=Erel_full(r,s,t,M); ser=E0+M*c['F1']+M*M*c['F2']
        print(f"  M={M}: full={E:+.10f} series={ser:+.10f} diff={E-ser:+.2e} (expect O(M^3)~{M**3:.1e})")
    print("\n=== (3) tables: coefficient F1(r), F2(r) for the Lou-Zhong ratios (equal tensions, 120 deg) ===")
    print("  E_GS = classical + M + E0/Lmin + M F1/Lmin^2 + M^2 F2/Lmin^3 ; arms Lmin*r_a")
    for D in (3,4):
        d=D-1; n=planar_star(d); t=np.zeros((3,d)); t[:,:2]=n; s=np.ones(3)
        print(f" D={D}:")
        for ratio in [(1,1,1),(1,1,2),(1,1,3),(1,3,6),(2,3,5)]:
            r=np.array(ratio,float)/min(ratio)
            c=coeffs(r,s,t)
            print(f"   {ratio}:  E0(rel,abs)=({c['E0']:+.4f}, {c['E0']-(d-1)*np.pi/24*np.sum(1/r):+.4f})  F1={c['F1']:+.5f}  F2={c['F2']:+.5f}")
    print("\n=== (4) unequal-tension k-string trivalent, D=4, arms (1,1,1) and (1,1.3,1.7): F1, trT^-1 ===")
    sg=np.array([1.0,1.0,1.3])  # force balance fixes angles via tension triangle
    s1,s2,s3=sg
    # directions: t3 opposite to s1 t1+s2 t2
    c12=(s3**2-s1**2-s2**2)/(2*s1*s2); ang=np.arccos(c12)
    t1=np.array([1,0.]); t2=np.array([np.cos(ang),np.sin(ang)]); v=-(s1*t1+s2*t2); t3=v/np.linalg.norm(v)
    tt=np.zeros((3,3)); tt[0,:2]=t1; tt[1,:2]=t2; tt[2,:2]=t3
    print("   force:",np.linalg.norm((sg[:,None]*tt).sum(0)))
    for r in ([1,1,1],[1,1.3,1.7]):
        c=coeffs(r,sg,tt); print(f"   r={r}: F1={c['F1']:+.5f}  (equal-arm value -pi trT^-1/48 = {-np.pi*c['trTi']/48:+.5f})  F2={c['F2']:+.5f}")
