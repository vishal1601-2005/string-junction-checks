"""
Casimir-force (junction-tadpole) correction at O(L^-3)  [new in the revision].

The longitudinal-displacement vertex S_disp^(1) of Lou-Zhong has a non-vanishing tadpole whenever sum_a t_a != 0:
the one-loop energy E0(W) depends on the classical junction position W, so the quantum equilibrium shifts by O(1/L)
and the energy changes by -1/2 g^T K^-1 g = O(L^-3), the SAME order as the M^2 and Nambu-Goto terms.
Closed form for equal arms R (any valence, tensions, directions; D-1 = d):
    g = -(pi/R^2) sum_a [ (D-2)/24 - s_a tr(T^-1 P_a)/16 ] t_a ,   K = T/R
    dE_F = -(pi^2/(2 R^3)) u^T T^-1 u ,   u = sum_a [ (D-2)/24 - s_a tr(T^-1 P_a)/16 ] t_a
(the angular derivative of E0 vanishes at equal arms; see paper).  Vanishes for the symmetric star and for equal-tension N=3.
This script checks g against a numerical gradient of the exact Theorem-2 energy E0(W) and dE_F against a numerical
minimisation of V(W) = sum s_a|q_a-W| + E0(W).
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize

def P_of(t): return [np.eye(len(x))-np.outer(x,x) for x in t]

def E0_abs(q,W,s,D):
    """exact one-loop energy (Thm 2, M=0) with the junction at W and static quarks at q_a"""
    v=q-W; R=np.linalg.norm(v,axis=1); t=v/R[:,None]; P=P_of(t)
    T=sum(s[a]*P[a] for a in range(len(s))); lT=np.linalg.slogdet(T)[1]
    def f(y):
        A=sum(s[a]/np.tanh(y*R[a])*P[a] for a in range(len(s)))
        return np.linalg.slogdet(A)[1]-lT
    sm=R.min(); brk=[0,1e-3/sm,1e-1/sm,1/sm,5/sm,20/sm,80/sm]
    I=sum(quad(f,brk[i],brk[i+1],limit=400,epsabs=1e-14,epsrel=1e-12)[0] for i in range(len(brk)-1))/(2*np.pi)
    return I-(D-2)*np.pi/24*np.sum(1/R)

def classical(q,W,s): return np.sum(s*np.linalg.norm(q-W,axis=1))

def junction_geometry(sg,planar_dim=3):
    s1,s2,s3=sg
    c12=(s3**2-s1**2-s2**2)/(2*s1*s2); ang=np.arccos(c12)
    t1=np.array([1,0.]); t2=np.array([np.cos(ang),np.sin(ang)]); v=-(s1*t1+s2*t2); t3=v/np.linalg.norm(v)
    t=np.zeros((3,planar_dim)); t[0,:2]=t1; t[1,:2]=t2; t[2,:2]=t3
    return t

def closed_form(t,s,R,D):
    P=P_of(t); T=sum(s[a]*P[a] for a in range(len(s))); Ti=np.linalg.inv(T)
    w=np.array([(D-2)/24-s[a]*np.trace(Ti@P[a])/16 for a in range(len(s))])
    u=(w[:,None]*t).sum(0); g=-(np.pi/R**2)*u
    K=T/R
    dE=-0.5*g@np.linalg.solve(K,g)
    return g,dE,u,T

if __name__=="__main__":
    D=4
    for sg in (np.array([1.,1.,1.]),np.array([1.,1.,1.3]),np.array([1.,1.4,1.0])):
        s=sg; t=junction_geometry(sg)
        print(f"\n=== tensions {sg}: force residual {np.linalg.norm((s[:,None]*t).sum(0)):.1e}, |sum t_a| = {np.linalg.norm(t.sum(0)):.4f} ===")
        for R in (20.0,40.0,80.0):
            q=R*t; W0=np.zeros(3)
            g,dE,u,T=closed_form(t,s,R,D)
            h=1e-3*R; gnum=np.zeros(3)
            for i in range(3):
                e=np.zeros(3); e[i]=h
                gnum[i]=(E0_abs(q,W0+e,s,D)-E0_abs(q,W0-e,s,D))/(2*h)
            K=T/R; delta=-np.linalg.solve(K,g)
            V=lambda W: classical(q,W,s)+E0_abs(q,W,s,D)
            dEnum=V(W0+delta)-V(W0)
            print(f"  R={R:5.0f}: |g_num - g_closed| = {np.linalg.norm(gnum-g):.2e} (|g|={np.linalg.norm(g):.2e});  "
                  f"dE_closed*R^3 = {dE*R**3:+.6f}   dE_numeric(shifted W)*R^3 = {dEnum*R**3:+.6f}")
