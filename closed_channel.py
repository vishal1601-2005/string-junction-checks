"""
Closed-channel (open-closed duality) checks for the planar trivalent junction, sigma=1, 120 deg, d spatial dims.
Open channel: Z = Tr exp(-2 pi R H) built from the roots of the characteristic functions + zero-point energy from the
determinant formula.  Closed channel: s-wave contact prediction (Komargodski-Zhong), generalized to arbitrary arm lengths.
"""
import numpy as np
from scipy.integrate import quad, dblquad
from scipy.special import kv

# ------------------------------------------------------------------ open-channel pieces
def Erel_planar(L, M, d):
    L=np.array(L,float)
    def f(y):
        c=1/np.tanh(y*L); my=M*y
        out=np.log((my+c.sum())/(my+3.0))
        inn=np.log((my**2+my*c.sum()+0.75*(c[0]*c[1]+c[1]*c[2]+c[2]*c[0]))/(my+1.5)**2)
        return ((d-2)*out+inn)/(2*np.pi)
    s=L.min()
    a,_=quad(f,0,0.05/s,limit=400); b,_=quad(f,0.05/s,40/s,limit=800)
    return a+b
def E0abs(L,M,d):
    L=np.array(L,float)
    return Erel_planar(L,M,d)-(d-1)*np.pi/24*np.sum(1/L)

def roots_planar(L,M,wmax,h=1e-4):
    L=np.array(L,float); k=np.arange(1e-6,wmax,h)
    s=np.sin(np.outer(k,L)); c=np.cos(np.outer(k,L))
    P=s.prod(1)
    Phi_o=M*k*P-(c[:,0]*s[:,1]*s[:,2]+c[:,1]*s[:,0]*s[:,2]+c[:,2]*s[:,0]*s[:,1])
    Phi_i=(M*k)**2*P-M*k*(c[:,0]*s[:,1]*s[:,2]+c[:,1]*s[:,0]*s[:,2]+c[:,2]*s[:,0]*s[:,1]) \
          +0.75*(c[:,0]*c[:,1]*s[:,2]+c[:,0]*c[:,2]*s[:,1]+c[:,1]*c[:,2]*s[:,0])
    def refine(Ph):
        i=np.where(np.sign(Ph[:-1])*np.sign(Ph[1:])<0)[0]
        a=k[i].copy(); b=k[i+1].copy()
        def F(x):
            ss=np.sin(np.outer(x,L)); cc=np.cos(np.outer(x,L)); PP=ss.prod(1)
            t1=cc[:,0]*ss[:,1]*ss[:,2]+cc[:,1]*ss[:,0]*ss[:,2]+cc[:,2]*ss[:,0]*ss[:,1]
            t2=cc[:,0]*cc[:,1]*ss[:,2]+cc[:,0]*cc[:,2]*ss[:,1]+cc[:,1]*cc[:,2]*ss[:,0]
            return Ph_kind(x,PP,t1,t2)
        return a,b,F
    out=[]
    for kind in ('o','i'):
        Ph=Phi_o if kind=='o' else Phi_i
        i=np.where(np.sign(Ph[:-1])*np.sign(Ph[1:])<0)[0]
        a=k[i].copy(); b=k[i+1].copy()
        def F(x,kind=kind):
            ss=np.sin(np.outer(x,L)); cc=np.cos(np.outer(x,L)); PP=ss.prod(1)
            t1=cc[:,0]*ss[:,1]*ss[:,2]+cc[:,1]*ss[:,0]*ss[:,2]+cc[:,2]*ss[:,0]*ss[:,1]
            t2=cc[:,0]*cc[:,1]*ss[:,2]+cc[:,0]*cc[:,2]*ss[:,1]+cc[:,1]*cc[:,2]*ss[:,0]
            if kind=='o': return M*x*PP-t1
            return (M*x)**2*PP-M*x*t1+0.75*t2
        fa=F(a)
        for _ in range(60):
            m=0.5*(a+b); fm=F(m); same=np.sign(fm)==np.sign(fa); a=np.where(same,m,a); b=np.where(same,b,m); fa=np.where(same,fm,fa)
        r=0.5*(a+b)
        # count check: roots below wmax vs sum floor
        exp=int(np.sum(np.floor(wmax*L/np.pi)))
        out.append((r,len(r)-exp))
    return out

def lnZ_open(L,R,M,d,wmax=None):
    wmax = wmax or 40.0/(2*np.pi*R)
    (ro,co),(ri,ci)=roots_planar(L,M,wmax)
    th=-np.sum(np.log1p(-np.exp(-2*np.pi*R*ro)))*(d-2) - np.sum(np.log1p(-np.exp(-2*np.pi*R*ri)))*2/2*1.0
    # NB: for d=3 there is 1 out-of-plane channel; for general d the (d-2) out-of-plane channels share roots.
    # in-plane: Phi_in is the 2x2 determinant (two channels) -> its roots carry the full in-plane tower once.
    return -2*np.pi*R*E0abs(L,M,d)+th,(co,ci)

# ------------------------------------------------------------------ closed-channel prediction
def lnZ_pred(L,R,d,g0=None):
    L=np.array(L,float)
    detT=(3.0)**(d-2)*(9/4)             # sigma=1, planar 120 deg, equal tensions: out (3)^(d-2), in (3/2)^2
    if g0 is None: g0=np.sqrt(detT)/(2*np.pi)**(d/2)
    detTL=(np.sum(1/L))**(d-2)*0.75*(1/(L[0]*L[1])+1/(L[1]*L[2])+1/(L[0]*L[2]))
    return (np.log(g0)+np.sum((d-1)/2*np.log(np.pi*R/L)+(d-1)*L/(12*R))-d/2*np.log(R)-0.5*np.log(detTL))

if __name__=="__main__":
    d=3; R=1.0
    print("=== (1) near-equal arms (1,1.002,1.004)*L: open-channel numerics vs Komargodski-Zhong modular form (27) at mean L ===")
    for L in (2.0,3.0,4.0):
        Ls=L*np.array([1.0,1.002,1.004])/1.002
        lz,(co,ci)=lnZ_open(Ls,R,0.0,d,)
        qt=np.exp(-2*L/R); n=np.arange(1,400)
        KZ=((d-1.5)*np.log(np.pi*R/L)-d/2*np.log(2)-(d-1)/8*np.log(qt)
            -d*np.sum(np.log1p(-qt**(2*n)))-(d-3)*np.sum(np.log1p(-qt**n)))
        print(f"   L/R={L:.1f}: ln Z numerics {lz:.8f}   KZ(27) {KZ:.8f}   diff {lz-KZ:+.2e}   root-count offsets {co,ci}")
    print("\n=== (2) UNEQUAL arms, M=0: ratio Z_open / Z_closed-channel-prediction (should -> 1 as q~ -> 0) ===")
    shape=np.array([1.0,1.2360679,1.5811388])
    for s in (2.0,3.0,4.0,6.0,8.0):
        L=s*shape
        lz,(co,ci)=lnZ_open(L,R,0.0,d)
        lp=lnZ_pred(L,R,d)
        qa=np.exp(-2*L/R); qxy=qa[0]*qa[1]
        print(f"   L_a={np.round(L,3)}: ln(ratio)={lz-lp:+.3e}   q~_x q~_y={qxy:.2e}  ln(ratio)/(q~_x q~_y)={(lz-lp)/qxy:.2f}   root offsets {co,ci}")
    print("\n=== (3) O(M) term: d lnZ/dM (quantum part) vs tr T^-1/(12 R)  [5/36 = 0.138889 for d=3, sigma=1; exactly equal arms are skipped: degenerate roots are invisible to sign-change root finding] ===")
    dM=0.02
    for label,shp in (("near-eq",np.array([1.,1.002,1.004])/1.002),("unequal",shape)):
        for s in (3.0,5.0,8.0):
            L=s*shp
            lp,_=lnZ_open(L,R,+dM,d); lm,_=lnZ_open(L,R,-dM,d)
            print(f"   {label:8s} L_a={np.round(L,2)}: dlnZ/dM = {(lp-lm)/(2*dM):.5f}")
