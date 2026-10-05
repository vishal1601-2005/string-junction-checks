"""
Second, independent regulator for the three sunset mode sums (Appendix C): per-mode Gaussian cutoff exp(-(eps*omega)^2)
instead of the total-energy cutoff exp(-eps*E) of displacement_sector.py.  Units L = pi (omega_r = r-1/2, omega_n = n).
F(eps) = int_0^inf ds [product of three single-mode sums with e^{-s*omega} chi(omega)];
its eps->0 expansion is fitted to  a1 ln(eps)/eps^3 + a2/eps^3 + a3 ln(eps)/eps + a4/eps + C + a5 eps ln(eps) + a6 eps + a7 eps^2 ln(eps)
and the universal L^-3 coefficient is C.  (Extended precision: numpy longdouble sums, mpmath least squares.)
"""
import numpy as np, mpmath as mp
mp.mp.dps=40
LD=np.longdouble
x,wq=np.polynomial.legendre.leggauss(48)
def gl(a,b):
    return (LD(b)-LD(a))/2*LD(1)*x.astype(LD)+(LD(b)+LD(a))/2, (LD(b)-LD(a))/2*wq.astype(LD)
def F(eps,which):
    eps=LD(eps); nmax=int(9/float(eps))+5
    r=np.arange(1,nmax,dtype=LD); wN=r-LD(0.5); wD=r
    chiN=np.exp(-(eps*wN)**2); chiD=np.exp(-(eps*wD)**2)
    segs=[0,eps,5*eps,LD(0.5),2,8,25,80]
    tot=LD(0)
    for a,b in zip(segs[:-1],segs[1:]):
        s,w=gl(a,b)
        E=np.exp(-np.outer(s,wN))*chiN; Ed=np.exp(-np.outer(s,wD))*chiD
        S0=E.sum(1); S1=(E/wN).sum(1); S2=(E*wN).sum(1); T2=(Ed*wD).sum(1)
        g={'A0':S1*S2**2,'A1':S0**2*S2,'B0':S1*T2**2}[which]
        tot+=(w*g).sum()
    return tot
def fit(which,epss):
    rows=[]; rhs=[]
    for e in epss:
        e=mp.mpf(e); rows.append([mp.log(e)/e**3,1/e**3,mp.log(e)/e,1/e,1,e*mp.log(e),e,e**2*mp.log(e)]); rhs.append(mp.mpf(str(F(e,which))))
    return mp.lu_solve(mp.matrix(rows),mp.matrix(rhs))
if __name__=="__main__":
    epss=[0.06,0.07,0.08,0.09,0.10,0.115,0.13,0.15,0.17,0.20]
    for w in ('A0','A1','B0'):
        c=fit(w,epss); c2=fit(w,epss[1:-1]); 
        print(f"{w}: C = {mp.nstr(c[4],5)}  (fit on 8 points: {mp.nstr(c2[4],5)});  a1..a4 = {[mp.nstr(v,6) for v in c[:4]]}")
