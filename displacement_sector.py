"""
O(L^-3) ground-state energy: displacement sector and Nambu-Goto quartic term  [new in the revision]

Hamiltonian formulation.  Moving the junction by delta_a = t_a.w towards quark a shortens arm a, so (first order in the
vertex, M=0 propagators)    H3 = sum_a (t_a . w) eps_a(0),   eps_a = pi_a^2/(2 s_a) + s_a x_a'^2/2   (energy density).
 (1) tadpole:   <eps_a(0)> = (pi/R^2) [ s_a tr(T^-1 P_a)/16 - (D-2)/24 ]  == - dE0/dR_a      (checked numerically here)
 (2) sunset:    three-quantum intermediate states.  With 1/E = int ds e^{-sE} every mode sum factorises; the universal
                (eps^0, i.e. L^-3) coefficient of the three basic integrals is computed below with a smooth energy cutoff
                e^{-eps E}.  All three vanish -> the sunset has no ground-state contribution for ANY equal-arm junction.
 (3) NG quartic term, first-order perturbation theory with Hadamard finite part in sigma: reproduces (18) for general
                tensions/directions:   dE_NG = -(pi^2/(4608 R^3)) sum_a s_a^-1 [2(D-2) - 3 s_a tr(T^-1 P_a)]^2.
"""
import numpy as np, mpmath as mp, sympy as sp
import sys; sys.path.insert(0,'.')
from casimir_force import E0_abs, P_of
from modes import random_balanced

# ---------------------------------------------------------------- (2) universal parts of the sunset integrals
mp.mp.dps=220
lc=lambda s: mp.log(mp.coth(s/4))
integrands={
 'A0 (NND-type: int G (d_t d_t G)^2 )':lambda s: lc(s)*(mp.cosh(s/2)/(4*mp.sinh(s/2)**2))**2,
 'A1 (NND-type: int (d_t G)^2 d_t d_t G)':lambda s: (1/(2*mp.sinh(s/2)))**2*mp.cosh(s/2)/(4*mp.sinh(s/2)**2),
 'B0 (ND-D-D: int G (d_s d_s G_D)^2 )':lambda s: lc(s)*(1/(4*mp.sinh(s/2)**2))**2}
sing={'A0':(-1,2*mp.log(2),-mp.mpf(1)/12,mp.mpf(1)/48+mp.log(2)/6),
      'A1':(0,1,0,-mp.mpf(1)/24),
      'B0':(-1,2*mp.log(2),mp.mpf(1)/6,mp.mpf(1)/48-mp.log(2)/3)}
def finite_part(key,f):
    l4,c4,l2,c2=sing[key]
    fs=lambda s: l4*mp.log(s)/s**4+c4/s**4+l2*mp.log(s)/s**2+c2/s**2
    pts=[mp.mpf(10)**(-k) for k in range(24,-1,-2)]   # 1e-24 ... 1
    reg=mp.quad(lambda s:f(s)-fs(s),pts)
    tail=mp.quad(f,[1,5,20,mp.inf])
    fp=l4*(-mp.mpf(1)/9)+c4*(-mp.mpf(1)/3)+l2*(-1)+c2*(-1)
    return reg+tail+fp
def run_sunset():
    print("=== (2) universal (eps^0) part of the three sunset mode-sum integrals (smooth cutoff e^{-eps E}, L=pi units) ===")
    for (name,f),key in zip(integrands.items(),('A0','A1','B0')):
        print(f"   {name:46s}  finite part = {mp.nstr(finite_part(key,f),6)}")
    # exact antiderivative for A1: F(eps) = 1/(24 sinh^3(eps/2)), odd in eps -> no eps^0 term
    e=sp.symbols('e',positive=True); print("   A1 exact: int_eps^inf = 1/(24 sinh^3(eps/2)) -> series:",sp.series(1/(24*sp.sinh(e/2)**3),e,0,3))

# ---------------------------------------------------------------- (1) tadpole vs dE0/dR_a
def run_tadpole():
    print("\n=== (1) <eps_a(0)> closed form vs numerical dE0/dR_a (exact Theorem-2 energy), D=4 random non-planar N=4 ===")
    rng=np.random.default_rng(5); t,s=random_balanced(4,rng); D=4
    P=P_of(t); T=sum(s[a]*P[a] for a in range(4)); Ti=np.linalg.inv(T)
    R=30.0; q=R*t; W=np.zeros(3)
    def E(Rs):
        # exact E0 with independent arm lengths Rs at fixed directions t  (Theorem 2, M=0)
        from scipy.integrate import quad
        lT=np.linalg.slogdet(T)[1]
        def f(y):
            A=sum(s[a]/np.tanh(y*Rs[a])*P[a] for a in range(4)); return np.linalg.slogdet(A)[1]-lT
        sm=min(Rs); br=[0,1e-3/sm,1e-1/sm,1/sm,5/sm,20/sm,80/sm]
        I=sum(quad(f,br[i],br[i+1],limit=400,epsabs=1e-15,epsrel=1e-13)[0] for i in range(len(br)-1))/(2*np.pi)
        return I-(D-2)*np.pi/24*np.sum(1/np.array(Rs))
    for a in range(4):
        h=1e-3*R; Rp=np.full(4,R); Rm=Rp.copy(); Rp[a]+=h; Rm[a]-=h
        num=(E(Rp)-E(Rm))/(2*h)
        eps=(np.pi/R**2)*(s[a]*np.trace(Ti@P[a])/16-(D-2)/24)
        print(f"   arm {a}: dE0/dR_a = {num:+.10e}   -<eps_a> = {-eps:+.10e}   diff = {num+eps:+.1e}")

# ---------------------------------------------------------------- (3) NG quartic: Hadamard finite parts and coefficient
def run_ng():
    print("\n=== (3) Nambu-Goto quartic term: Hadamard finite parts int_0^pi du {1, cos u, cos^2 u}/sin^4 u ===")
    u,e=sp.symbols('u e',positive=True)
    for name,F in (("1/sin^4",-sp.cot(u)-sp.cot(u)**3/3),("cos/sin^4",-1/(3*sp.sin(u)**3)),
                   ("cos^2/sin^4",-sp.cot(u)-sp.cot(u)**3/3+sp.cot(u))):
        val=sp.series(F.subs(u,sp.pi-e)-F.subs(u,e),e,0,1).removeO()
        print(f"   FP int {name:12s} = {sp.simplify(val.subs(e,0)) if val.has(e) is False else sp.nsimplify(val.coeff(e,0))}   (odd powers of 1/eps dropped)")
    print("   => the connected part int (a_i-b_i)^2 has zero finite part; only (sum_i (a_i+b_i))^2 survives:")
    print("      a_i+b_i = -(pi/(24 L^2)) m_i,  m_i = 2-3c_i  ->  dE = -(pi^2/(4608 s L^3)) (sum_i m_i)^2 per arm")
    D=sp.symbols('D')
    for N in (3,4,5,6):
        sig=1; c_out=sp.Rational(1,N); c_in=sp.Rational(2,N)
        Sm=(D-3)*(2-3*c_out)+(2-3*c_in)
        tot=-sp.pi**2*N*Sm**2/(4608*sig)
        eq18=-sp.pi**2*(2*N*(D-2)-3*(D-1))**2/(4608*N*sig)
        print(f"   N={N}: per-arm sum m = {sp.simplify(Sm)};  total - eq.(18) = {sp.simplify(tot-eq18)}")
    print("   N=3: coefficient of pi^2 (D-3)^2/(sigma R^3):", sp.nsimplify(sp.simplify(-sp.pi**2*3*((D-3)*(2-3*sp.Rational(1,3))+(2-2))**2/4608/(sp.pi**2*(D-3)**2))), " (expect -1/1536)")
    print("   general-tension formula, random D=4 junction: dE_NG*R^3*sigma_min units:")
    rng=np.random.default_rng(3); t,s=random_balanced(5,rng); D_=4
    P=P_of(t); T=sum(s[a]*P[a] for a in range(5)); Ti=np.linalg.inv(T)
    tot=-np.pi**2/4608*sum((2*(D_-2)-3*s[a]*np.trace(Ti@P[a]))**2/s[a] for a in range(5))
    print(f"      -pi^2/4608 sum_a [2(D-2)-3 s_a tr(T^-1 P_a)]^2/s_a = {tot:+.8f}")
if __name__=="__main__":
    run_sunset(); run_tadpole(); run_ng()
