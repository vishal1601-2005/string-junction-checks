"""
(1) c-number validation of the local displacement vertex.  A single Dirichlet-Dirichlet string of length L whose end is moved
    inward by a c-number delta: exact E(L-delta) = -c/(L-delta), c=(D-2) pi/24, so the O(delta^2) energy is -(D-2) pi delta^2/(24 L^3).
    Local vertex H3 = delta*(sigma/2) x'(0)^2 (energy density at the end, constraint included), <S^(2)>=0.  Second-order
    perturbation theory gives  E2 = -(delta^2 (D-2) /(2 L^2)) * (pi/L) * Sum_{n,m} n m/(n+m).
    With the vertex-separation regulator  Sum -> int_eps^inf ds [1/(4 sinh^2(s/2))]^2  the finite part is 1/12 (exact antiderivative):
    E2 = -(D-2) pi delta^2/(24 L^3): EXACT AGREEMENT.
(2) Universality of the sunset finite parts within the class of vertex-separation (point-splitting) regulators
    F_h(eps) = int_0^inf h(s/eps) g(s) ds, any smooth h with h(0)=0 faster than s^3 (so the regulated integral converges), h(inf)=1: g has only even powers (times logs), so no eps^0 term arises
    from the regulator; the constant is the Hadamard finite part (=0).  Numerical test with two different profiles h.
"""
import sympy as sp, numpy as np, mpmath as mp
s,e=sp.symbols('s e',positive=True)
c=sp.coth(s/2)
F=(c-c**3/3)/8-sp.Rational(1,12)    # antiderivative of 1/(16 sinh^4(s/2)), vanishing at infinity
print("(1) check antiderivative:",sp.simplify(sp.diff(F,s)-1/(16*sp.sinh(s/2)**4)))
Fe=sp.series(-F.subs(s,e),e,0,3)   # int_eps^inf = F(inf)-F(eps) = -F(eps)
print("    int_eps^inf [1/(4 sinh^2(s/2))]^2 ds =",Fe,"  -> finite part =",sp.series(-F.subs(s,e),e,0,2).removeO().coeff(e,0))
print("    => E2 = -(D-2)*(1/2)*(pi/12)*delta^2/L^3 = -(D-2) pi delta^2/(24 L^3)  (exact: -(D-2) pi delta^2/(24 L^3))")

mp.mp.dps=50
lc=lambda x: mp.log(mp.coth(x/4))
g={'A0':lambda x: lc(x)*(mp.cosh(x/2)/(4*mp.sinh(x/2)**2))**2,
   'B0':lambda x: lc(x)*(1/(4*mp.sinh(x/2)**2))**2}
profiles={'h1(x)=exp(-1/x^2)':lambda x: mp.e**(-1/(x*x)),'h2(x)=tanh(x)^6':lambda x: mp.tanh(x)**6}
print("\n(2) point-splitting regulators: constant term C of int h(s/eps) g(s) ds (fit incl. ln terms)")
for name,h in profiles.items():
    for key,gg in g.items():
        epss=[mp.mpf(k)/1000 for k in (6,8,10,13,17,22,28,36)]
        rows=[];rhs=[]
        for ee in epss:
            val=mp.quad(lambda x: h(x/ee)*gg(x),[0,ee,5*ee,0.05,0.5,2,8,30,120])
            rows.append([mp.log(ee)/ee**3,1/ee**3,mp.log(ee)/ee,1/ee,1,ee*mp.log(ee),ee,ee**2*mp.log(ee)]); rhs.append(val)
        x=mp.lu_solve(mp.matrix(rows),mp.matrix(rhs))
        print(f"    {name:20s} {key}: C = {mp.nstr(x[4],4)}")
