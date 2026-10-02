import numpy as np
from scipy.integrate import quad
from scipy.special import spence

def Li2(x):  # real dilog for |x|<1 : spence(1-x)
    return spence(1-x)

def Erel_M0(Ls, d):
    """Excess over 3 decoupled Dirichlet strings, M=0, N=3 equal tensions at 120 deg, d spatial dims.
       channels: (d-2) out-of-plane + 2 in-plane (planar 2x2 block) -> d channels total"""
    L = np.array(Ls, float)
    def f(y):
        c = 1.0/np.tanh(y*L)
        out = np.log(c.sum()/3.0)
        inn = np.log((c[0]*c[1]+c[1]*c[2]+c[2]*c[0])/3.0)
        return ((d-2)*out + inn)/(2*np.pi)
    s = L.min()
    a,_ = quad(f, 0, 0.05/s, limit=400)
    b,_ = quad(f, 0.05/s, 40.0/s, limit=800)
    return a+b

def E0abs(Ls, d):
    L=np.array(Ls,float)
    return Erel_M0(Ls,d) - (d-1)*np.pi/24*np.sum(1.0/L)

if __name__=="__main__":
    print("Check vs Lou-Zhong Table 2:  E_GS^(0) * min(L)  =  A*d + B")
    table = {(1,1,2):(-0.159,0.323),(1,1,3):(-0.144,0.298),(1,3,6):(-0.083,0.179),(2,3,5):(-0.130,0.265)}
    for Ls,(A,B) in table.items():
        mn=min(Ls)
        e3=E0abs(Ls,3)*mn; e4=E0abs(Ls,4)*mn
        # solve for A,B from d=3,4
        Af=e4-e3; Bf=e3-3*Af
        print(f"L={Ls}:  mine A={Af:+.4f} B={Bf:+.4f}   | published A={A:+.3f} B={B:+.3f}")
    print()
    print("equilateral: E_abs*L vs  -(d-2) pi/16:", E0abs((1,1,1),3), -np.pi/16, E0abs((1,1,1),5), -3*np.pi/16)
    print()
    # isosceles limits  (5.3): L1<<L2 : -[(d-2)Li2(1/3)+Li2(-1/3)]/(4 pi L1);  (5.4) L1>>L2
    for d in (3,4):
        L1=1.0; L2=200.0
        mine = E0abs((L1,L2,L2),d)
        pred = -((d-2)*Li2(1/3)+Li2(-1/3))/(4*np.pi*L1)
        print(f"d={d} L1<<L2 (L2/L1=200): mine {mine:.5f}  (5.3) {pred:.5f}")
        L1=200.0; L2=1.0
        mine = E0abs((L1,L2,L2),d)
        pred = -(2*d-5)*np.pi/(48*L2) - ((d-2)*Li2(-1/3)+Li2(1/3))/(4*np.pi*L2)
        print(f"d={d} L1>>L2 (L1/L2=200): mine {mine:.5f}  (5.4) {pred:.5f}")
