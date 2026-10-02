import numpy as np
from lz_check import E0abs, Li2
print("isosceles limits: difference to Lou-Zhong (5.3),(5.4) should shrink like 1/ratio")
for d in (3,4):
    for r in (200,2000,20000):
        mine = E0abs((1.0,float(r),float(r)),d)
        pred = -((d-2)*Li2(1/3)+Li2(-1/3))/(4*np.pi)
        mine2 = E0abs((float(r),1.0,1.0),d)
        pred2 = -(2*d-5)*np.pi/48 - ((d-2)*Li2(-1/3)+Li2(1/3))/(4*np.pi)
        print(f"d={d} ratio={r:6d}:  (5.3) diff {mine-pred:+.2e}   (5.4) diff {mine2-pred2:+.2e}")
