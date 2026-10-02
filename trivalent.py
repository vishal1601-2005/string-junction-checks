import numpy as np
rng=np.random.default_rng(3)
print("trivalent junction, arbitrary tensions (triangle inequality), planar, force balance from tension triangle")
for _ in range(4):
    while True:
        s=rng.uniform(0.5,2.0,3)
        if s[0]<s[1]+s[2] and s[1]<s[0]+s[2] and s[2]<s[0]+s[1]: break
    # arms: angle between arm a and b: cos th_ab = (s_c^2 - s_a^2 - s_b^2)/(2 s_a s_b)
    th01=np.arccos((s[2]**2-s[0]**2-s[1]**2)/(2*s[0]*s[1]))
    th02=-np.arccos((s[1]**2-s[0]**2-s[2]**2)/(2*s[0]*s[2]))
    t=np.array([[1,0],[np.cos(th01),np.sin(th01)],[np.cos(th02),np.sin(th02)]])
    fb=np.linalg.norm((s[:,None]*t).sum(0))
    T=np.zeros((2,2))
    for a in range(3):
        n=np.array([-t[a,1],t[a,0]]); T+=s[a]*np.outer(n,n)
    ev=np.linalg.eigvalsh(T); S=s.sum()
    Q=S*(S-2*s[0])*(S-2*s[1])*(S-2*s[2])
    u=4*s.prod()/Q; v=4*s.prod()/(Q*S)
    print(f" sigma={np.round(s,3)} |force|={fb:.1e}  trTinplane^-1: num {np.sum(1/ev):.6f} formula {u:.6f} | trT^-2: num {np.sum(1/ev**2):.6f} formula {u*u-2*v:.6f} | tr T={ev.sum():.5f} sum sigma={S:.5f}")
