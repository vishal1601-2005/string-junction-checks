import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq

rng = np.random.default_rng(7)

def random_balanced(N, rng, planar=False):
    """N unit vectors t_a in R^3 and tensions s_a with sum s_a t_a = 0."""
    while True:
        if planar:
            ang = rng.uniform(0, 2*np.pi, N-1)
            t = np.stack([np.cos(ang), np.sin(ang), 0*ang], 1)
        else:
            v = rng.normal(size=(N-1, 3)); t = v/np.linalg.norm(v, axis=1)[:, None]
        s = rng.uniform(0.6, 1.6, N-1)
        f = (s[:, None]*t).sum(0)
        sN = np.linalg.norm(f)
        if sN > 0.3:
            t = np.vstack([t, -f/sN]); s = np.append(s, sN)
            return t, s

def transverse_basis(t):
    # 3x2 orthonormal basis of plane perpendicular to t
    a = np.array([1., 0, 0]) if abs(t[0]) < 0.9 else np.array([0, 1., 0])
    e1 = np.cross(t, a); e1 /= np.linalg.norm(e1)
    e2 = np.cross(t, e1)
    return np.stack([e1, e2], 1)

def build(t, s, R, M, n, half_bead=True):
    N = len(s)
    dim = 3 + N*n*2
    K = np.zeros((dim, dim)); Mm = np.zeros((dim, dim))
    T = np.zeros((3, 3))
    for a in range(N):
        B = transverse_basis(t[a]); P = B @ B.T
        T += s[a]*P
        h = R[a]/(n+1)
        base = 3 + a*n*2
        # kinetic for beads
        for j in range(n):
            idx = base + 2*j
            Mm[idx:idx+2, idx:idx+2] += s[a]*h*np.eye(2)
        # junction half-bead correction (trapezoid rule) -> O(1/n^2)
        if half_bead:
            Mm[:3, :3] += 0.5*s[a]*h*P
        # springs
        c = s[a]/h
        # u0 = B^T w
        # term |u1 - B^T w|^2
        i1 = base
        # K blocks
        K[i1:i1+2, i1:i1+2] += c*np.eye(2)
        K[:3, :3] += c*(B@B.T)
        K[i1:i1+2, :3] += -c*B.T
        K[:3, i1:i1+2] += -c*B
        for j in range(n-1):
            i = base + 2*j; k = base + 2*(j+1)
            K[i:i+2, i:i+2] += c*np.eye(2)
            K[k:k+2, k:k+2] += c*np.eye(2)
            K[i:i+2, k:k+2] += -c*np.eye(2)
            K[k:k+2, i:i+2] += -c*np.eye(2)
        il = base + 2*(n-1)
        K[il:il+2, il:il+2] += c*np.eye(2)
    Mm[:3, :3] += M*np.eye(3)
    return K, Mm, T

def Fdet(k, t, s, R, M):
    N = len(s); Tk = np.zeros((3, 3))
    for a in range(N):
        B = transverse_basis(t[a]); P = B @ B.T
        Tk += s[a]*P/np.tan(k*R[a])
    return np.linalg.det(M*k*np.eye(3) - Tk)

def Phi(k, t, s, R, M):
    # entire characteristic function: prod sin^2 * det
    pre = np.prod([np.sin(k*r)**2 for r in R])
    return pre*Fdet(k, t, s, R, M)

def analytic_roots_equal(Sig, M, R, nlev):
    """roots of tan(kR)=Sig/(M k), one per level n=0..nlev-1"""
    out = []
    for n in range(nlev):
        f = lambda k: M*k*np.sin(k*R) - Sig*np.cos(k*R)
        out.append(brentq(f, n*np.pi/R+1e-12, (n+0.5)*np.pi/R))
    return np.array(out)

if __name__ == "__main__":
    M = 0.5
    print("=== Test A: equal arms R=5, random non-planar balanced junctions (D=4) ===")
    for N in (3, 4, 5, 7):
        t, s = random_balanced(N, rng)
        R = np.full(N, 5.0)
        n = 150
        out = {}
        for hb in (False, True):
            K, Mm, T = build(t, s, R, M, n, half_bead=hb)
            w2 = eigh(K, Mm, eigvals_only=True)
            out[hb] = np.sqrt(np.abs(w2))
        Sig = np.linalg.eigvalsh(T)
        # sum rule check
        print(f"N={N}: force balance |sum s t| = {np.linalg.norm((s[:,None]*t).sum(0)):.1e}, "
              f"tr T={np.trace(T):.6f}  (D-2)*sum s={2*s.sum():.6f}")
        print("   eig(T) =", np.round(Sig, 5))
        # analytic dynamical roots for levels 0,1,2
        for i, S in enumerate(Sig):
            ar = analytic_roots_equal(S, M, 5.0, 3)
            # nearest discrete eigenvalues
            for hb in (False, True):
                d = [out[hb][np.argmin(abs(out[hb]-x))] for x in ar]
                err = np.array(d)-ar
                print(f"   channel {i} (Sigma={S:.4f}) halfbead={hb}: analytic {np.round(ar,5)}  err {np.array2string(err, formatter={'float_kind':lambda x:'%+.2e'%x})}")
        # tower multiplicity at k=pi/R: count discrete eigenvalues within window of pi/R
        kD = np.pi/5.0
        w = out[True]
        cnt = np.sum(abs(w-kD) < 0.004)
        print(f"   Dirichlet-tower multiplicity at k=pi/R: found {cnt}, predicted N(D-2)-(D-1) = {2*N-3}")
    print()
    print("=== Test B: unequal arms, characteristic function Phi(k)=prod sin^2 * det[Mk - sum s cot P] ===")
    for N in (3, 4):
        t, s = random_balanced(N, rng)
        R = np.array([4.0, 5.0, 6.3, 5.4][:N])
        n = 300
        K, Mm, T = build(t, s, R, M, n, half_bead=True)
        w = np.sort(np.sqrt(np.abs(eigh(K, Mm, eigvals_only=True))))
        # find roots of Phi on fine grid in (0, 0.9)
        ks = np.linspace(1e-4, 0.9, 40001)
        vals = np.array([Phi(k, t, s, R, M) for k in ks])
        sgn = np.sign(vals)
        idx = np.where(sgn[:-1]*sgn[1:] < 0)[0]
        roots = np.array([brentq(lambda k: Phi(k, t, s, R, M), ks[i], ks[i+1]) for i in idx])
        sel = w[w < 0.9][:len(roots)]
        print(f"N={N}, R={R}: #roots(Phi)={len(roots)}, #discrete modes below 0.9={np.sum(w<0.9)}")
        print("   max |disc - analytic| over matched modes:", np.max(abs(sel[:len(roots)]-roots)))
        print("   first 6 analytic:", np.round(roots[:6], 5))
        print("   first 6 discrete:", np.round(sel[:6], 5))
