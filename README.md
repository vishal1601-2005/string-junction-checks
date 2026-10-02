# string-junction-checks

Python scripts that reproduce every numerical statement in the manuscript
**"Quantum fluctuations of general confining-string junctions: spectrum, exact one-loop energy, and Z_N junctions"**
(V. H. Thakur). Each script prints its results; `run_all.sh` runs everything and stores the output in `results/`.

## Quick start
```bash
pip install -r requirements.txt
bash run_all.sh          # about a minute on a laptop
```
Tested with Python 3, numpy 2.4, scipy 1.17, sympy 1.14, mpmath 1.3.

## What each script checks
| Script | Checks | Manuscript |
|---|---|---|
| `modes.py` | Discretized bead-spring junction (random non-planar, unequal tensions, equal and unequal arms) vs. analytic mode conditions; Dirichlet-tower multiplicity N(D-2)-(D-1); sum rule tr T = (D-2) sum(sigma) | Thm. 1, Sec. 3 |
| `conv.py` | Discretization order: plain stencil O(1/n), half-bead junction O(1/n^2) | Sec. 3 |
| `zeta.py` | Cutoff mode sum vs. integral (equal arms); R-independent offset | Thm. 2, Sec. 4 |
| `extra_checks.py` | (a) log-divergence coefficient tr T/(2 pi M); (b) S-matrix weighted unitarity, Kirchhoff limit, det(1+SE)=0 at the roots; (c) fusion criterion and N=4 "H" minimization; (d) unequal arms with M != 0, mode sum vs. integral | Secs. 4, 7, 8 |
| `lz_check.py`, `lz_iso.py` | Determinant formula (M=0, N=3) vs. Lou-Zhong Table 2 and isosceles limits | Table 1 |
| `series_and_quartic.py` | Large-R series (residual ~ R^-4), star traces, general-N Nambu-Goto quartic algebra (symbolic) | Secs. 5, 6, App. B |
| `trivalent.py` | Closed forms for tr T^-1, tr T^-2 of trivalent k-string junctions | Sec. 5 |

Reference outputs from the version used in the paper are in `results/`.

## Notes
* The Lou-Zhong comparison values (Table 2 of arXiv:2602.17771) are typed into `lz_check.py`.
* Dimension convention: `D` = spacetime dimension; Lou-Zhong's `d` = D-1.

## Citation
See `CITATION.cff`. A DOI will be added after the first release.

## License
MIT (see `LICENSE`).
