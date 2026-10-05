# string-junction-checks

Python scripts that reproduce every numerical statement in the manuscript
**"Beyond the baryon: one-loop spectra and closed-string vertices of general confining-string junctions"**
(V. H. Thakur). Each script prints its results; `run_all.sh` runs everything and stores the output in `results/`.

## Quick start
```bash
pip install -r requirements.txt
bash run_all.sh          # a few minutes on a laptop
```
Tested with Python 3, numpy 2.4, scipy 1.17, sympy 1.14, mpmath 1.3.

Note: `closed_channel*.py` find mode frequencies by sign changes of the characteristic function, so exactly degenerate arm lengths (e.g. equal arms) are deliberately avoided there; near-equal arms are used instead.

## What each script checks
| Script | Checks | Manuscript |
|---|---|---|
| `modes.py` | Discretized bead-spring junction (random non-planar, unequal tensions, equal and unequal arms) vs. analytic mode conditions; Dirichlet-tower multiplicity N(D-2)-(D-1); sum rule tr T = (D-2) sum(sigma) | Thm. 1, Sec. 3 |
| `conv.py` | Discretization order: plain stencil O(1/n), half-bead junction O(1/n^2) | Sec. 3 |
| `zeta.py` | Cutoff mode sum vs. integral (equal arms); R-independent offset | Thm. 2, Sec. 4 |
| `extra_checks.py` | (a) log-divergence coefficient tr T/(2 pi M); (b) S-matrix weighted unitarity, Kirchhoff limit, det(1+SE)=0 at the roots; (c) fusion criterion and N=4 "H" minimization; (d) unequal arms with M != 0, mode sum vs. integral | Secs. 4, 7, 8 |
| `lz_check.py`, `lz_iso.py` | Determinant formula (M=0, N=3) vs. Lou-Zhong Table 2 and isosceles limits | Table 1 |
| `series_and_quartic.py` | Large-R series (residual ~ R^-4), star traces, general-N Nambu-Goto quartic algebra (symbolic) | Secs. 5, 6, App. B |
| `closed_channel.py` | Closed-channel tests for the planar trivalent junction: near-equal arms vs. Komargodski-Zhong (27); unequal arms vs. the s-wave prediction (locality of the vertex); O(M) coefficient vs. tr T^-1/12 | Sec. 8, Table 2 |
| `closed_channel_general.py` | Gaussian normalization vs. exact Bessel integral (d=2); generic non-planar N=4 junction with unequal tensions and arms vs. the general closed-channel prediction | Sec. 8, Table 2 |
| `trivalent.py` | Closed forms for tr T^-1, tr T^-2 of trivalent k-string junctions | Sec. 5 |

Reference outputs from the version used in the paper are in `results/`.

## Notes
* The Lou-Zhong comparison values (Table 2 of arXiv:2602.17771) are typed into `lz_check.py`.
* Dimension convention: `D` = spacetime dimension; Lou-Zhong's `d` = D-1.

## Citation
See `CITATION.cff`. A DOI will be added after the first release.

## License
MIT (see `LICENSE`).

## Reproducibility notes

- Old Table 2 (`closed_channel.py`, part 2) quoted `ln(ratio)/(q~_x q~_y)` at L/R = 6 and 8: those entries sit at the
  numerical-noise floor of the quadrature (1e-12 .. 1e-16) and are **not** meaningful; the revised Table 2 is produced by
  `exact_closed.py`, which compares against an exact analytic expression and reports residuals ~1e-15.
- `run_all.sh` regenerates every file in `results/`.
- Software versions: see `requirements.txt` (numpy, scipy, sympy, mpmath).

### Added in the revision (scripts and the sections they support)
- `displacement_sector.py` - sunset finite parts (220-digit), tadpole vs dE0/dR_a, Nambu-Goto Hadamard derivation and general-tension formula (Sec. 6, App. B, C)
- `sunset_second_regulator.py` - per-mode Gaussian regulator for the three sunset sums (regulator-dependence caveat, App. C)
- `cnumber_and_pointsplit.py` - c-number test of the displacement vertex (exact -(D-2) pi delta^2/(24 L^3)) and point-splitting universality with two profiles (App. C)
- `ward_identity_regulator_test.py` - exact translation Ward identities (finite parts +1/12 and -1/24) in two regularisation schemes (App. C)
