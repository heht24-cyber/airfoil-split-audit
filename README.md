# Airfoil Split Audit

Can two UIUC airfoil filenames put the same geometry in training and testing? This audit measures the overlap directly on the official Selig coordinate archive and produces a train/test manifest with no matching pair across the split at the chosen tolerance.

## Findings

The ZIP contains 1,665 coordinate files. The parser accepted 1,622 single-contour records and logged 43 exclusions. In a random split of 324 test files, the expected exposure is the fraction of test files with at least one training neighbor. The last column is the measured largest connected-component size.

| Maximum vertical difference (% chord) | Pairs | Expected exposure | Largest group |
| ---: | ---: | ---: | ---: |
| Numerical equality (1e-12 chord) | 6 | 0.59% | 2 |
| 0.01% | 10 | 0.99% | 2 |
| 0.05% | 11 | 1.09% | 2 |
| 0.10% | 18 | 1.78% | 2 |
| 0.20% | 46 | 4.27% | 3 |
| 0.50% | 763 | 31.04% | 130 |

All six equality pairs have computed distance zero. Their file stems are `fx711530 / fx71l150`, `goe523 / goe525`, `goe801 / goe802`, `m3 / nacam3`, `n64215 / naca642215`, and `rae2822 / rae69ck`. Coordinate equality establishes equality of these archived representations. Historical design identity and aerodynamic equivalence require separate evidence.

In 1,000 seeded 80/20 splits, exposure at 0.50% averaged **31.12%** with 5th–95th percentiles of 27.47–34.57%. Every grouped run generated a 324-file test partition with zero cross-split matching pairs at its own threshold. At 0.50%, the largest group contains `naca65210.dat` and `ag19.dat`, separated by **4.06% chord** under the same metric. A connected component constrains split assignment through chains of direct neighbors and can contain pairs much farther apart than the tolerance.

These are corpus-level geometry results under the stated metric. The audit uses no aerodynamic labels and estimates no model-error inflation. Exclusions consist of 36 nonmonotone surfaces, six trailing-edge endpoint mismatches and one contour without enough points on both sides of the leading edge.

## Reproduce and reuse

Download [coord_seligFmt.zip](https://m-selig.ae.illinois.edu/ads/archives/coord_seligFmt.zip) from the [UIUC Airfoil Coordinates Database](https://m-selig.ae.illinois.edu/ads/coord_database.html). The tested archive (UIUC page revision 2026-02-23) has SHA-256:

```text
5a17e5400322d7d1b0171554cee3764bcf34aa2cc349e0e4e95aeafe5062fda7
```

```sh
python -m pip install numpy scipy
python audit.py --self-test
python audit.py coord_seligFmt.zip --out results
```

- `summary.json` contains the archive fingerprint, expected and simulated overlap, and the diameter of the largest group with witness filenames.
- `pairs.csv` contains each matched pair and its distance, through the largest requested threshold.
- `groups_*.csv` contains one row per accepted file, its deterministic group ID and a seeded train/test assignment. Join by ZIP member path and pass the group ID to a grouped splitter if adding more folds. Choose a tolerance based on how your study defines unseen geometry.
- `excluded.csv` identifies each rejected file and the reason for exclusion.

## Method

The parser expects UIUC's Selig ordering, upper trailing edge to leading edge to lower trailing edge. It subtracts the leading-edge coordinates, divides both axes by the horizontal chord and preserves the file's axis orientation. Each surface is a piecewise-linear height function of x. The shorter trailing-edge x domain extends at constant height by up to 0.5% chord. This convention enables common-domain comparisons and may affect very close pairs with misaligned trailing edges.

The distance is the maximum absolute height difference over both surfaces and all chordwise positions. A fixed cosine grid screens candidate pairs using a lower bound. The full distance is then evaluated on the union of both curves' knots, where a piecewise-linear difference attains its maximum absolute value. Similarity means proximity of these polylines in this coordinate frame. Sampling density, alignment and the shape of the original continuous contour can affect comparisons.

For a file conditioned to be in the test set, all its `d` neighbors also land there with probability `∏(t−j)/(n−j)` for `j = 1,…,d`, where `t` is the fixed test size and `n` is the corpus size. The probability is zero when `d ≥ t`. Average one minus that probability over all files to get expected exposure. Group assignment keeps each connected component together and chooses a prefix of a seeded group permutation with test size closest to the target. For other inputs, the grouped test count can differ from the target, and when all files form one component there is no two-way grouped split.

The upstream coordinates are available from UIUC. The repository distributes the audit implementation and derived statistics.
