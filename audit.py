"""Audit train/test overlap in UIUC Selig-format airfoil coordinates."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


DEFAULT_TOLERANCES = (1e-12, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3)
GRID = (1 - np.cos(np.linspace(0, np.pi, 65))) / 2


def profile(data):
    """Return two piecewise-linear surfaces in the file's chordwise frame."""
    rows = []
    seen_header = False
    for line in data.decode("latin-1").replace("\x1a", "").splitlines():
        fields = line.partition("#")[0].replace(",", " ").split()
        if not fields:
            continue
        try:
            point = tuple(map(float, fields))
        except ValueError:
            if rows or seen_header:
                raise ValueError("unexpected nonnumeric line") from None
            seen_header = True
            continue
        if len(point) != 2 or not np.isfinite(point).all():
            raise ValueError("expected two finite coordinates")
        if not rows or rows[-1] != point:
            rows.append(point)

    points = np.asarray(rows)
    if len(points) < 5:
        raise ValueError("too few coordinates")
    nose = int(np.argmin(points[:, 0]))
    if nose < 2 or nose >= len(points) - 2:
        raise ValueError("leading edge does not separate two surfaces")
    chord = max(points[0, 0], points[-1, 0]) - points[nose, 0]
    if chord <= 0 or abs(points[0, 0] - points[-1, 0]) > 0.005 * chord:
        raise ValueError("trailing-edge endpoints exceed 0.5% chord in x")
    points = (points - points[nose]) / chord
    upper, lower = points[nose::-1], points[nose:]
    for surface in (upper, lower):
        if np.any(np.diff(surface[:, 0]) <= 0):
            raise ValueError("surface is not strictly chordwise monotone")
    return upper, lower


def sample(surface, x):
    return np.interp(x, surface[:, 0], surface[:, 1])


def surface_distance(a, b):
    knots = np.union1d(a[:, 0], b[:, 0])
    return float(np.max(np.abs(sample(a, knots) - sample(b, knots))))


def matches(profiles, radius):
    # The grid distance lower-bounds the full piecewise-linear distance.
    grid_values = [np.concatenate([sample(s, GRID) for s in p]) for p in profiles]
    candidates = cKDTree(grid_values).query_pairs(
        radius + 1e-14, p=np.inf, output_type="ndarray"
    )
    distances = np.array([
        max(surface_distance(a, b) for a, b in zip(profiles[i], profiles[j]))
        for i, j in candidates
    ])
    accepted = distances <= radius
    pairs, distances = candidates[accepted], distances[accepted]
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    return pairs[order], distances[order]


def expected_exposure(degree, test_size):
    """Expected test-row fraction with a training neighbor in a fixed-size split."""
    n = len(degree)
    by_degree = np.ones(int(degree.max()) + 1)
    for d in range(1, min(len(by_degree), test_size)):
        by_degree[d] = by_degree[d - 1] * (test_size - d) / (n - d)
    by_degree[test_size:] = 0
    return float(np.mean(1 - by_degree[degree]))


def test_exposure(test, left, right):
    exposed = np.zeros(len(test), dtype=bool)
    exposed[left[test[left] & ~test[right]]] = True
    exposed[right[test[right] & ~test[left]]] = True
    return int(exposed.sum())


def audit_split(n, pairs, test_size, repeats, seed):
    left, right = pairs.T
    graph = coo_matrix((np.ones(len(pairs)), (left, right)), shape=(n, n))
    count, groups = connected_components(graph, directed=False)
    first = np.full(count, n)
    np.minimum.at(first, groups, np.arange(n))
    sizes = np.bincount(groups)
    degree = np.bincount(pairs.ravel(), minlength=n)
    rng = np.random.default_rng(seed)
    exposure = []
    for _ in range(repeats):
        test = np.zeros(n, dtype=bool)
        test[rng.choice(n, test_size, replace=False)] = True
        exposure.append(test_exposure(test, left, right) / test_size)

    if count == 1:
        grouped_test = None
        grouped_size = None
        grouped_exposed = None
    else:
        order = np.random.default_rng(seed).permutation(count)
        totals = np.cumsum(sizes[order])[:-1]
        cut = int(np.argmin(np.abs(totals - test_size))) + 1
        grouped_test = np.isin(groups, order[:cut])
        grouped_size = int(grouped_test.sum())
        grouped_exposed = test_exposure(grouped_test, left, right)
        if grouped_exposed:
            raise RuntimeError("component split has a cross-split pair")
    return first[groups], grouped_test, {
        "pairs": len(pairs),
        "files_with_neighbor": int(np.count_nonzero(degree)),
        "groups_with_neighbors": int(np.sum(sizes > 1)),
        "largest_group": int(sizes.max()),
        "random_test_size": test_size,
        "random_expected_exposed_fraction": expected_exposure(degree, test_size),
        "random_empirical_mean": float(np.mean(exposure)),
        "random_empirical_p05_p95": np.quantile(exposure, [0.05, 0.95]).tolist(),
        "grouped_test_size": grouped_size,
        "grouped_exposed_rows": grouped_exposed,
    }


def component_diameter(profiles, components):
    """Return diameter and witnesses for the largest connected component."""
    largest = np.bincount(components).argmax()
    members = np.flatnonzero(np.asarray(components) == largest)
    diameter, first, second = 0.0, int(members[0]), int(members[-1])
    for side in range(2):
        surfaces = [profiles[i][side] for i in members]
        knots = np.unique(np.concatenate([s[:, 0] for s in surfaces]))
        values = np.array([sample(surface, knots) for surface in surfaces])
        spread = np.ptp(values, axis=0)
        column = int(spread.argmax())
        if spread[column] > diameter:
            diameter = float(spread[column])
            first = int(members[values[:, column].argmin()])
            second = int(members[values[:, column].argmax()])
    return diameter, first, second


def read_archive(path):
    names, profiles, rejected = [], [], []
    with ZipFile(path) as archive:
        members = sorted(info.filename for info in archive.infolist()
                         if not info.is_dir() and info.filename.lower().endswith(".dat"))
        if len(set(members)) != len(members):
            raise ValueError("duplicate .dat member paths")
        for name in members:
            try:
                geometry = profile(archive.read(name))
            except ValueError as error:
                rejected.append((name, str(error)))
            else:
                names.append(name)
                profiles.append(geometry)
    return names, profiles, rejected


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(header)
        writer.writerows(rows)


def self_test():
    a = b"A\n1 .01\n.5 .1\n0 0\n.5 -.1\n1 -.01\n"
    # Scaling plus a redundant vertex on the first upper segment.
    b = b"B\n3 .02\n2.5 .11\n2 .2\n1 0\n2 -.2\n3 -.02\n"
    c = a.replace(b".5 .1", b".5 .101")
    groups, distances = matches([profile(a), profile(b), profile(c)], 1e-12)
    np.testing.assert_array_equal(groups, [[0, 1]])
    assert distances[0] < 1e-14
    # A narrow kink can be missed by the grid and must fail knot verification.
    upper, lower = profile(a)
    kink_x = [.509, .51, .511]
    kink_y = sample(upper, kink_x)
    kink_y[1] += .005
    inserted = np.column_stack([kink_x, kink_y])
    kink = np.vstack([upper[:2], inserted, upper[2:]])
    np.testing.assert_allclose(sample(upper, GRID), sample(kink, GRID), atol=1e-15)
    assert not len(matches([(upper, lower), (kink, lower)], .004)[0])
    diameter, i, j = component_diameter([(upper, lower), (kink, lower)], [0, 0])
    assert abs(diameter - .005) < 1e-12 and {i, j} == {0, 1}
    # A four-vertex chain is grouped transitively. Degree-one end and
    # degree-two interior have exposure 2/3 and 1 when test size is two.
    chain = np.array([[0, 1], [1, 2], [2, 3]])
    labels, mask, report = audit_split(5, chain, 2, 10, 0)
    np.testing.assert_array_equal(labels, [0, 0, 0, 0, 4])
    assert mask.any() and report["grouped_exposed_rows"] == 0
    assert abs(expected_exposure(np.array([1, 2, 2, 1]), 2) - 5/6) < 1e-12
    assert not len(matches([profile(a), profile(c)], 1e-12)[0])
    print("Self-test passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, nargs="?", help="UIUC Selig ZIP archive")
    parser.add_argument("--out", type=Path, default=Path("audit-output"))
    parser.add_argument("--tolerances", type=float, nargs="+", default=DEFAULT_TOLERANCES)
    parser.add_argument("--test-fraction", type=float, default=.2)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.archive is None:
        parser.error("provide the official Selig ZIP archive")
    tolerances = sorted(set(args.tolerances))
    if any(not np.isfinite(t) or t < 0 for t in tolerances):
        parser.error("tolerances must be finite and nonnegative")
    if not 0 < args.test_fraction < 1 or args.repeats < 1 or args.seed < 0:
        parser.error("need a test fraction in (0, 1), positive repeats and nonnegative seed")
    try:
        names, profiles, rejected = read_archive(args.archive)
    except (OSError, BadZipFile, ValueError) as error:
        parser.error(str(error))
    n = len(names)
    if n < 2:
        parser.error("need two or more eligible airfoils")
    test_size = max(1, min(n - 1, round(n * args.test_fraction)))
    args.out.mkdir(parents=True, exist_ok=True)
    pairs, distance = matches(profiles, tolerances[-1])
    write_csv(args.out / "pairs.csv", ("file_a", "file_b", "distance_chord"),
              ((names[i], names[j], d) for (i, j), d in zip(pairs, distance)))
    write_csv(args.out / "excluded.csv", ("file", "reason"), rejected)
    summary = {
        "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        "dat_members": n + len(rejected),
        "eligible": n,
        "excluded": len(rejected),
        "seed": args.seed,
        "repeats": args.repeats,
        "thresholds": [],
    }
    for tolerance in tolerances:
        labels, test, report = audit_split(
            n, pairs[distance <= tolerance], test_size, args.repeats, args.seed
        )
        assignments = ["unavailable"] * n if test is None else np.where(test, "test", "train")
        write_csv(args.out / f"groups_{tolerance:.17g}.csv", ("file", "group_id", "subset"),
                  zip(names, (names[i] for i in labels), assignments))
        diameter, i, j = component_diameter(profiles, labels)
        summary["thresholds"].append({
            "distance_chord": tolerance, **report,
            "largest_group_diameter_chord": diameter,
            "diameter_witness_files": [names[i], names[j]],
        })
    report_json = json.dumps(summary, indent=2) + "\n"
    (args.out / "summary.json").write_text(report_json, encoding="utf-8")
    print(report_json, end="")


if __name__ == "__main__":
    main()
