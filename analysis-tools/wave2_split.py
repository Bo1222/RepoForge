#!/usr/bin/env python3
"""Draw a disjoint Wave2 dev30 + holdout50 from unused Verified instances.

The used T7 split (dev30 + holdout100) is read from the frozen profile and is
never eligible.  The output is a pre-registration artifact: IDs, seed, hashes
and the method are written before any environment preparation or model call.
"""

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path


def digest(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def draw(rows, n, seed):
    rng = random.Random(seed)
    total = len(rows)
    by = Counter(row["repo"] for row in rows)
    alloc = {repo: max(1, round(n * count / total)) for repo, count in by.items()}
    while sum(alloc.values()) > n:
        repo = max((r for r in alloc if alloc[r] > 1),
                   key=lambda r: alloc[r] - n * by[r] / total)
        alloc[repo] -= 1
    while sum(alloc.values()) < n:
        repo = max(alloc, key=lambda r: n * by[r] / total - alloc[r])
        if alloc[repo] < by[repo]:
            alloc[repo] += 1
        else:
            break
    selected = []
    for repo, count in sorted(alloc.items()):
        pool = sorted(row["instance_id"] for row in rows if row["repo"] == repo)
        selected.extend(rng.sample(pool, min(count, len(pool))))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    candidate = args.candidate.resolve()
    frozen = json.loads((candidate.parents[2] / "repoforge/evaluation/profiles/frozen_split.json").read_text())
    used = set(frozen["split"]["dev"]) | set(frozen["split"]["holdout"])
    from datasets import load_dataset
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test",
                      cache_dir=str(candidate / "host-dataset-cache"))
    rows = [{"instance_id": row["instance_id"], "repo": row["repo"]}
            for row in ds if row["instance_id"] not in used]
    holdout = draw(rows, 50, args.seed)
    remaining = [row for row in rows if row["instance_id"] not in set(holdout)]
    dev = draw(remaining, 30, args.seed + 1)
    assert not (set(dev) & set(holdout))
    assert not (set(dev) | set(holdout)) & used
    result = {
        "dataset": "princeton-nlp/SWE-bench_Verified",
        "source_frozen_split_sha256": hashlib.sha256(
            (candidate.parents[2] / "repoforge/evaluation/profiles/frozen_split.json").read_bytes()).hexdigest(),
        "seed": args.seed,
        "method": "repository-stratified largest remainder, without replacement; disjoint dev30/holdout50",
        "used_t7_count": len(used),
        "eligible_count": len(rows),
        "dev": dev,
        "holdout": holdout,
        "dev_sha256": digest(dev),
        "holdout_sha256": digest(holdout),
        "dev_repo_counts": dict(sorted(Counter(row["repo"] for row in rows
                                                 if row["instance_id"] in set(dev)).items())),
        "holdout_repo_counts": dict(sorted(Counter(row["repo"] for row in rows
                                                    if row["instance_id"] in set(holdout)).items())),
        "overlap_with_t7": 0,
        "official_harness": "not_run",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(json.dumps({k: result[k] for k in ("dev_sha256", "holdout_sha256", "used_t7_count",
                                             "eligible_count", "dev_repo_counts",
                                             "holdout_repo_counts")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
