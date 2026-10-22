#!/usr/bin/env python3
"""Validate the Wave2 split before any environment or model work."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
profile = json.loads((ROOT / "moatless/evaluation/profiles/wave2_split.json").read_text())
frozen = json.loads((ROOT / "moatless/evaluation/profiles/frozen_split.json").read_text())
used = set(frozen["split"]["dev"]) | set(frozen["split"]["holdout"])
dev, holdout = profile["split"]["dev"], profile["split"]["holdout"]
assert len(dev) == 30 and len(holdout) == 50
assert not (set(dev) & set(holdout))
assert not (set(dev) | set(holdout)) & used
assert hashlib.sha256(json.dumps(dev, sort_keys=True).encode()).hexdigest() == profile["split"]["dev_sha256"]
assert hashlib.sha256(json.dumps(holdout, sort_keys=True).encode()).hexdigest() == profile["split"]["holdout_sha256"]
print(json.dumps({"dev": len(dev), "holdout": len(holdout), "overlap": 0,
                  "dev_sha256": profile["split"]["dev_sha256"],
                  "holdout_sha256": profile["split"]["holdout_sha256"],
                  "official_harness": "not_run", "provider_calls": 0}, indent=1))
