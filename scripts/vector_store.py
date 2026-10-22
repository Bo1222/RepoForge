"""List or verify the local shared code indexes without provider calls."""

import argparse
import json
from pathlib import Path

from moatless.index.repository_store import RepositoryIndexStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "verify"))
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "experiments/vector_store")
    args = parser.parse_args()
    store = RepositoryIndexStore(args.root)
    if args.command == "verify":
        rows = store.refresh_catalog()
    else:
        path = store.root / "catalog.json"
        if not path.exists():
            raise SystemExit("No catalog; create or import an index first")
        rows = json.loads(path.read_text())["indexes"]
    print(json.dumps({"root": str(store.root), "indexes": rows, "provider_calls": 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
