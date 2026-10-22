"""Verified, single-file HTTP range recovery for proxy-limited large downloads."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time

import requests


def download(url, target, size, sha256, workers=16):
    target = Path(target)
    partial = target.with_suffix(target.suffix + ".ranged")
    receipt = partial.with_suffix(partial.suffix + ".json")
    chunk_size = 8 * 1024 * 1024
    completed = set(json.loads(receipt.read_text()) if receipt.exists() else [])
    fd = os.open(partial, os.O_RDWR | os.O_CREAT, 0o600)
    os.ftruncate(fd, size)

    def fetch(i):
        start = i * chunk_size
        end = min(size, start + chunk_size) - 1
        expected = f"bytes {start}-{end}/{size}"
        for attempt in range(8):
            try:
                # Distinct query avoids proxies caching a different Range under
                # the same URL. Never print signed redirects or auth headers.
                separator = "&" if "?" in url else "?"
                with requests.get(
                    url + separator + f"range_start={start}",
                    headers={"Range": f"bytes={start}-{end}"},
                    timeout=(20, 120),
                ) as response:
                    if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                        raise RuntimeError(f"Range {i}: HTTP {response.status_code}; mismatched Content-Range")
                    content = response.content
                    if len(content) != end - start + 1:
                        raise RuntimeError(f"Range {i}: incomplete response")
                    position = start
                    while content:
                        written = os.pwrite(fd, content, position)
                        content = content[written:]
                        position += written
                    return i
            except Exception as error:
                print(json.dumps({"range": i, "attempt": attempt, "error_type": type(error).__name__}), flush=True)
                if attempt == 7:
                    raise RuntimeError(f"Range {i} exhausted retries") from None
                time.sleep(min(10, 1 + attempt))

    count = (size + chunk_size - 1) // chunk_size
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch, i) for i in range(count) if i not in completed]
            for future in as_completed(futures):
                i = future.result()
                completed.add(i)
                temporary = receipt.with_suffix(".tmp")
                temporary.write_text(json.dumps(sorted(completed)))
                temporary.replace(receipt)
                print(json.dumps({"completed": len(completed), "total": count}), flush=True)
        os.fsync(fd)
    finally:
        os.close(fd)
    digest = hashlib.sha256()
    with partial.open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(data)
    if digest.hexdigest() != sha256:
        raise RuntimeError("Downloaded file SHA256 does not match official metadata")
    partial.replace(target)
    receipt.unlink()
    print(json.dumps({"file": target.name, "bytes": size, "sha256": sha256}), flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("target")
    parser.add_argument("size", type=int)
    parser.add_argument("sha256")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    download(args.url, args.target, args.size, args.sha256, args.workers)
