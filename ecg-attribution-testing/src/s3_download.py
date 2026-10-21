"""Download files of a PhysioNet project from its open AWS mirror (s3://physionet-open), checking SHA-256.

Usage: python s3_download.py <project/version> <dest dir> <regex of paths to keep> [threads]
The project's SHA256SUMS.txt is the manifest; every downloaded file is hashed and re-fetched on mismatch.
PhysioNet's own server delivered about 5 KB/s to the compute server on 2026-10-15, hence the mirror.
"""
import hashlib
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

proj, dest, keep = sys.argv[1], sys.argv[2], re.compile(sys.argv[3])
threads = int(sys.argv[4]) if len(sys.argv) > 4 else 32
base = "https://physionet-open.s3.amazonaws.com/%s/" % proj.strip("/")
os.makedirs(dest, exist_ok=True)
man = urllib.request.urlopen(base + "SHA256SUMS.txt", timeout=60).read().decode()
open(os.path.join(dest, "SHA256SUMS.txt"), "w").write(man)
todo = [(h, p) for h, p in (l.split(None, 1) for l in man.splitlines() if l.strip()) if keep.search(p)]
print("files to fetch:", len(todo), flush=True)


def sha(fn):
    return hashlib.sha256(open(fn, "rb").read()).hexdigest()


def get(item):
    h, p = item
    fn = os.path.join(dest, p)
    if os.path.exists(fn) and sha(fn) == h:
        return "ok"
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    for _ in range(6):
        try:
            data = urllib.request.urlopen(base + p, timeout=120).read()
            if hashlib.sha256(data).hexdigest() == h:
                open(fn, "wb").write(data)
                return "ok"
        except Exception:
            pass
    return "FAILED " + p


bad, n = [], 0
with ThreadPoolExecutor(threads) as ex:
    for r in ex.map(get, todo):
        n += 1
        if r != "ok":
            bad.append(r)
        if n % 2000 == 0:
            print(n, "done", len(bad), "failed", flush=True)
print("\n".join(bad))
print("finished", n, "files,", len(bad), "failed")
print("DOWNLOAD_DONE" if not bad else "DOWNLOAD_INCOMPLETE")
