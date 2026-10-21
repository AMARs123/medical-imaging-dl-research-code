"""Parallel byte-range download of one Dataverse file (S3 behind a 303 redirect), then md5 check.

Usage: python pdownload.py <datafile id> <name> <size> <md5> [parts]
A single connection to the Dataverse S3 bucket ran at about 144 KB/s from the server (2026-10-15), so the file
is fetched as `parts` byte ranges in parallel. Each part is resumable; the presigned S3 link is requested again
for every attempt because it expires. The file is kept only if its md5 matches the Dataverse record.
"""
import hashlib
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

fid, name, size, md5 = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
parts = int(sys.argv[5]) if len(sys.argv) > 5 else 16
api = "https://dataverse.harvard.edu/api/access/datafile/%s?format=original" % fid


def s3url():
    out = subprocess.run(["curl", "-sI", api], capture_output=True, text=True).stdout
    return [l.split(" ", 1)[1].strip() for l in out.splitlines() if l.lower().startswith("location:")][0]


step = size // parts + 1


def part(i):
    lo, hi = i * step, min(size, (i + 1) * step) - 1
    fn = "%s.part%02d" % (name, i)
    for _ in range(50):
        have = os.path.getsize(fn) if os.path.exists(fn) else 0
        if have >= hi - lo + 1:
            return
        with open(fn, "ab") as o:
            subprocess.run(["curl", "-s", "--max-time", "900", "-r", "%d-%d" % (lo + have, hi), s3url()], stdout=o)
        time.sleep(2)
    raise RuntimeError("part %d failed" % i)


with ThreadPoolExecutor(parts) as ex:
    list(ex.map(part, range(parts)))
h = hashlib.md5()
with open(name, "wb") as o:
    for i in range(parts):
        b = open("%s.part%02d" % (name, i), "rb").read()
        o.write(b)
        h.update(b)
h = h.hexdigest()
print(name, "md5", h, "OK" if h == md5 else "MISMATCH", flush=True)
if h == md5:
    for i in range(parts):
        os.remove("%s.part%02d" % (name, i))
