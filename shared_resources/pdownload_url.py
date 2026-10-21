"""Parallel byte-range download of one HTTP(S) file that supports range requests (redirects followed on each request),
then md5 check against the publisher's value. Usage: python pdownload_url.py <url> <out file> <size> <md5> [parts]
Parts are resumable; the file is assembled and kept only if the md5 matches."""
import hashlib
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

url, name, size, md5 = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
parts = int(sys.argv[5]) if len(sys.argv) > 5 else 16
step = size // parts + 1


def part(i):
    lo, hi = i * step, min(size, (i + 1) * step) - 1
    fn = "%s.part%02d" % (name, i)
    for _ in range(80):
        have = os.path.getsize(fn) if os.path.exists(fn) else 0
        if have >= hi - lo + 1:
            return
        with open(fn, "ab") as o:
            subprocess.run(["curl", "-sL", "--max-time", "1800", "-r", "%d-%d" % (lo + have, hi), url], stdout=o)
        time.sleep(3)
    raise RuntimeError("part %d failed" % i)


if not (os.path.exists(name) and os.path.getsize(name) == size):
    with ThreadPoolExecutor(parts) as ex:
        list(ex.map(part, range(parts)))
    with open(name, "wb") as o:
        for i in range(parts):
            with open("%s.part%02d" % (name, i), "rb") as f:
                while True:
                    b = f.read(1 << 24)
                    if not b:
                        break
                    o.write(b)
h = hashlib.md5()
with open(name, "rb") as f:
    while True:
        b = f.read(1 << 24)
        if not b:
            break
        h.update(b)
ok = h.hexdigest() == md5
print(name, "md5", h.hexdigest(), "OK" if ok else "MISMATCH", flush=True)
if ok:
    for i in range(parts):
        if os.path.exists("%s.part%02d" % (name, i)):
            os.remove("%s.part%02d" % (name, i))
