"""Extract selected members of a remote ZIP file over HTTP range requests, without downloading the whole archive.

Usage:
  python remote_zip.py list    <url> <out.csv>                       # central directory -> csv (name, size, offset, crc)
  python remote_zip.py extract <url> <dest dir> <names.txt|regex> [threads]
The central directory is read from the end of the file (ZIP64 supported). Each selected member is fetched as one
byte range (local header + data), decompressed, and checked against the CRC-32 stored in the archive; a member whose
CRC does not match is fetched again and never written. Several members are fetched in parallel.
"""
import csv
import os
import re
import struct
import sys
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor

UA = {"User-Agent": "research-download (remote_zip.py)"}


def fetch(url, lo, hi, tries=8):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers=dict(UA, Range="bytes=%d-%d" % (lo, hi)))
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) == hi - lo + 1:
                return data
        except Exception:
            pass
        time.sleep(min(60, 3 * (t + 1)))
    raise IOError("range %d-%d failed" % (lo, hi))


def size_of(url):
    req = urllib.request.Request(url, headers=dict(UA, Range="bytes=0-0"))
    with urllib.request.urlopen(req, timeout=60) as r:
        cr = r.headers.get("Content-Range")
        return int(cr.split("/")[-1])


def central_directory(url):
    n = size_of(url)
    tail = fetch(url, max(0, n - 65536 - 22), n - 1)
    e = tail.rfind(b"PK\x05\x06")
    cd_size, cd_off = struct.unpack("<II", tail[e + 12:e + 20])
    if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:  # ZIP64
        l = tail.rfind(b"PK\x06\x07")
        z64_off = struct.unpack("<Q", tail[l + 8:l + 16])[0]
        z = fetch(url, z64_off, z64_off + 55)
        cd_size, cd_off = struct.unpack("<QQ", z[40:56])
    cd = fetch(url, cd_off, cd_off + cd_size - 1)
    out, p = [], 0
    while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        (method, crc, csize, usize, nlen, xlen, clen, off) = (struct.unpack("<H", cd[p + 10:p + 12])[0],) + struct.unpack("<III", cd[p + 16:p + 28]) + \
            struct.unpack("<HHH", cd[p + 28:p + 34]) + (struct.unpack("<I", cd[p + 42:p + 46])[0],)
        name = cd[p + 46:p + 46 + nlen].decode("utf8", "replace")
        extra = cd[p + 46 + nlen:p + 46 + nlen + xlen]
        q = 0
        while q + 4 <= len(extra):  # ZIP64 extended information
            hid, hl = struct.unpack("<HH", extra[q:q + 4])
            if hid == 1:
                vals, r = [], q + 4
                for flag in (usize, csize, off):
                    if flag == 0xFFFFFFFF:
                        vals.append(struct.unpack("<Q", extra[r:r + 8])[0]); r += 8
                    else:
                        vals.append(flag)
                usize, csize, off = vals
            q += 4 + hl
        out.append(dict(name=name, method=method, crc=crc, csize=csize, usize=usize, offset=off))
        p += 46 + nlen + xlen + clen
    return out


def extract_one(url, m, dest):
    fn = os.path.join(dest, m["name"])
    if os.path.exists(fn) and os.path.getsize(fn) == m["usize"]:
        if zlib.crc32(open(fn, "rb").read()) & 0xFFFFFFFF == m["crc"]:
            return "ok"
    for _ in range(3):
        head = fetch(url, m["offset"], m["offset"] + 29)
        nlen, xlen = struct.unpack("<HH", head[26:30])
        start = m["offset"] + 30 + nlen + xlen
        raw = fetch(url, start, start + m["csize"] - 1) if m["csize"] else b""
        data = raw if m["method"] == 0 else zlib.decompress(raw, -15)
        if zlib.crc32(data) & 0xFFFFFFFF == m["crc"]:
            os.makedirs(os.path.dirname(fn) or ".", exist_ok=True)
            open(fn, "wb").write(data)
            return "ok"
    return "CRC_FAILED " + m["name"]


if __name__ == "__main__":
    cmd, url = sys.argv[1], sys.argv[2]
    members = central_directory(url)
    if cmd == "list":
        with open(sys.argv[3], "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(members[0]))
            w.writeheader(); w.writerows(members)
        print("members", len(members))
    else:
        dest, sel = sys.argv[3], sys.argv[4]
        threads = int(sys.argv[5]) if len(sys.argv) > 5 else 16
        if os.path.exists(sel):
            want = set(l.strip() for l in open(sel) if l.strip())
            todo = [m for m in members if m["name"] in want or os.path.basename(m["name"]) in want]
        else:
            rx = re.compile(sel)
            todo = [m for m in members if rx.search(m["name"]) and not m["name"].endswith("/")]
        print("selected", len(todo), "of", len(members), flush=True)
        bad, n = [], 0
        with ThreadPoolExecutor(threads) as ex:
            for r in ex.map(lambda m: extract_one(url, m, dest), todo):
                n += 1
                if r != "ok":
                    bad.append(r)
                if n % 500 == 0:
                    print(n, "done", len(bad), "failed", flush=True)
        print("\n".join(bad))
        print("EXTRACT_DONE" if not bad else "EXTRACT_INCOMPLETE", n, len(bad))
