"""Verify the Hugging Face mirror of DDR (ctmedtech/DDR-dataset) against the authors' own archive (17 Oct 2026).
The authors' Google Drive archive is split into DDR-dataset.zip.001-.010; only the last part could be downloaded (quota).
Its central directory lists the name, size and CRC-32 of every file of the archive, so every mirrored grading image can be
checked byte-for-byte without the other nine parts.
Usage: python verify_ddr.py <Paper_05 dir>   -> data/processed/ddr_mirror_check.csv and a summary line"""
import csv
import os
import struct
import sys
import zlib

R = sys.argv[1]
part = os.path.join(R, "data", "raw", "ddr", "DDR-dataset.zip.010")
mirror = os.path.join(R, "data", "raw", "ddr_mirror", "DR_grading")
blob = open(part, "rb").read()

# end of central directory (classic, then zip64 locator/record if present)
e = blob.rfind(b"PK\x05\x06")
assert e >= 0, "no end-of-central-directory record in the last part"
disk, cd_disk, n_here, n_total, cd_size, cd_off = struct.unpack("<HHHHII", blob[e + 4:e + 20])
loc = blob.rfind(b"PK\x06\x07", 0, e)
if loc >= 0 and (cd_off == 0xFFFFFFFF or n_total == 0xFFFF):
    z = blob.rfind(b"PK\x06\x06", 0, loc)
    _, _, _, disk, cd_disk, n_here, n_total, cd_size, cd_off = struct.unpack("<QHHIIQQQQ", blob[z + 4:z + 56])
print("archive disks %d, central directory on disk %d at offset %d, %d entries (%d bytes)" % (disk + 1, cd_disk + 1, cd_off, n_total, cd_size))
# the parts are plain byte chunks of one archive (disk count 1), so offsets are global; the central directory ends where
# the zip64 end record (or the classic end record) begins
cd_end = z if loc >= 0 and (cd_off == 0xFFFFFFFF or n_total == 0xFFFF or "z" in dir()) else e
p, entries = cd_end - cd_size, {}
assert p >= 0, "central directory starts before the last part"
for _ in range(n_total):
    assert blob[p:p + 4] == b"PK\x01\x02", "bad central-directory entry at %d" % p
    crc, csize, usize = struct.unpack("<III", blob[p + 16:p + 28])
    nlen, xlen, clen = struct.unpack("<HHH", blob[p + 28:p + 34])
    name = blob[p + 46:p + 46 + nlen].decode("utf8", "replace")
    extra = blob[p + 46 + nlen:p + 46 + nlen + xlen]
    if usize == 0xFFFFFFFF:  # zip64 extended information
        q = 0
        while q < len(extra):
            hid, hl = struct.unpack("<HH", extra[q:q + 4])
            if hid == 1:
                usize = struct.unpack("<Q", extra[q + 4:q + 12])[0]
            q += 4 + hl
    entries[name] = (crc, usize)
    p += 46 + nlen + xlen + clen
rows, n_ok, n_bad, n_missing = [], 0, 0, 0
for sub in ["train", "valid", "test"]:
    d = os.path.join(mirror, sub)
    for f in sorted(os.listdir(d)):
        key = [k for k in ("DDR-dataset/DR_grading/%s/%s" % (sub, f), "DR_grading/%s/%s" % (sub, f)) if k in entries]
        data = open(os.path.join(d, f), "rb").read()
        crc = zlib.crc32(data) & 0xFFFFFFFF
        if not key:
            n_missing += 1
            rows.append((sub, f, len(data), "%08x" % crc, "", "", "NOT_IN_ARCHIVE"))
            continue
        ecrc, esize = entries[key[0]]
        ok = crc == ecrc and len(data) == esize
        n_ok += ok
        n_bad += not ok
        rows.append((sub, f, len(data), "%08x" % crc, "%08x" % ecrc, esize, "OK" if ok else "MISMATCH"))
for f in ["train.txt", "valid.txt", "test.txt"]:  # the label files
    data = open(os.path.join(mirror, f), "rb").read()
    crc = zlib.crc32(data) & 0xFFFFFFFF
    key = [k for k in entries if k.endswith("DR_grading/" + f)]
    st = "NOT_IN_ARCHIVE" if not key else ("OK" if entries[key[0]] == (crc, len(data)) else "MISMATCH")
    rows.append(("labels", f, len(data), "%08x" % crc, "%08x" % entries[key[0]][0] if key else "", entries[key[0]][1] if key else "", st))
    print("label file", f, st)
out = os.path.join(R, "data", "processed", "ddr_mirror_check.csv")
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["split", "file", "size", "crc32_mirror", "crc32_archive", "size_archive", "status"])
    w.writerows(rows)
grading = [k for k in entries if "/DR_grading/" in "/" + k and not k.endswith("/") and k.lower().endswith((".jpg", ".png", ".jpeg"))]
print("mirror images %d: OK %d, MISMATCH %d, NOT_IN_ARCHIVE %d; grading images listed in the archive %d" % (
    len(rows), n_ok, n_bad, n_missing, len(grading)))
