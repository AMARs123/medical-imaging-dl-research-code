"""Check that the Hugging Face mirror of NIH ChestX-ray14 (alkzar90/NIH-Chest-X-ray-dataset) holds the same image
bytes as the official NIH Clinical Center download, before any image is taken from the mirror.

Official part 1 (images_001.tar.gz, NIH Box link listed in NIH's batch_download_zips.py) is streamed once; the CRC-32 of
every PNG is computed and compared with the CRC-32 recorded in the central directory of the mirror's images_001.zip.
Usage: python verify_nih_mirror.py <out.csv>
"""
import os
import sys
import tarfile
import urllib.request
import zlib

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared_resources"))
from remote_zip import central_directory  # noqa: E402

BOX = "https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz"
HF = "https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data/images/images_001.zip"
hf = {os.path.basename(m["name"]): m["crc"] for m in central_directory(HF) if m["name"].endswith(".png")}
print("mirror part 1 PNGs:", len(hf), flush=True)
rows = []
with urllib.request.urlopen(urllib.request.Request(BOX, headers={"User-Agent": "research"}), timeout=300) as r:
    with tarfile.open(fileobj=r, mode="r|gz") as t:
        for m in t:
            if m.isfile() and m.name.endswith(".png"):
                crc = zlib.crc32(t.extractfile(m).read()) & 0xFFFFFFFF
                b = os.path.basename(m.name)
                rows.append(dict(name=b, crc_official=crc, crc_mirror=hf.get(b), match=hf.get(b) == crc))
                if len(rows) % 1000 == 0:
                    print(len(rows), flush=True)
df = pd.DataFrame(rows)
df.to_csv(sys.argv[1], index=False)
print("official PNGs:", len(df), "| in mirror:", int(df.crc_mirror.notna().sum()), "| identical:", int(df.match.sum()),
      "| mirror-only:", len(set(hf) - set(df.name)))
print("MIRROR_VERIFIED" if df.match.all() and len(df) == len(hf) else "MIRROR_DIFFERS")
