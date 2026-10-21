"""Download the vision model llama-server runs (Qwen3-VL, GGUF) into vlm/ next to this script.

    python download_vlm.py                Qwen3-VL-8B Q4_K_M + its mmproj, 6.2 GB (what the pipeline was tuned with)
    python download_vlm.py --model 2b     Qwen3-VL-2B Q8_0 + its mmproj, 2.7 GB (for GPUs with < 8 GB; not measured)

Standard library only. An interrupted download continues where it stopped when run again; every
file is checked against its SHA-256 before it is used."""
import argparse
import hashlib
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

# (repo, revision, file, size, sha256): the exact files the pipeline was run with
MODELS = {
    "8b": [
        ("Qwen/Qwen3-VL-8B-Instruct-GGUF", "f982a07559d4a2f6c8744d840bf6fccab30eea96",
         "Qwen3VL-8B-Instruct-Q4_K_M.gguf", 5027784800,
         "67d1659bfe71b89d50b45a4ad1a9e5b997e5bb16ce5da66a6a6167abd569e9e2"),
        ("Qwen/Qwen3-VL-8B-Instruct-GGUF", "f982a07559d4a2f6c8744d840bf6fccab30eea96",
         "mmproj-Qwen3VL-8B-Instruct-F16.gguf", 1159029824,
         "ca524100ebf825c9a870db1c580d03879e0da0ab2541697e2458e64891cf9d38"),
    ],
    "2b": [
        ("Qwen/Qwen3-VL-2B-Instruct-GGUF", "52d6c8ffea26cc873ac5ad116f8631268d7eb503",
         "Qwen3VL-2B-Instruct-Q8_0.gguf", 1834427424,
         "1e8db19207c8ce0733ddd78c2eff8a9e22c27c82f4443df94c25792ed8fe04f2"),
        ("Qwen/Qwen3-VL-2B-Instruct-GGUF", "52d6c8ffea26cc873ac5ad116f8631268d7eb503",
         "mmproj-Qwen3VL-2B-Instruct-F16.gguf", 819394848,
         "c3d5afbef5287953acd57b4043d2269456e5761a4eaccb3b71b062996970aea5"),
    ],
}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: str, size: int) -> None:
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    if have > size:
        os.remove(part)
        have = 0
    if have < size:
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        with urllib.request.urlopen(req, timeout=60) as r:
            if have and r.status != 206:              # the server ignored the range: start over
                have = 0
            with open(part, "ab" if have else "wb") as f:
                t0, last, start = time.time(), 0.0, have
                while True:
                    block = r.read(1 << 20)
                    if not block:
                        break
                    f.write(block)
                    have += len(block)
                    now = time.time()
                    if now - last >= 2 or have == size:
                        speed = (have - start) / max(1e-6, now - t0)
                        left = (size - have) / max(1.0, speed)
                        print(f"\r  {have / 1e9:5.2f} / {size / 1e9:.2f} GB  {speed / 1e6:6.1f} MB/s  "
                              f"{int(left // 60)}m {int(left % 60):02d}s left ", end="", file=sys.stderr)
                        last = now
        print(file=sys.stderr)
    if os.path.getsize(part) != size:
        sys.exit(f"{os.path.basename(dest)}: download incomplete ({os.path.getsize(part)} of {size} bytes); run again")
    os.replace(part, dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODELS), default="8b", help="8b (default) or 2b")
    ap.add_argument("--dir", default=os.path.join(HERE, "vlm"), help="where to put the files (default: vlm/)")
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)
    for repo, rev, name, size, digest in MODELS[a.model]:
        dest = os.path.join(a.dir, name)
        if os.path.exists(dest) and os.path.getsize(dest) == size:
            print(f"{name}: already there", file=sys.stderr)
        else:
            print(f"{name} ({size / 1e9:.2f} GB) from huggingface.co/{repo}", file=sys.stderr)
            fetch(f"https://huggingface.co/{repo}/resolve/{rev}/{name}", dest, size)
        print(f"  checking SHA-256 ...", end=" ", file=sys.stderr, flush=True)
        if sha256(dest) != digest:
            os.remove(dest)
            sys.exit(f"\n{name}: checksum mismatch, file removed; run again")
        print("ok", file=sys.stderr)
    print(f"done: {a.dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
