"""Read all text in a directory of Minecraft screenshots: one OUT/<path>.json per screenshot.

    python read_screenshots.py SCREENSHOTS/ OUT/
    python read_screenshots.py SCREENSHOTS/ OUT/ --combine             also OUT/results.jsonl, all in one
    python read_screenshots.py SCREENSHOTS/ OUT/ --llama-server PATH   start llama-server for the run
    python read_screenshots.py SCREENSHOTS/ OUT/ --no-vlm              CPU only, no vision model (see README)

Stopped part way (Ctrl-C, crash, reboot)? Run the same command again: screenshots that already have
their JSON are skipped. Each JSON is written in one piece when its screenshot is done."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
import urllib.parse
import urllib.request
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

VERSION = "2026-10-18"
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VLM_FILES = {  # --model -> (model, mmproj), as download_vlm.py names them
    "8b": ("Qwen3VL-8B-Instruct-Q4_K_M.gguf", "mmproj-Qwen3VL-8B-Instruct-F16.gguf"),
    "2b": ("Qwen3VL-2B-Instruct-Q8_0.gguf", "mmproj-Qwen3VL-2B-Instruct-F16.gguf"),
}


def find_images(root: str, skip: str) -> list[str]:
    """Sorted image paths under root, relative to it, leaving out the output directory."""
    out = []
    skip = os.path.abspath(skip)
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if os.path.abspath(os.path.join(d, x)) != skip)
        for f in files:
            if os.path.splitext(f)[1].lower() in IMAGE_EXT:
                out.append(os.path.relpath(os.path.join(d, f), root))
    return sorted(out)


def out_path(out_dir: str, rel: str) -> str:
    return os.path.join(out_dir, rel + ".json")          # keeps the extension: shot.png and shot.jpg stay apart


def write_json(path: str, obj) -> None:
    """Written to a temporary file and renamed, so a file that exists is always complete."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def is_done(path: str, retry_errors: bool) -> bool:
    if not os.path.exists(path):
        return False
    if not retry_errors:
        return True
    try:
        with open(path, encoding="utf-8") as f:
            return "error" not in json.load(f)
    except (OSError, ValueError):
        return False


def server_up(url: str) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=2) as r:
            return b"ok" in r.read()
    except Exception:
        return False


def served_model(url: str):
    """File name of the model a running llama-server has loaded, or None if it does not say."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=5) as r:
            return os.path.basename(json.load(r)["data"][0]["id"])
    except Exception:
        return None


def start_server(a):
    model, proj = (os.path.join(a.vlm_dir, f) for f in VLM_FILES[a.model or "8b"])
    missing = [os.path.basename(f) for f in (model, proj) if not os.path.exists(f)]
    if missing:
        sys.exit(f"{', '.join(missing)} not in {a.vlm_dir}: run  python download_vlm.py --model {a.model or '8b'}")
    port = urllib.parse.urlparse(a.vlm_url).port or 8080
    log_path = os.path.join(a.out_dir, "llama-server.log")
    os.makedirs(a.out_dir, exist_ok=True)
    log = open(log_path, "w")
    cmd = [a.llama_server, "-m", model, "--mmproj", proj, "-ngl", "99", "-c", str(2048 * a.slots),
           "-np", str(a.slots), "--host", "127.0.0.1", "--port", str(port)]
    kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    p = subprocess.Popen(cmd, stdout=log, stderr=log, stdin=subprocess.DEVNULL, **kw)
    print(f"starting llama-server with {os.path.basename(model)} (log: {log_path}) ...", file=sys.stderr)
    for _ in range(600):
        if server_up(a.vlm_url):
            print("llama-server ready", file=sys.stderr)
            return p
        if p.poll() is not None:
            sys.exit(f"llama-server exited (code {p.returncode}); see {log_path}")
        time.sleep(0.5)
    stop_server(p)
    sys.exit(f"llama-server did not come up in 5 minutes; see {log_path}")


def stop_server(p) -> None:
    if p is None or p.poll() is not None:
        return
    try:
        if os.name == "nt":
            p.terminate()
        else:
            os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=20)
    except Exception:
        try:
            if os.name == "nt":
                p.kill()
            else:
                os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            pass


_W = {}


def worker_init(cfg: dict) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)          # Ctrl-C is handled by the main process
    import mc_ocr
    if not os.environ.get("MCOCR_DET_THREADS"):
        mc_ocr.DET_THREADS = cfg["det_threads"]
    _W.update(cfg, mc_ocr=mc_ocr, glyphs=mc_ocr.load_font())


def work(rel: str):
    """('ok' | 'error' | 'fatal', rel, result). Fatal: a setup problem every screenshot would hit
    (the vision model server went away, a model file is missing), so the run stops instead."""
    mc_ocr = _W["mc_ocr"]
    t = time.time()
    try:
        r = mc_ocr.extract(os.path.join(_W["in_dir"], rel), glyphs=_W["glyphs"], fallback=_W["fallback"],
                           vlm_backend="llama", vlm_url=_W["vlm_url"])
    except SystemExit as e:
        return "fatal", rel, str(e)
    except Exception:
        return "error", rel, {"image": rel, "mcocr": VERSION, "error": traceback.format_exc()[-1500:]}
    r.pop("vlm_whole", None)
    r["image"] = rel
    return "ok", rel, {"image": rel, "mcocr": VERSION, "mode": _W["mode"], "vlm": _W["vlm"],
                       "seconds": round(time.time() - t, 2), **r}


def fmt_time(s: float) -> str:
    s = int(s)
    return f"{s // 3600}h {s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m {s % 60:02d}s"


def combine(in_dir: str, out_dir: str) -> None:
    images = find_images(in_dir, out_dir)
    path = os.path.join(out_dir, "results.jsonl")
    n = 0
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        for rel in images:
            p = out_path(out_dir, rel)
            if os.path.exists(p):
                with open(p, encoding="utf-8") as g:
                    f.write(json.dumps(json.load(g), ensure_ascii=False) + "\n")
                n += 1
    os.replace(path + ".tmp", path)
    print(f"{path}: {n} screenshots", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("in_dir", help="directory of screenshots (searched recursively)")
    ap.add_argument("out_dir", help="where the JSON files go (same sub-folders as the screenshots)")
    ap.add_argument("--jobs", type=int, default=4, help="screenshots processed in parallel (default 4; "
                    "the vision model is the bottleneck, more than 4-6 rarely helps)")
    ap.add_argument("--no-vlm", action="store_true", help="no vision model: the exact font layer and mcrec "
                    "only (CPU only; text in non-Minecraft fonts is not read)")
    ap.add_argument("--vlm-url", default="http://127.0.0.1:8080", help="llama-server address")
    ap.add_argument("--llama-server", metavar="PATH", help="start this llama-server binary for the run "
                    "(unless one already answers at --vlm-url) and stop it at the end")
    ap.add_argument("--model", choices=sorted(VLM_FILES), help="vision model --llama-server starts: 8b (default, "
                    "what the pipeline was tuned with) or 2b (smaller GPUs). With a server already running, its "
                    "model is used; asking for another one stops the run")
    ap.add_argument("--vlm-dir", default=os.path.join(HERE, "vlm"), help="where the model files are (default: vlm/)")
    ap.add_argument("--slots", type=int, default=4, help="parallel requests llama-server handles (default 4)")
    ap.add_argument("--retry-errors", action="store_true", help="redo screenshots whose JSON records an error")
    ap.add_argument("--limit", type=int, help="process at most this many screenshots this run (to try it out)")
    ap.add_argument("--combine", action="store_true", help="also write OUT/results.jsonl: every screenshot's "
                    "result, one per line (rebuilt from the per-screenshot files, so resumed runs are complete)")
    a = ap.parse_args()

    if not os.path.isdir(a.in_dir):
        sys.exit(f"not a directory: {a.in_dir}")
    images = find_images(a.in_dir, a.out_dir)
    todo = [rel for rel in images if not is_done(out_path(a.out_dir, rel), a.retry_errors)]
    print(f"{len(images)} screenshots, {len(images) - len(todo)} already done, {len(todo)} to do", file=sys.stderr)
    if a.limit is not None:
        todo = todo[:a.limit]

    server = None
    if todo and not a.no_vlm and not server_up(a.vlm_url):
        if not a.llama_server:
            sys.exit(f"no vision model server answers at {a.vlm_url}.\n"
                     f"Start one (see README: ./serve_vlm.sh PATH/TO/llama-server), or pass --llama-server PATH "
                     f"to have this script start it, or use --no-vlm.")
        server = start_server(a)
    vlm = None
    if todo and not a.no_vlm:
        vlm = served_model(a.vlm_url)
        if a.model and vlm and vlm != VLM_FILES[a.model][0]:
            stop_server(server)
            sys.exit(f"the server at {a.vlm_url} is running {vlm}, not the {a.model} model; stop it first, "
                     f"or leave out --model to use it")
        print(f"vision model: {vlm or 'unknown (the server does not say)'}", file=sys.stderr)

    cfg = {"in_dir": a.in_dir, "vlm_url": a.vlm_url, "fallback": "mcrec" if a.no_vlm else "vlm",
           "mode": "no-vlm" if a.no_vlm else "vlm", "vlm": vlm, "det_threads": 2 if a.jobs > 1 else 0}
    done = errors = 0
    t0 = time.time()
    last, first = t0, None          # rate from the first finished screenshot on (start-up excluded)
    status = 0
    pool = None
    try:
        if todo:
            pool = Pool(max(1, a.jobs), initializer=worker_init, initargs=(cfg,))
            for kind, rel, res in pool.imap_unordered(work, todo, chunksize=1):
                if kind == "fatal":
                    print(f"\nstopping: {res}\n(rerun the same command to continue where it stopped)", file=sys.stderr)
                    status = 2
                    break
                write_json(out_path(a.out_dir, rel), res)
                done += 1
                errors += kind == "error"
                if kind == "error":
                    print(f"error on {rel}: {res['error'].strip().splitlines()[-1]}", file=sys.stderr)
                now = time.time()
                first = first or now
                if now - last >= 30 or done == len(todo):
                    rate = (now - first) / (done - 1) if done > 1 else now - t0
                    print(f"[{done}/{len(todo)}] {rate:.1f} s per screenshot, {fmt_time(now - t0)} so far, "
                          f"about {fmt_time(rate * (len(todo) - done))} left" + (f", {errors} errors" if errors else ""),
                          file=sys.stderr)
                    last = now
    except KeyboardInterrupt:
        print(f"\nstopped after {done} screenshots; rerun the same command to continue", file=sys.stderr)
        status = 130
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        stop_server(server)
    if status == 0:
        print(f"done: {done} screenshots in {fmt_time(time.time() - t0)}" +
              (f", {errors} with errors (their JSON says why; --retry-errors redoes them)" if errors else ""),
              file=sys.stderr)
    if a.combine and status == 0:
        combine(a.in_dir, a.out_dir)
    sys.exit(status)


if __name__ == "__main__":
    main()
