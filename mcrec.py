#!/usr/bin/env python3
"""
mcrec.py - a small text-line recogniser (CRNN + CTC) for Minecraft's font: what mc_ocr.py's exact
layer cannot match pixel for pixel (resized screenshots, faded chat, text over icons, signs). It
also flags whether a crop is Minecraft-font text at all; other fonts are left to the vision model.

    python mcrec.py train RUN.jsonl DET.jsonl [STEPS]   # exact-layer run + its detector boxes -> models/
    python mcrec.py read IMAGE x y w h                  # read one box

Inference needs numpy, Pillow and onnxruntime; training needs PyTorch and mcrec_data.py.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "models")
H = 32
MAX_W = 1024
STRIDE = 2                  # input columns per output frame (each model's mcrec.json has its own); at 4
                            # a run of narrow glyphs ('.....') cannot be written out: CTC needs a blank between repeats


# ------------------------------------------------------------------ preprocessing (shared)

def to_input(img: Image.Image) -> np.ndarray:
    """RGB crop -> (3, H, W) float32 in [0, 1], height H, aspect kept, width capped."""
    w = max(8, min(MAX_W, round(img.width * H / max(1, img.height))))
    a = np.asarray(img.convert("RGB").resize((w, H), Image.BILINEAR), np.float32) / 255.0
    return a.transpose(2, 0, 1)


def batch(arrs: list[np.ndarray], round_to: int = 1, stride: int = STRIDE) -> tuple[np.ndarray, np.ndarray]:
    """Pad to one width: always at least 16 columns of black after the widest crop, since the
    network reads the last character from the columns that follow it (in training every batch is
    padded)."""
    W = -(-(max(a.shape[2] for a in arrs) + 16) // round_to) * round_to
    out = np.zeros((len(arrs), 3, H, W), np.float32)
    for i, a in enumerate(arrs):
        out[i, :, :, :a.shape[2]] = a
    return out, np.array([a.shape[2] // stride for a in arrs], np.int64)


def ctc_decode(probs: np.ndarray, length: int, charset: str) -> tuple[str, float]:
    """Greedy CTC decode of (T, C) probabilities -> text and confidence: the lowest, over the
    characters, of each character's best frame (a character spans several frames and the first
    one under-rates it). 1.0 for an empty reading: the blank won everywhere."""
    p = probs[:length]
    best = p.argmax(axis=1)
    out, confs, prev = [], [], 0
    for t, k in enumerate(best):
        if k != 0 and k != prev:
            out.append(charset[k - 1])
            confs.append(float(p[t, k]))
        elif k != 0:
            confs[-1] = max(confs[-1], float(p[t, k]))
        prev = k
    if not out:
        return "", float(p[:, 0].min()) if len(p) else 1.0
    return "".join(out), min(confs)


# ------------------------------------------------------------------ inference

_SESSION: dict = {}          # model dir -> (onnx session, charset, stride)
SECOND = "mcrec2"            # a second model in this subfolder is consulted where the main one is unsure
SECOND_GAP = 0.1             # ... and its reading taken when it is this much more confident


def load(model_dir: str = MODEL_DIR):
    if model_dir not in _SESSION:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        sess = ort.InferenceSession(os.path.join(model_dir, "mcrec.onnx"), opts, providers=["CPUExecutionProvider"])
        with open(os.path.join(model_dir, "mcrec.json"), encoding="utf-8") as f:
            meta = json.load(f)
        _SESSION[model_dir] = (sess, meta["charset"], meta.get("stride", 4))
    return _SESSION[model_dir][:2]


def stride_of(model_dir: str = MODEL_DIR) -> int:
    load(model_dir)
    return _SESSION[model_dir][2]


def available(model_dir: str = MODEL_DIR) -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return os.path.exists(os.path.join(model_dir, "mcrec.onnx"))


def infer_one(crops: list[Image.Image], model_dir: str = MODEL_DIR) -> list[tuple[np.ndarray, float]]:
    """Per crop, one model's (T, C) frame probabilities (class 0 = CTC blank, class k = charset[k-1])
    and the probability that the crop is Minecraft-font text."""
    if not crops:
        return []
    sess, _ = load(model_dir)
    stride = stride_of(model_dir)
    order = sorted(range(len(crops)), key=lambda i: crops[i].width / max(1, crops[i].height))
    res: dict[int, tuple[np.ndarray, float]] = {}
    for s in range(0, len(order), 16):
        idx = order[s:s + 16]
        x, lens = batch([to_input(crops[i]) for i in idx], stride=stride)
        outs = sess.run(None, {"x": x})                # (N, T, C) softmax, (N,) Minecraft font
        mc = outs[1] if len(outs) > 1 else np.ones(len(idx), np.float32)
        for j, i in enumerate(idx):
            res[i] = (outs[0][j, :int(lens[j])], float(mc[j]))
    return [res[i] for i in range(len(crops))]


def infer(crops: list[Image.Image], model_dir: str = MODEL_DIR) -> list[tuple[np.ndarray, float]]:
    """infer_one with the main model; where a second model is installed (model_dir/mcrec2), its
    output replaces the main one's for a crop when it reads something else at least SECOND_GAP more
    confidently (the second reads italic item names and runs of dots; the main one is surer of
    messy everyday text)."""
    first = infer_one(crops, model_dir)
    second_dir = os.path.join(model_dir, SECOND)
    if not crops or not available(second_dir):
        return first
    _, charset = load(model_dir)
    _, charset2 = load(second_dir)
    if charset2 != charset:
        return first
    out = []
    for (p1, f1), (p2, f2) in zip(first, infer_one(crops, second_dir)):
        t1, c1 = ctc_decode(p1, len(p1), charset)
        t2, c2 = ctc_decode(p2, len(p2), charset)
        t1, t2 = " ".join(t1.split()), " ".join(t2.split())
        # not when the second reading only leaves out characters of the first: the second model
        # confidently drops an edge character ('_Steve_' -> 'Steve_', '>!' -> '>')
        use2 = c2 >= c1 + SECOND_GAP and t2 != t1 and not _subsequence(t2, t1)
        out.append((p2, f2) if use2 else (p1, f1))
    return out


def _subsequence(a: str, b: str) -> bool:
    """Is a what is left of b after deleting some of its characters?"""
    it = iter(b)
    return all(ch in it for ch in a)


def read(crops: list[Image.Image], model_dir: str = MODEL_DIR) -> list[tuple[str, float]]:
    """Read text-line crops -> [(text, confidence)].  Text in other fonts reads as nothing (the
    recogniser reads it too, but the vision model does better there)."""
    _, charset = load(model_dir)
    return [ctc_decode(p, len(p), charset) if mc >= 0.5 else ("", 1.0 - mc) for p, mc in infer(crops, model_dir)]


def ctc_logprob(probs: np.ndarray, text: str, charset: str) -> float:
    """log P(text | crop) under the recogniser (CTC forward algorithm): how well a reading from
    elsewhere fits the pixels.  -inf for characters the recogniser does not know."""
    idx = {c: i + 1 for i, c in enumerate(charset)}
    if any(c not in idx for c in text):
        return float("-inf")
    lab = [0]
    for c in text:
        lab += [idx[c], 0]
    lab = np.array(lab)
    S, T = len(lab), len(probs)
    lp = np.log(np.maximum(probs, 1e-30))[:, lab]                     # (T, S)
    skip = np.zeros(S, bool)                                          # s-2 -> s allowed: a character
    skip[2:] = (lab[2:] != 0) & (lab[2:] != lab[:-2])                 # differing from the one before
    a = np.full(S, -np.inf)
    a[0] = lp[0, 0]
    if S > 1:
        a[1] = lp[0, 1]
    prev1, prev2 = np.full(S, -np.inf), np.full(S, -np.inf)
    for t in range(1, T):
        prev1[1:] = a[:-1]
        prev2[2:] = np.where(skip[2:], a[:-2], -np.inf)
        a = np.logaddexp(np.logaddexp(a, prev1), prev2) + lp[t]
    return float(np.logaddexp(a[-1], a[-2]) if S > 1 else a[-1])


# ------------------------------------------------------------------ training

def gather(runs: list[str], dets: list[str]):
    """(image, exact lines, detector boxes) for every screenshot that has both."""
    boxes = {}
    for d in dets:
        for r in map(json.loads, open(d)):
            if "error" not in r:
                boxes[r["image"]] = r["boxes"]
    out = []
    for run in runs:
        for r in map(json.loads, open(run)):
            if "error" not in r and r["image"] in boxes:
                out.append((r["image"], r["lines"], boxes[r["image"]]))
    return out


def prepare(runs: list[str], dets: list[str], seed: int = 0, n_val_images: int = 70):
    """Cut the real crops and scenery patches once.  Validation screenshots are never trained on."""
    import mc_ocr
    import mcrec_data as D
    from multiprocessing import Pool
    rng = random.Random(seed)
    per_image = gather(runs, dets)
    rng.shuffle(per_image)
    va_imgs, tr_imgs = per_image[:n_val_images], per_image[n_val_images:]
    jobs = lambda imgs, n: [(im, lines, boxes, n, seed + i) for i, (im, lines, boxes) in enumerate(imgs)]
    with Pool(16) as pool:
        tr = pool.map(D.real_items, jobs(tr_imgs, 3), chunksize=4)
        va = pool.map(D.real_items, jobs(va_imgs, 3), chunksize=4)
        bgs = [p for ps in pool.map(D.background_patches, jobs(tr_imgs[:1200], 2), chunksize=4) for p in ps]
    cs = set(D.charset())
    real_tr = [it for items, _ in tr for it in items if set(it["label"]) <= cs]
    real_va = [it for items, _ in va for it in items if set(it["label"]) <= cs]
    geoms = [g for _, gs in tr for g in gs]
    pool_texts = [l["text"] for _, lines, _ in tr_imgs for l in lines
                  if l["source"] == "mcfont" and l["confidence"] >= 1.0 and "�" not in l["text"]
                  and mc_ocr.trusted(l["text"]) and set(l["text"]) <= cs]
    return real_tr, real_va, geoms, bgs, pool_texts


def validation_sets(mix, real_va, n_synth: int = 1500, seed: int = 12345):
    """Fixed crops to measure on: held-out real detector boxes as they are, the same degraded, and
    synthetic text (Minecraft font, then TrueType, which must be flagged as another font)."""
    rng = random.Random(seed)
    clean, degraded = [], []
    for it in real_va:
        if not it["label"]:
            continue
        d = it["det"]
        img = Image.fromarray(it["img"]).crop(tuple(int(round(v)) for v in d))
        clean.append((img, it["label"], 1))
        degraded.append(mix.real_view(rng, it))
    synth = [mix.synth_view(rng) for _ in range(n_synth)]
    signs = [mix.synth_view(rng, style="sign") for _ in range(n_synth // 3)]
    italic = [mix.synth_view(rng, italic=True) for _ in range(n_synth // 3)]
    ttf = [mix.ttf_view(rng) for _ in range(n_synth // 3)]
    return {"real": clean, "real degraded": degraded, "synth": synth, "signs": signs, "italic": italic, "ttf": ttf}


def train(runs: list[str], dets: list[str], steps: int = 20000, out_dir: str = MODEL_DIR, seed: int = 0,
          bs: int = 64, workers: int = 12, max_vram_gb: float = 6.0, max_columns: int = 64 * 272):
    """max_vram_gb caps PyTorch's share of the GPU: the card also drives the desktop and other
    programs, and running out of video memory can hang it."""
    # ROCm: MIOpen otherwise searches kernels exhaustively for every new input shape
    os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, IterableDataset
    import mcrec_data as D

    t0 = time.time()
    real_tr, real_va, geoms, bgs, pool_texts = prepare(runs, dets, seed)
    mix = D.Mixture(real_tr, geoms, bgs, pool_texts)
    charset = D.charset()
    cidx = {c: i + 1 for i, c in enumerate(charset)}
    vsets = validation_sets(mix, real_va)
    print(f"{len(mix.real)} real line crops, {len(mix.neg)} scenery crops, {len(geoms)} detector geometries, "
          f"{len(bgs)} background patches, {len(pool_texts)} texts; validation "
          + ", ".join(f"{k} {len(v)}" for k, v in vsets.items()) + f" ({time.time() - t0:.0f}s)", file=sys.stderr)

    def collate(data):
        x, lens = batch([to_input(img) for img, _, _ in data], round_to=128)   # few shapes for the GPU kernels
        tg = [torch.tensor([cidx[c] for c in t if c in cidx], dtype=torch.long) for _, t, _ in data]
        mc = torch.tensor([float(m) for _, _, m in data])
        return torch.from_numpy(x), torch.from_numpy(lens), tg, [t for _, t, _ in data], mc

    class Stream(IterableDataset):
        """Endless batches; each worker draws 4 at a time and groups them by width (padding short lines
        to the widest slows learning). A batch holds at most `max_columns` input columns, so batches of
        long lines are smaller: GPU memory grows with the columns."""

        def __iter__(self):
            info = torch.utils.data.get_worker_info()
            rng = random.Random(seed * 1000 + (info.id if info else 0) + time.time_ns())
            while True:
                views = [mix.draw(rng) for _ in range(bs * 4)]
                views.sort(key=lambda v: v[0].width / max(1, v[0].height))
                cols = lambda v: min(MAX_W, round(v[0].width * H / max(1, v[0].height))) + 16
                groups, cur = [], []
                for v in views:
                    if cur and (len(cur) >= bs or (len(cur) + 1) * max(cols(v), max(map(cols, cur))) > max_columns):
                        groups.append(cur)
                        cur = []
                    cur.append(v)
                if cur:
                    groups.append(cur)
                rng.shuffle(groups)
                for g in groups:
                    yield collate(g)

    loader = DataLoader(Stream(), batch_size=None, num_workers=workers, prefetch_factor=4, persistent_workers=True)
    model = CRNN(len(charset) + 1, STRIDE)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda" and max_vram_gb:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, max_vram_gb * 2 ** 30 / total))
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1.5e-3, total_steps=steps, pct_start=0.06)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    bce = nn.BCEWithLogitsLoss()

    def forward(x, lens, tg, mc):
        logits, font = model(x.to(dev))
        logp = logits.log_softmax(2).permute(1, 0, 2)                     # (T, N, C)
        il = torch.clamp(lens, max=logp.shape[0])
        tl = torch.tensor([len(t) for t in tg], dtype=torch.long)
        tgt = torch.cat(tg) if int(tl.sum()) else torch.zeros(0, dtype=torch.long)
        return ctc(logp, tgt, il, tl) + 0.5 * bce(font, mc.to(dev)), logp, torch.sigmoid(font)

    def evaluate():
        """Per set: exact readings; how often the font flag is right; for Minecraft text, how many
        crops the pipeline would take from the recogniser (flagged, confidence >= 0.9) and how many
        of those are right."""
        model.eval()
        res = []
        with torch.no_grad():
            for name, items in vsets.items():
                ok = flag_ok = taken = taken_ok = 0
                order = sorted(range(len(items)), key=lambda i: items[i][0].width / max(1, items[i][0].height))
                for s in range(0, len(order), bs):
                    chunk = [items[i] for i in order[s:s + bs]]
                    x, lens, tg, texts, mc = collate(chunk)
                    _, logp, pmc = forward(x, lens, tg, mc)
                    probs = logp.permute(1, 0, 2).exp().cpu().numpy()
                    for j, t in enumerate(texts):
                        out, conf = ctc_decode(probs[j], int(lens[j]), charset)
                        ok += out == t
                        flag_ok += (float(pmc[j]) >= 0.5) == bool(mc[j])
                        if out and conf >= 0.9 and float(pmc[j]) >= 0.5:
                            taken += 1
                            taken_ok += out == t
                res.append(f"{name} {ok / len(items):.1%} read, font flag {flag_ok / len(items):.1%}, taken "
                           f"{taken / len(items):.0%} ({taken_ok / max(1, taken):.1%} right)")
        model.train()
        return "; ".join(res)

    model.train()
    tot, n = 0.0, 0
    for step, (x, lens, tg, _, mc) in enumerate(loader, 1):
        loss, _, _ = forward(x, lens, tg, mc)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        tot, n = tot + loss.item(), n + 1
        if step % 250 == 0:
            vram = f"  vram peak {torch.cuda.max_memory_reserved() / 2 ** 30:.1f} GB" if dev.type == "cuda" else ""
            print(f"step {step}/{steps}  loss {tot / n:.3f}  {time.time() - t0:.0f}s{vram}", file=sys.stderr, flush=True)
            tot, n = 0.0, 0
        if step % 2500 == 0 or step == steps:
            print(f"  val @ {step}: {evaluate()}", file=sys.stderr, flush=True)
            export(model, out_dir, {"charset": charset, "height": H, "stride": STRIDE, "trained_on": runs,
                                    "detector_boxes": dets, "steps": step, "real_lines": len(mix.real)})
        if step == steps:
            break
    print(f"wrote {out_dir}/mcrec.onnx", file=sys.stderr)


def export(model, out_dir: str, meta: dict):
    """Save the weights and an ONNX copy for onnxruntime (written at every validation, so a
    checkpoint can be tried while training goes on)."""
    import copy
    import torch
    import torch.nn as nn

    class Export(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            logits, font = self.m(x)
            return logits.softmax(2), torch.sigmoid(font)

    os.makedirs(out_dir, exist_ok=True)
    m = copy.deepcopy(model).eval().cpu()
    torch.save(m.state_dict(), os.path.join(out_dir, "mcrec.pt"))
    tmp = os.path.join(out_dir, "mcrec.onnx.tmp")
    torch.onnx.export(Export(m), torch.zeros(1, 3, H, 256), tmp, input_names=["x"], output_names=["p", "mc"],
                      dynamic_axes={"x": {0: "n", 3: "w"}, "p": {0: "n", 1: "t"}, "mc": {0: "n"}},
                      opset_version=17, dynamo=False)
    os.replace(tmp, os.path.join(out_dir, "mcrec.onnx"))
    json.dump(meta, open(os.path.join(out_dir, "mcrec.json"), "w"), ensure_ascii=False)


def CRNN(n_classes: int, stride: int = STRIDE):
    """Convolutional feature extractor, 1-D convolutions over the frames (ROCm's LSTM kernels fault
    on these shapes), CTC classes per frame; plus a logit for "this is Minecraft-font text", from
    the features pooled over the line."""
    import torch
    import torch.nn as nn

    def block(i, o, pool):
        layers = [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]
        if pool:
            layers.append(nn.MaxPool2d(pool))
        return layers

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(
                *block(3, 64, (2, 2)), *block(64, 128, (2, 2) if stride == 4 else (2, 1)),   # 8 x W/stride
                *block(128, 256, None), *block(256, 256, (2, 1)),        # 4 x W/4
                *block(256, 384, None), *block(384, 384, (2, 1)),        # 2 x W/4
                nn.Conv2d(384, 384, (2, 1), bias=False), nn.BatchNorm2d(384), nn.ReLU(inplace=True))
            k = 5 if stride == 4 else 9          # the same context in input columns at either stride
            self.seq = nn.Sequential(
                nn.Conv1d(384, 384, k, padding=k // 2, bias=False), nn.BatchNorm1d(384), nn.ReLU(inplace=True),
                nn.Conv1d(384, 384, k, padding=k // 2, bias=False), nn.BatchNorm1d(384), nn.ReLU(inplace=True),
                nn.Conv1d(384, n_classes, 1))
            self.font = nn.Sequential(nn.Conv1d(384, 64, 1), nn.ReLU(inplace=True))
            self.font_fc = nn.Linear(128, 1)

        def forward(self, x):
            f = self.cnn(x).squeeze(2)                                    # (N, 384, T)
            g = self.font(f)
            font = self.font_fc(torch.cat([g.amax(2), g.mean(2)], 1)).squeeze(1)
            return self.seq(f).permute(0, 2, 1), font                     # (N, T, C) logits, (N,) logit

    return Net()


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "train":
        return train([sys.argv[2]], [sys.argv[3]], *(int(v) for v in sys.argv[4:5]))
    if len(sys.argv) == 7 and sys.argv[1] == "read":
        pil = Image.open(sys.argv[2]).convert("RGB")
        x, y, w, h = map(float, sys.argv[3:])
        print(read([pil.crop((int(x), int(y), int(x + w), int(y + h)))])[0])
        return
    print(__doc__)


if __name__ == "__main__":
    main()
