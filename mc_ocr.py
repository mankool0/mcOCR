#!/usr/bin/env python3
"""
mc_ocr.py - all the text in a Minecraft screenshot.

Layer 1 reads Minecraft's own bitmap font exactly. GUI text is drawn from fixed atlases at an
integer GUI scale, with a shadow one scaled pixel down-right whose colour is text_colour >> 2:
text is found by that shadow rule and read by template matching, with its colour and position.

Layer 2 (--fallback) reads what layer 1 did not explain: boxes from a text detector, read by the
Minecraft-font recogniser (mcrec) and a vision model.

    python mc_ocr.py screenshot.png [--json] [--fallback vlm --vlm-backend llama]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import math
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

# ------------------------------------------------------------------ constants

GLYPH_H = 8          # ascii glyph rows: 7 above the baseline + 1 descender row
WIN_TOP = 3          # rows above ascii row 0 that accented capitals use (ascent 10 vs 7)
WIN_H = 12           # decode window: ascii rows -3 .. 8
SPACE_ADVANCE = 4    # from font/include/space.json
DEFAULT_SCALES = tuple(range(1, 9))          # GUI scale 8 is the auto maximum on a 4K screen
WEIGHTS = (1 << np.arange(WIN_H)).astype(np.int64)
HIGH_MASK = (1 << 9) - 1                    # window rows 0..8; rows 9..11 (below the baseline) may
                                            # carry the next chat line's accents, so extra ink there is free
LOW_MASK = ((1 << WIN_H) - 1) & ~HIGH_MASK
DESC_MASK = (1 << 10) | (1 << 11)           # descender rows: glyph ink there may be missing (old F3 boxes
                                            # clip it) and is tolerated at a penalty
TOP_ROW = 1 << WIN_TOP                      # ascii row 0 (the tops of capitals, t, the i dot): the shadow of
                                            # an underline in the line above lands there and hides it
MAY_MISS = DESC_MASK | TOP_ROW
POP = np.array([bin(i).count("1") for i in range(1 << WIN_H)], np.int64)
QWEN_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
MAX_VLM_PIXELS = {
    "hf": 1_600_000,         # full attention gets very slow above ~1.5 Mpx
    "llama": 6_000_000,      # no such slowdown seen
}
VLM_PROMPT = ("Transcribe all text in this image exactly as written, one output line per line of "
              "text, top to bottom. Reply with only the text.")
# one detector box at a time (the fallback); NONE for icons and scenery
BOX_PROMPT = ("This is a crop of a Minecraft screenshot that may contain HUD or chat text. Transcribe the text "
              "exactly, character for character: keep capitalisation, digits, punctuation, brackets and "
              "symbols, and do not correct spelling. Ignore text cut off at the top or bottom edge. If there "
              "is no readable text (only icons, items or scenery), reply NONE. Reply with only the text.")

MC_COLORS = {
    "black": (0, 0, 0), "dark_blue": (0, 0, 170), "dark_green": (0, 170, 0),
    "dark_aqua": (0, 170, 170), "dark_red": (170, 0, 0), "dark_purple": (170, 0, 170),
    "gold": (255, 170, 0), "gray": (170, 170, 170), "dark_gray": (85, 85, 85),
    "blue": (85, 85, 255), "green": (85, 255, 85), "aqua": (85, 255, 255),
    "red": (255, 85, 85), "light_purple": (255, 85, 255), "yellow": (255, 255, 85),
    "white": (255, 255, 255), "xp_green": (128, 255, 32),
}

# ascii_legacy.png is ascii.png from 1.8-1.12 clients: 14 punctuation glyphs differ, same layout
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "font")
FONT_FILES = ["default.json", "ascii.png", "ascii_legacy.png", "accented.png", "nonlatin_european.png"]
# pixel-identical glyphs in several atlases (Latin a / Cyrillic а): the earlier atlas wins
PROVIDER_PRIORITY = ["ascii.png", "ascii_legacy.png", "accented.png", "nonlatin_european.png"]
HOMOGLYPHS: dict[str, str] = {}   # char -> char it is pixel-identical to (filled by load_font)
DEBUG = bool(os.environ.get("MCOCR_DEBUG"))   # trace the line finder on stderr


# ----------------------------------------------------------------------- font

@dataclass
class Glyph:
    char: str
    bitmap: np.ndarray      # (WIN_H, width) bool, rows aligned to the common window
    width: int              # ink width in glyph pixels
    advance: int
    bold: bool
    sig: tuple              # per-column bit signatures
    ink: int
    atlas: str
    low_info: bool = False   # a dot, a bar or a solid block: what wrong-scale pixel blocks decode to
    siglist: list = field(default_factory=list)


def font_files(font_dir: Optional[str] = None) -> dict[str, bytes]:
    d = font_dir or FONT_DIR
    missing = [n for n in FONT_FILES if not os.path.exists(os.path.join(d, n))]
    if missing:
        sys.exit(f"font files missing from {d}: {', '.join(missing)}")
    out = {}
    for n in FONT_FILES:
        with open(os.path.join(d, n), "rb") as f:
            out[n] = f.read()
    return out


def load_font(font_dir: Optional[str] = None, bold: bool = True) -> list[Glyph]:
    files = font_files(font_dir)
    providers = [(p["file"].split("/")[-1], int(p.get("ascent", 7)), int(p.get("height", 8)), p["chars"])
                 for p in json.loads(files["default.json"].decode("utf-8"))["providers"] if p.get("type") == "bitmap"]
    providers.append(("ascii_legacy.png", 7, 8, next(p[3] for p in providers if p[0] == "ascii.png")))
    providers.sort(key=lambda t: PROVIDER_PRIORITY.index(t[0]) if t[0] in PROVIDER_PRIORITY else 99)

    glyphs: list[Glyph] = []
    seen: dict[bytes, str] = {}
    HOMOGLYPHS.clear()

    def add(ch, bm, is_bold, atlas):
        w = bm.shape[1]
        sig = tuple(int(v) for v in (bm * WEIGHTS[:, None]).sum(axis=0))
        ink = int(bm.sum())
        rr, cc = np.flatnonzero(bm.any(axis=1)), np.flatnonzero(bm.any(axis=0))
        solid = ink == (rr[-1] - rr[0] + 1) * (cc[-1] - cc[0] + 1)
        low = ink <= 3 or len(rr) <= 1 or len(cc) <= 1 or solid
        glyphs.append(Glyph(ch, bm, w, w + 1, is_bold, sig, ink, atlas, low))

    for name, ascent, height, rows in providers:
        atlas = np.asarray(Image.open(io.BytesIO(files[name])).convert("RGBA"))[..., 3] > 0
        ncols = max(len(r) for r in rows)
        cw, ch_ = atlas.shape[1] // ncols, atlas.shape[0] // len(rows)
        if height != ch_:
            print(f"warning: {name} is rendered at {height}px from {ch_}px cells (HD font); "
                  f"not pixel-exact, skipped", file=sys.stderr)
            continue
        off = GLYPH_H - 1 + WIN_TOP - ascent       # provider row 0 -> window row
        if off < 0 or off + ch_ > WIN_H:
            print(f"warning: {name} (ascent {ascent}, {ch_} rows) does not fit the window, skipped",
                  file=sys.stderr)
            continue
        for ri, row in enumerate(rows):
            for ci, ch in enumerate(row):
                if ch in ("\x00", " ", "\ufffd") or ci >= ncols:   # U+FFFD is our unknown marker
                    continue
                cell = atlas[ri * ch_:(ri + 1) * ch_, ci * cw:(ci + 1) * cw]
                cols = np.flatnonzero(cell.any(axis=0))
                if len(cols) == 0:
                    continue
                w = int(cols.max()) + 1
                bm = np.zeros((WIN_H, w), bool)
                bm[off:off + ch_] = cell[:, :w]
                key = bytes([w]) + bm.tobytes()
                if key in seen:
                    HOMOGLYPHS[ch] = seen[key]
                    continue
                seen[key] = ch
                add(ch, bm, False, name)
                if bold:  # bold = glyph drawn again one pixel to the right
                    bb = np.zeros((WIN_H, w + 1), bool)
                    bb[:, :w] |= bm
                    bb[:, 1:] |= bm
                    add(ch, bb, True, name)
    # widest first, normal before bold; stable, so atlas priority breaks remaining ties
    glyphs.sort(key=lambda g: (-g.width, g.bold))
    return glyphs


def glyph_index(glyphs: list[Glyph]) -> tuple[dict, dict]:
    """exact: (first, second column signature) -> glyphs, widest first.
    near: first column with rows 9..11 masked off, with and without one extra pixel -> per glyph
    width, a (n, width+1) array of target signatures (gap column appended), the glyphs and how many
    extra pixels each may carry in rows 0..8 (none for small glyphs, one or two for large ones)."""
    exact: dict[tuple, list[Glyph]] = {}
    tmp: dict[int, dict[int, list[Glyph]]] = {}
    for g in glyphs:
        g.siglist = list(g.sig)
        s0 = g.sig[0]
        exact.setdefault((s0, g.sig[1] if g.width > 1 else 0), []).append(g)
        m0 = s0 & HIGH_MASK
        for k in {m0, m0 & ~TOP_ROW} | {m0 | (1 << b) for b in range(9) if not m0 & (1 << b)}:
            tmp.setdefault(k, {}).setdefault(g.width, []).append(g)
    near: dict[int, list] = {}
    for k, byw in tmp.items():
        near[k] = [(w, np.array([gl.siglist + [0] for gl in gs], np.int64), gs,
                    np.array([0 if gl.ink < 7 else 1 if gl.ink < 10 else 2 for gl in gs], np.int64))
                   for w, gs in sorted(byw.items(), reverse=True)]
    return exact, near, ascii_alternatives(glyphs)


def ascii_alternatives(glyphs: list[Glyph]) -> dict:
    """What prefer_ascii needs: the ASCII glyphs as one padded bitmap stack, and the foreign glyphs
    that are pixel-identical to a run of ASCII glyphs one column apart ('‼' is '!!')."""
    asc = [g for g in glyphs if g.atlas.startswith("ascii")]
    wmax = max(g.width for g in asc)
    bm = np.zeros((len(asc), WIN_H, wmax), bool)
    for i, g in enumerate(asc):
        bm[i, :, :g.width] = g.bitmap
    rows = bm.any(axis=2)
    seqs: dict[bytes, list[Glyph]] = {}
    plain = [g for g in asc if not g.bold]
    narrow = [g for g in plain if g.width <= 2]
    for a in plain:
        for b in plain:
            seqs.setdefault(_concat([a, b]), [a, b])
        if a.width <= 2:
            for b in narrow:
                for c in narrow:
                    seqs.setdefault(_concat([a, b, c]), [a, b, c])
    seq = {id(g): seqs[_concat([g])] for g in glyphs
           if not g.atlas.startswith("ascii") and not g.bold and _concat([g]) in seqs}
    return {"glyphs": asc, "bm": bm, "w": np.array([g.width for g in asc]), "top": rows.argmax(axis=1),
            "ink": np.array([g.ink for g in asc]), "bold": np.array([g.bold for g in asc]),
            "letter": np.array([g.char.isalnum() for g in asc]), "seq": seq}


def _concat(gs: list[Glyph]) -> bytes:
    """Bitmap key of glyphs drawn one empty column apart."""
    parts = []
    for i, g in enumerate(gs):
        if i:
            parts.append(np.zeros((WIN_H, 1), bool))
        parts.append(g.bitmap)
    bm = np.concatenate(parts, axis=1)
    return bytes([bm.shape[1]]) + bm.tobytes()


# ------------------------------------------------------------------ foreground masks

def shadow_pre(img: np.ndarray, tol: int = 2, min_bright: int = 70, tol_bright: int = 6):
    """Scale-independent part of shadow_mask. Dim pixels hit the 1:4 ratio by chance (edges of
    translucent GUI boxes) and get a strict tolerance; bright text a looser one (1.8-1.12 draw the
    F3 shadow at low alpha)."""
    bmax = img.max(axis=2)
    return img >> 2, bmax >= min_bright, np.where(bmax >= 128, tol_bright, tol).astype(np.int16)[..., None]


def shadow_mask(img: np.ndarray, s: int, tol: int = 2, min_bright: int = 70, tol_bright: int = 6,
                pre=None) -> np.ndarray:
    """Pixels that are text drawn with Minecraft's drop shadow at GUI scale s: at p+(s,s) each has
    its shadow (colour p>>2) or another pixel of its colour that ends in one further down. Shadows
    (max channel 63) stay below min_bright; a looser tol lets dim GUI-box edges pass as text."""
    H, W, _ = img.shape
    F = np.zeros((H, W), bool)
    if H <= s or W <= s:
        return F
    if pre is None:
        pre = shadow_pre(img, tol, min_bright, tol_bright)
    quarter, bright_all, tolv_all = pre
    base = img[:H - s, :W - s]
    diag = img[s:, s:]
    bright = bright_all[:H - s, :W - s]
    is_shadow = (np.abs(diag - quarter[:H - s, :W - s]) <= tolv_all[:H - s, :W - s]).all(axis=2)
    same = (np.abs(diag - base) <= 3).all(axis=2) & bright
    F[:H - s, :W - s] = is_shadow & bright
    for _ in range(WIN_H):
        new = same & F[s:, s:] & ~F[:H - s, :W - s]
        if not new.any():
            break
        F[:H - s, :W - s] |= new
    sh = np.zeros_like(F)                   # a pixel that is the shadow of text is never text
    sh[s:, s:] = F[:H - s, :W - s] & is_shadow
    F &= ~sh
    return F


def outline_pre(img: np.ndarray, color=(128, 255, 32), tol: int = 24):
    c = np.array(color, np.int16)
    is_c = (np.abs(img - c) <= tol).all(axis=2)
    return is_c, is_c | (img.max(axis=2) <= 24)


def outline_mask(img: np.ndarray, s: int, color=(128, 255, 32), tol: int = 24, pre=None) -> np.ndarray:
    """Text drawn with a black outline (the XP level): colour pixels whose 4-neighbours at
    distance s are either the same colour or black."""
    is_c, allowed = pre if pre is not None else outline_pre(img, color, tol)
    ok = is_c.copy()
    if not ok.any():
        return ok
    for dy, dx in ((s, 0), (-s, 0), (0, s), (0, -s)):
        ok &= np.roll(allowed, (dy, dx), axis=(0, 1))
    return ok


def color_mask(img: np.ndarray, color, tol: int = 12) -> np.ndarray:
    """Plain colour match, for text drawn without a shadow (signs, some mod HUDs)."""
    c = np.array(color, np.int16)
    return (np.abs(img - c) <= tol).all(axis=2)


GUI_LABEL = (64, 64, 64)   # 0x404040: container titles and GUI labels, drawn without a shadow


def auto_text_colors(img: np.ndarray, max_colors: int = 10, min_count: int = 120) -> list[tuple]:
    """Bright exact RGB values that are frequent. Text is drawn in flat colours, so text with an
    unusable shadow (old F3) still shows up as one; textures and sky rarely repeat one exact bright
    value this often. The dark grey of container titles is added when present."""
    bright = img.max(axis=2) >= 100
    px = img[bright].astype(np.int64)
    out = []
    if len(px):
        keys = (px[:, 0] << 16) | (px[:, 1] << 8) | px[:, 2]
        vals, counts = np.unique(keys, return_counts=True)
        for i in np.argsort(-counts)[:max_colors * 3]:
            if counts[i] < min_count or len(out) >= max_colors:
                break
            if counts[i] > 0.12 * img.shape[0] * img.shape[1]:
                continue                               # over an eighth of the screen: scenery, not text
            v = int(vals[i])
            c = ((v >> 16) & 255, (v >> 8) & 255, v & 255)
            if strokes(color_mask(img, c, 0)):
                out.append(c)
    C = color_mask(img, GUI_LABEL, 0)
    if C.sum() >= 40 and strokes(C):
        out.append(GUI_LABEL)
    return out


def likely_scales(C: np.ndarray, scales) -> set:
    """GUI scales a flat-colour mask can hold text at.  At scale s every horizontal run of a glyph's
    ink is a multiple of s pixels; at scale 1 one-pixel runs are common.  Skipping the other scales
    keeps one mask per colour affordable."""
    d = np.diff(np.pad(C, ((0, 0), (1, 1))).astype(np.int8), axis=1)
    runs = np.flatnonzero(d.ravel() == -1) - np.flatnonzero(d.ravel() == 1)
    if len(runs) < 20:
        return set()
    out = set()
    for s in scales:
        if s == 1:
            if (runs == 1).mean() >= 0.2:
                out.add(1)
        else:
            short = runs[runs <= 6 * s]                # long runs are bars and panels
            if len(short) and (short % s == 0).mean() >= 0.6 and (runs == s).sum() >= 10:
                out.add(s)
    return out


def strokes(C: np.ndarray) -> bool:
    """Mostly thin strokes (text), not a GUI panel or a flat surface: under 30% survives a 5x5 erosion."""
    E = C.copy()
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dy or dx:
                E &= np.roll(C, (dy, dx), axis=(0, 1))
    return E.sum() <= 0.3 * C.sum()


# ------------------------------------------------------------------------ decoding

@dataclass
class Char:
    char: str
    x: int                 # glyph-resolution column within the line
    width: int
    bold: bool
    color: Optional[tuple] = None
    glyph: Optional["Glyph"] = None   # the template that matched (a char can have several variants)
    topless: bool = False             # matched without its top row (see decode_row)


@dataclass
class Line:
    text: str
    x: int
    y: int                 # top of the ascii glyph rows (accents may sit up to 3*scale above)
    width: int
    height: int
    scale: int
    chars: list = field(default_factory=list)
    matched: int = 0       # ink columns explained by glyphs
    unmatched: int = 0     # ink columns that matched nothing
    kind: str = "text"
    source: str = "mcfont"
    mask: str = "shadow"   # which foreground mask the line was read from

    @property
    def score(self) -> float:
        tot = self.matched + self.unmatched
        return self.matched / tot if tot else 0.0

    def colors(self) -> list[dict]:
        """Runs of the same colour/boldness -> [{text, rgb, name, bold}]"""
        runs: list[dict] = []
        for ch in self.chars:
            if ch.char == " " and runs:
                runs[-1]["text"] += " "
            elif runs and runs[-1]["rgb"] == ch.color and runs[-1]["bold"] == ch.bold:
                runs[-1]["text"] += ch.char
            else:
                runs.append({"text": ch.char, "rgb": ch.color, "name": nearest_color(ch.color), "bold": ch.bold})
        if runs:
            runs[-1]["text"] = runs[-1]["text"].rstrip()
        return runs


def nearest_color(rgb) -> Optional[str]:
    if rgb is None:
        return None
    v = np.array(rgb, float)
    best, bd = None, 1e9
    for name, c in MC_COLORS.items():
        d = np.abs(v - np.array(c)).max()
        if d < bd:
            best, bd = name, d
    # fading chat text is alpha-blended towards the background, so allow a generous radius
    return best if bd <= 70 else "#%02x%02x%02x" % tuple(int(x) for x in rgb)


def strip_underline(L: np.ndarray, min_run: int = 12) -> np.ndarray:
    """Clickable chat text (URLs, screenshot links) is underlined: a solid row of ink spanning
    several glyphs just below the descender row.  Remove runs that long from rows 10..11."""
    out = L
    for r in (10, 11):
        row = L[r]
        if row.sum() < min_run:
            continue
        d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        for a, b in zip(starts, ends):
            if b - a >= min_run:
                if out is L:
                    out = L.copy()
                out[r, a:b] = False
    return out


def decode_row(L: np.ndarray, index: tuple) -> tuple[list[Char], int, int]:
    """L: (WIN_H, n) bool glyph-resolution ink. Greedy longest exact match, left to right, against
    the raw ink, then with isolated pixels removed, then near-exactly: descender rows may be
    missing, ink below the baseline (the next line's accents) is free, large glyphs may carry a stray
    pixel or two, and a whole top row may be hidden (an underline's shadow from the line above, a
    clipped panel). Near-exact attempts are rationed so junk rows at the wrong scale stay cheap."""
    exact, near = index[0], index[1]
    n = L.shape[1]
    L = strip_underline(L)
    nb = np.zeros_like(L)
    nb[1:] |= L[:-1]
    nb[:-1] |= L[1:]
    nb[:, 1:] |= L[:, :-1]
    nb[:, :-1] |= L[:, 1:]
    sigs = np.zeros(n + 16, np.int64)
    sigs[:n] = (L * WEIGHTS[:, None]).sum(axis=0)
    sigs2 = np.zeros(n + 16, np.int64)
    sigs2[:n] = ((L & nb) * WEIGHTS[:, None]).sum(axis=0)
    sl, sl2 = sigs.tolist(), sigs2.tolist()
    out: list[Char] = []
    matched = unmatched = 0
    gap = 0
    near_budget = 1            # near-exact attempts allowed; each exact hit earns one more
    x = 0
    while x < n and sl[x] == 0:
        x += 1
    while x < n:
        if sl[x] == 0:
            gap += 1
            x += 1
            continue
        hit, near_hit, topless = None, False, False
        for arr in (sl, sl2):
            for g in exact.get((arr[x], arr[x + 1]), ()):
                w = g.width
                if arr[x + w] == 0 and arr[x:x + w] == g.siglist:
                    hit = g                   # list is widest-first, normal before bold
                    break
            if hit is not None:
                break
        if hit is not None and not hit.atlas.startswith("ascii"):
            # an exact foreign hit that is an ASCII glyph plus ink below the baseline (t -> t with dot
            # below) is usually the ASCII glyph with the next line's accent under it: prefer ASCII
            W = sigs[x:x + 16]
            for w, T, gs, allow in near.get(sl[x] & HIGH_MASK, ()):
                Wn = W[:w + 1]
                ok = ((Wn & T) == T).all(axis=1) & (POP[(Wn & ~T) & HIGH_MASK].sum(axis=1) == 0)
                if ok.any():
                    for i in np.flatnonzero(ok):
                        if gs[i].atlas.startswith("ascii") and gs[i].width >= hit.width - 1:
                            hit, near_hit = gs[i], True
                            break
                    if near_hit:
                        break
        if hit is None and sl2[x] != 0 and near_budget > 0:
            best = None
            for w, T, gs, allow in near.get(sl[x] & HIGH_MASK, ()):
                W = sigs[x:x + w + 1]
                need = T & ~MAY_MISS
                present = ((W & need) == need).all(axis=1)
                if not present.any():
                    continue
                extra = W & ~T
                hi = POP[extra & HIGH_MASK].sum(axis=1)
                lo = POP[extra & LOW_MASK].sum(axis=1)
                miss = POP[T & ~W & DESC_MASK].sum(axis=1)
                miss_top = POP[T & ~W & TOP_ROW].sum(axis=1)      # a hidden top row goes missing as a whole
                ok = present & (hi <= allow) & (lo <= 6) & (miss <= 2)
                if ok.any():
                    score = hi + lo + 2 * miss + miss_top
                    i = int(np.argmin(np.where(ok, score, 99)))
                    cand = (int(score[i]), -w, gs[i].bold)
                    if best is None or cand < best[0]:
                        best = (cand, gs[i], int(miss_top[i]) > 0)
            if best is not None:
                hit, near_hit, topless = best[1], True, best[2]
            else:
                near_budget -= 1
        if hit is None:
            if sl2[x] == 0:                   # only isolated pixels here: noise, not a glyph
                x += 1
                continue
            x0 = x
            x += 1
            # skip this blob of ink, but stop at a column with ink only below the baseline so an
            # unreadable glyph does not swallow its neighbour through the next line's accents
            while x < n and sl2[x] != 0 and (sl2[x] & HIGH_MASK) != 0:
                x += 1
            out.append(Char("\ufffd", x0, x - x0, False))
            unmatched += x - x0
            gap = 0
            continue
        if out and gap >= 2:
            for i in range(max(1, int(round(gap / SPACE_ADVANCE)))):
                out.append(Char(" ", x - gap + i * SPACE_ADVANCE, SPACE_ADVANCE, False))
        gap = 0
        if not near_hit:
            near_budget += 1
        out.append(Char(hit.char, x, hit.width, hit.bold, glyph=hit, topless=topless))
        matched += hit.width
        x += hit.advance
    return out, matched, unmatched


REAL_SCRIPTS = ("CYRILLIC", "GREEK", "HEBREW", "ARMENIAN", "GEORGIAN")
# accented letters of the European languages players actually type; never "corrected" to ASCII
EUROPEAN = set("ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÑÒÓÔÕÖØÙÚÛÜÝßàáâãäåæçèéêëìíîïñòóôõöøùúûüýÿŒœŸÐðÞþ"   # Western
               "ĄąĆćĘęŁłŃńŚśŹźŻżČčĎďĚěŇňŘřŠšŤťŮůŽžŐőŰűĞğŞşİȘșȚțĂă")              # Central, Turkish, Romanian
# punctuation-like glyphs that icons, slot frames and stray pixels decode to at the ends of a line
EDGE_JUNK = set(".,'`") | {"¨", "´", "¯", "˘", "˜", "·", "˙", "‚", "„", "′", "″", "‶", "‷", "¸",
                           "¡", "¿", "◘", "₋", "⁻", "•", "­", "·"}


def script(ch: str) -> str:
    try:
        return unicodedata.name(ch).split()[0]
    except ValueError:
        return ""


def prefer_ascii(chars: list[Char], L: np.ndarray, L0: np.ndarray, alt: dict) -> list[Char]:
    """Undo look-alike substitutions. A Latin letter that lost or gained a pixel (an overlapping
    shadow, an icon edge) often matches an accented or nonlatin glyph exactly (i -> dotless i,
    l -> iota, t -> t with dot below). Each such glyph becomes the ASCII glyph that fits within a
    pixel or two (missing ink judged against L0, the ink before anything was peeled away; extra ink
    against L). Kept as read: European accented letters, glyphs with an accent above the ASCII
    letter, and letters of another alphabet in a word that also has one with no ASCII look-alike.
    Foreign glyphs identical to an ASCII sequence become that sequence ('‼' -> '!!')."""
    if all(c.glyph is None or c.glyph.atlas.startswith("ascii") for c in chars):
        return chars
    pad = ((0, 0), (0, 4))                 # candidates may reach a column or two past the last glyph
    L, L0 = np.pad(L, pad), np.pad(L0, pad)
    alts = [ascii_lookalike(chars, k, L, L0, alt) if chars[k].glyph is not None
            and not chars[k].glyph.atlas.startswith("ascii") and chars[k].char not in EUROPEAN else None
            for k in range(len(chars))]
    alts = [a if a is None or len(a) > 1 or chars[k].char.isalnum() else None for k, a in enumerate(alts)]
    anchor = [c.glyph is not None and c.char.isalpha() and script(c.char) in REAL_SCRIPTS and alts[k] is None
              for k, c in enumerate(chars)]
    # runs of 2-3 adjacent foreign glyphs that no single replacement explains may be one ASCII glyph
    # broken up ('7' with its top row hidden reads as three runes)
    merged: dict[int, tuple[int, Char]] = {}
    k = 0
    while k < len(chars):
        run = k
        while (run + 1 < len(chars) and run + 1 - k < 3 and foreign(chars[run + 1]) and foreign(chars[k])
               and chars[run + 1].x == chars[run].x + chars[run].width + 1):
            run += 1
        if run > k and all(alts[j] is None and chars[j].char not in EUROPEAN for j in range(k, run + 1)):
            span = Char("", chars[k].x, chars[run].x + chars[run].width - chars[k].x, chars[k].bold,
                        glyph=Glyph("", np.zeros((WIN_H, 1), bool), chars[run].x + chars[run].width - chars[k].x,
                                    0, chars[k].bold, (), 0, "merged"))
            a = ascii_lookalike([*chars[:k], span, *chars[run + 1:]], k, L, L0, alt)
            if a is not None and len(a) == 1:
                merged[k] = (run, a[0])
        k = run + 1
    out: list[Char] = []
    k = 0
    while k < len(chars):
        c, a = chars[k], alts[k]
        if k in merged:
            out.append(merged[k][1])
            k = merged[k][0] + 1
            continue
        if a is None or (script(c.char) in REAL_SCRIPTS and c.char.isalpha() and any(
                anchor[j] for j in word_span(chars, k) if j != k)):
            out.append(c)
        else:
            out.extend(a)
        k += 1
    return respace(out)


def foreign(c: Char) -> bool:
    return c.glyph is not None and not c.glyph.atlas.startswith("ascii")


def ascii_lookalike(chars: list[Char], k: int, L: np.ndarray, L0: np.ndarray, alt: dict) -> Optional[list[Char]]:
    """The ASCII reading of foreign glyph chars[k] (see prefer_ascii), or None."""
    c, g = chars[k], chars[k].glyph
    seq = alt["seq"].get(id(g))
    if seq is not None:
        out, x = [], c.x
        for a in seq:
            out.append(Char(a.char, x, a.width, False, glyph=a))
            x += a.advance
        return out
    n = L.shape[1] - 4
    prev_end = max((p.x + p.width for p in chars[:k] if p.glyph is not None), default=-1)
    next_x = min((q.x for q in chars[k + 1:] if q.glyph is not None), default=n + 1)
    best = None
    for dx in (0, -1, 1):
        x = c.x + dx
        if x <= prev_end:
            continue
        for w in (g.width, g.width - 1, g.width + 1):
            if w < 1 or x + w >= next_x or x + w > n + 2:   # L is padded: the window ends at the last ink
                continue
            sel = np.flatnonzero((alt["w"] == w) & (alt["bold"] == c.bold) & alt["letter"])
            if not len(sel):
                continue
            s0, s1 = min(x, c.x), max(x + w, c.x + g.width)
            A = np.zeros((len(sel), WIN_H, s1 - s0), bool)
            A[:, :, x - s0:x - s0 + w] = alt["bm"][sel, :, :w]
            obs, orig = L[None, :, s0:s1], L0[None, :, s0:s1]
            top = alt["top"][sel]
            above = np.arange(WIN_H)[None, :, None] < top[:, None, None]
            body = ~above & (np.arange(WIN_H)[None, :, None] < 10)     # rows 10..11: the next line's accents
            extra = obs & ~A
            gone = A & ~orig
            top_gone = gone[:, WIN_TOP, :].any(axis=1)      # a hidden top row goes as a whole: one miss
            missing = gone.sum(axis=(1, 2)) - gone[:, WIN_TOP, :].sum(axis=1) + top_gone
            diff = missing + (extra & body).sum(axis=(1, 2))
            tol = np.maximum(1, alt["ink"][sel] // 6)
            ok = ((extra & above).sum(axis=(1, 2)) == 0) & (diff <= tol)
            for i in np.flatnonzero(ok):
                cand = (int(diff[i]), abs(dx) + abs(w - g.width), -int(alt["ink"][sel[i]]))
                if best is None or cand < best[0]:
                    best = (cand, alt["glyphs"][sel[i]], x)
    if best is None:
        return None
    a = best[1]
    return [Char(a.char, best[2], a.width, a.bold, glyph=a)]


def word_span(chars: list[Char], k: int) -> range:
    """Indices of the space-delimited word around chars[k]."""
    a = k
    while a > 0 and chars[a - 1].char != " ":
        a -= 1
    b = k
    while b + 1 < len(chars) and chars[b + 1].char != " ":
        b += 1
    return range(a, b + 1)


def respace(chars: list[Char]) -> list[Char]:
    """Re-derive the spaces from glyph positions, the way decode_row inserts them."""
    gl = [c for c in chars if c.glyph is not None or c.char == "�"]
    out: list[Char] = []
    for c in gl:
        if out:
            p = out[-1]
            gap = c.x - (p.x + p.width + (1 if p.glyph is not None else 0))
            if gap >= 2:
                for i in range(max(1, int(round(gap / SPACE_ADVANCE)))):
                    out.append(Char(" ", c.x - gap + i * SPACE_ADVANCE, SPACE_ADVANCE, False))
        out.append(c)
    return out


DIGIT_LIKE = {**dict.fromkeys("Il|\u012f\u0131\u1d09", "1"), "\u042d": "3", "\u0417": "3"}
NUMERIC = set("0123456789.,:%-+/()[]")


def fix_digits(chars: list[Char]) -> list[Char]:
    """Inside a number, read look-alikes as the digit they stand for: a 1 whose flag is hidden reads
    as I, l or |, and a 3 with a stray pixel or two (hotbar counts drawn over item icons) as
    Cyrillic Э.  Only tokens made of digits, number punctuation and these look-alikes are touched
    ('I0' -> '10', 'I.55' -> '1.55', '6Э' -> '63')."""
    start = 0
    for k in range(len(chars) + 1):
        if k == len(chars) or chars[k].char == " ":
            tok = chars[start:k]
            t = "".join(c.char for c in tok)
            if any(ch.isdigit() for ch in t) and any(ch in DIGIT_LIKE for ch in t) and \
                    all(ch in NUMERIC or ch in DIGIT_LIKE for ch in t):
                for c in tok:
                    c.char = DIGIT_LIKE.get(c.char, c.char)
            start = k + 1
    return chars


def clean_edges(chars: list[Char]) -> list[Char]:
    """Drop lone combining marks (an accent glyph by itself is a stray pixel pattern) and, while
    the rest holds a letter or digit, punctuation-only tokens at either end that a space separates
    from it (an icon's edge read as ',' or a slot frame read as '·')."""
    chars = [c for c in chars if not (c.glyph is not None and unicodedata.category(c.char) == "Mn")]
    changed = True
    while changed:
        changed = False
        for end in (0, -1):
            toks = "".join(c.char for c in chars).split()
            if len(toks) < 2:
                continue
            rest = toks[1:] if end == 0 else toks[:-1]
            if all(ch in EDGE_JUNK for ch in toks[end]) and any(ch.isalnum() for ch in "".join(rest)):
                k, step = (0, 1) if end == 0 else (len(chars) - 1, -1)
                while chars[k].char != " ":
                    k += step
                chars = chars[k:] if end == 0 else chars[:k + 1]
                while chars and chars[0].char == " ":
                    chars.pop(0)
                while chars and chars[-1].char == " ":
                    chars.pop()
                changed = True
    return chars


WEAK = set("ilIj")   # single vertical strokes: what bars, frames and grid lines decode to


def is_text(chars: list[Char]) -> bool:
    """A reading counts as text if it has an ASCII letter or digit that is not a lone stroke (i, l,
    I, j: a run of those only counts next to some other letter or digit), or a word of 3+ letters
    in one of the other alphabets of the font.  Hotbar slot frames, bars and grass decode to
    strings of exotic punctuation and stray strokes (l with cedilla, inverse bullets, i i ii)."""
    t = "".join(c.char for c in chars)
    if any(ch.isascii() and ch.isalnum() and ch not in WEAK for ch in t):
        return True
    for word in t.split():
        run, prev = 0, None
        for ch in word:
            sc = script(ch) if ch.isalpha() else None
            run = run + 1 if sc is not None and sc == prev and sc in REAL_SCRIPTS else (1 if sc in REAL_SCRIPTS else 0)
            prev = sc
            if run >= 3:
                return True
    return False


def components(mask: np.ndarray):
    """4-connected components of a boolean mask -> (label image, [(label, x0, y0, x1, y1)])."""
    runs: list[tuple[int, int, int]] = []
    parent: list[int] = []

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    prev: list[int] = []
    for y in np.flatnonzero(mask.any(axis=1)):
        d = np.diff(np.concatenate(([0], mask[y].astype(np.int8), [0])))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
        cur = []
        for x0, x1 in zip(starts, ends):
            parent.append(len(runs))
            cur.append(len(runs))
            runs.append((int(y), int(x0), int(x1)))
        if prev and runs[prev[0]][0] == y - 1:
            j = 0
            for i in cur:
                a0, a1 = runs[i][1], runs[i][2]
                while j < len(prev) and runs[prev[j]][2] < a0:
                    j += 1
                k = j
                while k < len(prev) and runs[prev[k]][1] <= a1:
                    ra, rb = find(prev[k]), find(i)
                    if ra != rb:
                        parent[rb] = ra
                    k += 1
        prev = cur
    labels = np.zeros(mask.shape, np.int32)
    boxes: dict[int, list[int]] = {}
    for i, (y, x0, x1) in enumerate(runs):
        r = find(i)
        labels[y, x0:x1 + 1] = r + 1
        b = boxes.setdefault(r, [x0, y, x1, y])
        b[0], b[1], b[2], b[3] = min(b[0], x0), min(b[1], y), max(b[2], x1), max(b[3], y)
    return labels, [(r + 1, *b) for r, b in boxes.items()]


def grid_phase(sub: np.ndarray, s: int, axis: int) -> int:
    """Phase (0..s-1) of the s-pixel block grid along an axis: block edges are where the ink
    pattern changes between neighbouring columns (axis=1) or rows (axis=0)."""
    if s == 1:
        return 0
    if axis == 1:
        t = (sub[:, 1:] != sub[:, :-1]).any(axis=0)
    else:
        t = (sub[1:] != sub[:-1]).any(axis=1)
    idx = np.flatnonzero(t) + 1
    if len(idx) == 0:
        return 0
    return int(np.bincount(idx % s, minlength=s).argmax())


def find_lines(img: np.ndarray, F: np.ndarray, s: int, glyphs: list[Glyph], index: tuple,
               min_ink: int = 5, min_score: float = 0.6, giveup_after: int = 25,
               rejects: Optional[list] = None) -> list[Line]:
    """Lines of text at GUI scale s in mask F.  Candidates that look like text but do not decode
    (columns no glyph matches: a glyph partly covered) go to `rejects` as (x, y, w, h)."""
    H, W = F.shape
    # dilate right so a line's glyphs join into one blob, and two rows down so accents join their
    # base glyph; a blob may hold several stacked lines, peeled off top-down below
    D = F.copy()
    for k in range(s, 7 * s + 1, s):
        D[:, k:] |= F[:, :-k]
    for _ in range(2):
        D[s:] |= D[:-s]
    labels, comps = components(D)
    blobs = []
    for lab, bx0, by0, bx1, by1 in comps:
        # ink of this blob only; ink of neighbouring blobs inside the same bbox is not ours
        sub = F[by0:by1 + 1, bx0:bx1 + 1] & (labels[by0:by1 + 1, bx0:bx1 + 1] == lab)
        ink = int(sub.sum())
        if ink < min_ink * s * s:
            continue
        if sub.size > 64 * s * s and ink > 0.7 * sub.size:   # a solid region, not text
            continue
        blobs.append((ink, bx0, by0, bx1, by1, sub))
    blobs.sort(key=lambda b: -b[0])
    lines: list[Line] = []
    misses = hits = 0
    for ink, bx0, by0, bx1, by1, sub in blobs:
        if not hits and misses >= giveup_after:        # nothing in the biggest blobs: wrong scale
            break
        found = 0
        retreats = 0
        sub0 = sub.copy()                      # the blob's ink before any line was peeled off

        def window(ys, xs, src=None):
            src = sub if src is None else src
            out = np.zeros((len(ys), len(xs)), bool)
            ry, rx = ys - by0, xs - bx0
            oky, okx = (ry >= 0) & (ry < src.shape[0]), (rx >= 0) & (rx < src.shape[1])
            out[np.ix_(oky, okx)] = src[np.ix_(ry[oky], rx[okx])]
            return out

        while sub.any():
            prof = sub.sum(axis=1)
            rows_ink = np.flatnonzero(prof)
            top = by0 + int(rows_ink[0])
            cols = np.flatnonzero(sub.any(axis=0))
            xl, xr = bx0 + int(cols[0]), bx0 + int(cols[-1])
            phx, phy = grid_phase(sub, s, 1), grid_phase(sub, s, 0)
            xstart = bx0 + phx + ((xl - bx0 - phx) // s) * s
            ystart = by0 + phy + ((top - by0 - phy) // s) * s
            xs = np.arange(xstart, xr + 1, s)
            # candidate window tops: the first ink row may be any window row, or a stray pixel above the
            # line; rank them by the share of the first line's ink on the ascii rows, decode the best few
            cs = np.concatenate(([0], np.cumsum(prof)))
            r_lo, r_hi = top - by0, min(len(prof), top - by0 + (GLYPH_H + 1) * s)
            total = max(1, cs[r_hi] - cs[r_lo])
            cands = []
            for j in range(-(WIN_H - 1), GLYPH_H):
                y0 = ystart + j * s
                if y0 < -WIN_TOP * s or y0 + WIN_TOP * s > by1:
                    continue
                a = min(max(r_lo, y0 + WIN_TOP * s - by0), r_hi)
                b = min(max(r_lo, y0 + (WIN_TOP + GLYPH_H) * s - by0), r_hi)
                cands.append(((cs[b] - cs[a]) / total, j, y0))
            cands.sort(reverse=True)
            best = None
            for _, _, y0 in cands[:3]:
                ys = np.arange(y0, y0 + WIN_H * s, s)
                Lw = window(ys, xs)
                chars, m, u = decode_row(Lw, index)
                key = (m - 2 * u, -u)
                if best is None or key > best[0]:
                    best = (key, y0, chars, m, u, Lw)
            if best is None:
                break
            _, y0, chars, m, u, Lread = best
            # a bad reading of a line that mixes colours (a name drawn twice with a 50% ghost
            # copy, counts over item icons): retry on the pixels of each dominant exact colour
            restricted = None
            if u > 0:
                ys = np.arange(y0, y0 + WIN_H * s, s)
                oky, okx = (ys >= 0) & (ys < H), xs < W
                Lw = window(ys, xs)
                px = img[np.ix_(ys[oky], xs[okx])]
                ink = px[Lw[oky][:, okx]].astype(np.int64)
                if len(ink):
                    keys = (ink[:, 0] << 16) | (ink[:, 1] << 8) | ink[:, 2]
                    vals, counts = np.unique(keys, return_counts=True)
                    for i in np.argsort(-counts)[:2]:
                        if counts[i] < 0.2 * len(keys):
                            break
                        v = int(vals[i])
                        c = np.array([(v >> 16) & 255, (v >> 8) & 255, v & 255], np.int16)
                        Lc = np.zeros_like(Lw)
                        Lc[np.ix_(oky, okx)] = Lw[oky][:, okx] & (np.abs(px - c) <= 2).all(axis=2)
                        chars_c, m_c, u_c = decode_row(Lc, index)
                        if (m_c - 2 * u_c, -u_c) > (m - 2 * u, -u):
                            chars, m, u, restricted, Lread = chars_c, m_c, u_c, c, Lc
            score = m / (m + u) if m + u else 0.0
            while chars and chars[0].char == " ":
                chars.pop(0)
            while chars and chars[-1].char == " ":
                chars.pop()
            # acceptance and consumed pixels follow the raw glyphs; the reported text is the repaired
            # reading, and only if it is text (junk still consumes its band)
            decoded = chars
            if len(index) > 2:
                chars = prefer_ascii(chars, Lread, window(np.arange(y0, y0 + WIN_H * s, s), xs, sub0), index[2])
            chars = fix_digits(clean_edges(chars))
            glyphs_out = [c for c in chars if c.glyph is not None]
            # a lone glyph that only matched without its top row is a stroke, not text (a bar -> T)
            emit = bool(chars) and is_text(chars) and not (len(glyphs_out) == 1 and glyphs_out[0].topless)
            matched_glyphs = [c.glyph for c in decoded if c.glyph is not None]
            # reject what wrong-scale pixel blocks decode to (dots, bars, blocks, a lone Greek or runic
            # letter): need a letter or digit, or 3+ cleanly matched shaped glyphs
            n_low = sum(g.low_info for g in matched_glyphs)
            n_ascii = sum(g.atlas.startswith("ascii") for g in matched_glyphs)
            accepted = not (m < min_ink or score < min_score or not matched_glyphs or n_low > len(matched_glyphs) / 2 or not (
                    any(g.char.isalnum() and not g.low_info for g in matched_glyphs)
                    or (len(matched_glyphs) >= 3 and m >= 12 and u == 0)))
            if accepted and len(matched_glyphs) <= 3 and n_ascii < len(matched_glyphs) and not any(
                    g.char in "0123456789" for g in matched_glyphs):
                accepted = False
            if DEBUG:
                print(f"[peel s={s} blob x{bx0}-{bx1} y{by0}-{by1}] top={top} y0={y0} xl={xl} m={m} u={u} "
                      f"restricted={None if restricted is None else tuple(int(v) for v in restricted)} "
                      f"{('OK ' if emit else 'jnk') if accepted else 'rej'} text={''.join(c.char for c in decoded)[:70]!r}"
                      f" -> {''.join(c.char for c in chars)[:70]!r}", file=sys.stderr)
            lo = max(0, top - by0)
            if not accepted and rejects is not None and m + u >= 3:
                band = sub[max(0, y0 + WIN_TOP * s - by0):max(0, y0 + (WIN_TOP + GLYPH_H) * s - by0)]
                bc = np.flatnonzero(band.any(axis=0)) if band.size else []
                if len(bc) and 3 * s <= bc[-1] + 1 - bc[0] <= 60 * 6 * s:
                    rejects.append((bx0 + int(bc[0]), y0 + WIN_TOP * s, int(bc[-1] + 1 - bc[0]), GLYPH_H * s))
            if not accepted:
                if m >= 8 and retreats < 4:
                    # something matched but the line was rejected: junk above a real line probably pulled the
                    # alignment off; drop one glyph row and retry (a few times at most)
                    retreats += 1
                    sub[lo:lo + s] = False
                else:
                    sub[lo:min(sub.shape[0], y0 - by0 + (WIN_TOP + 6) * s)] = False
                    sub[lo:lo + s] = False
                continue
            # consume the matched glyph pixels and everything from the first ink row down to ascii row 5
            # (rows 6..8 may hold the next line's accents); a colour-restricted reading clears only its colour
            before = int(sub.sum())
            for c in decoded:
                g = c.glyph
                if g is None:
                    continue
                r0, c0 = y0 - by0, xstart - bx0 + c.x * s
                blk = np.kron(g.bitmap, np.ones((s, s), bool))
                ys0, ys1 = max(r0, 0), min(r0 + blk.shape[0], sub.shape[0])
                xs0, xs1 = max(c0, 0), min(c0 + blk.shape[1], sub.shape[1])
                if ys1 > ys0 and xs1 > xs0:
                    sub[ys0:ys1, xs0:xs1] &= ~blk[ys0 - r0:ys1 - r0, xs0 - c0:xs1 - c0]
            hi = min(sub.shape[0], y0 - by0 + (WIN_TOP + 6) * s)
            if hi > lo:
                if restricted is None:
                    sub[lo:hi] = False
                else:
                    region = img[by0 + lo:by0 + hi, bx0:bx1 + 1]
                    sub[lo:hi] &= ~(np.abs(region - restricted) <= 2).all(axis=2)
            if int(sub.sum()) == before:          # nothing consumed: still guarantee progress
                sub[lo:lo + s] = False
            retreats = 0
            found += 1
            if not emit:
                continue
            ys = np.arange(y0, y0 + WIN_H * s, s)
            ok = (ys >= 0) & (ys < H)
            for c in chars:
                g = c.glyph
                if g is None:
                    continue
                xs_c = xstart + np.arange(c.x, c.x + g.width) * s
                xs_ok = xs_c < W
                px = img[np.ix_(ys[ok], xs_c[xs_ok])][g.bitmap[ok][:, xs_ok]]
                if len(px):
                    c.color = tuple(int(v) for v in np.median(px, axis=0))
            x0c = chars[0].x
            for c in chars:
                c.x -= x0c
            lines.append(Line(text="".join(c.char for c in chars), x=int(xstart + x0c * s),
                              y=int(y0 + WIN_TOP * s), width=int((chars[-1].x + chars[-1].width) * s),
                              height=GLYPH_H * s, scale=s, chars=chars, matched=m, unmatched=u))
        if not found:
            misses += 1
        hits += found
    return lines


def merge_adjacent(lines: list[Line], max_spaces: int = 3) -> list[Line]:
    """Re-join pieces of one line that a run of 2+ spaces, or glyphs with no vertical overlap
    (a '"' then a ','), split into separate blobs."""
    out: list[Line] = []
    for l in sorted(lines, key=lambda l: (l.scale, l.y, l.x)):
        p = out[-1] if out else None
        if p and p.scale == l.scale and p.y == l.y and p.source == l.source == "mcfont":
            s = l.scale
            gap = (l.x - (p.x + p.width)) / s
            n = int(round(gap / SPACE_ADVANCE))
            if gap > 0 and abs(gap - n * SPACE_ADVANCE) <= 1 and 0 <= n <= max_spaces:
                shift = (l.x - p.x) // s
                for i in range(n):
                    p.chars.append(Char(" ", p.chars[-1].x + p.chars[-1].width + 1 + i * SPACE_ADVANCE,
                                        SPACE_ADVANCE, False))
                for c in l.chars:
                    c.x += shift
                p.chars += l.chars
                p.text = p.text + " " * n + l.text
                p.width = l.x + l.width - p.x
                p.matched += l.matched
                p.unmatched += l.unmatched
                continue
        out.append(l)
    return sorted(out, key=lambda l: (l.y, l.x))


def clean(text: str) -> bool:
    """Only ASCII, the accented letters of European languages and letters of the other alphabets."""
    return all(ch.isascii() or ch in EUROPEAN or (ch.isalpha() and script(ch) in REAL_SCRIPTS) for ch in text)


def trusted(text: str) -> bool:
    """clean, and no lone letter of another alphabet inside a Latin word ('Liς.' is a damaged 'Light')."""
    if not clean(text):
        return False
    for word in text.split():
        foreign = sum(1 for ch in word if ch.isalpha() and script(ch) in REAL_SCRIPTS)
        if foreign == 1 and any(ch.isascii() and ch.isalpha() for ch in word):
            return False
    return True


def implausible(text: str) -> int:
    """Characters a real line rarely has: unknown glyphs, symbols outside ASCII and the European
    letters, lone letters of another alphabet, and dots or commas glued to the front of a word
    (a player head or ping icon beside a name reads as '.', ',' or a Hebrew letter)."""
    n = 0
    for word in text.split():
        real = sum(1 for ch in word if ch.isalpha() and script(ch) in REAL_SCRIPTS)
        lead = len(word) - len(word.lstrip(".,'`"))
        if lead and lead < len(word) and word[lead].isalnum():
            n += lead
        for ch in word:
            if ch == "\ufffd" or (not ch.isascii() and ch not in EUROPEAN
                                   and not (ch.isalpha() and script(ch) in REAL_SCRIPTS and real >= 2)):
                n += 1
    return n


def glyph_spans(l: Line) -> list[tuple[int, int]]:
    """Image x ranges of a line's glyphs (the whole box for model readings)."""
    if not l.chars:
        return [(l.x, l.x + l.width)]
    s = l.scale
    return [(l.x + c.x * s, l.x + (c.x + c.width) * s) for c in l.chars if c.char != " "]


def collide(a: Line, b: Line) -> bool:
    """Two readings claim the same pixels: their rows overlap and so do some of their glyphs (a count
    in the gap of '( )' does not collide with the brackets)."""
    if not (a.y < b.y + b.height and b.y < a.y + a.height and a.x < b.x + b.width and b.x < a.x + a.width):
        return False
    sb = glyph_spans(b)
    return any(x0 < y1 and y0 < x1 for x0, x1 in glyph_spans(a) for y0, y1 in sb)


def dedupe(lines: list[Line]) -> list[Line]:
    """Different scales/masks can explain the same pixels; keep the best-scoring, longest reading."""
    # ties go to the reading that explains more pixels (wrong-scale junk explains only edges), each
    # implausible character discounting it by a quarter: a flat-colour reading that glues a ping icon
    # to a name ('.Name') loses to the clean shadow reading 'Name', a fragment like 'ncy·' to the line
    lines = sorted(lines, key=lambda l: (-l.score, -l.matched * max(l.scale, 1) / (1 + 0.25 * implausible(l.text))))
    kept: list[Line] = []
    for l in lines:
        if not any(collide(l, k) for k in kept):
            kept.append(l)
    return splice(sorted(kept, key=lambda l: (l.y, l.x)))


def splice(lines: list[Line]) -> list[Line]:
    """Put a short reading that sits in a gap of a longer one on the same row back into it: a count
    in another colour inside brackets, 'Text Radar (' + '1' + ')'."""
    out = list(lines)
    for b in sorted(lines, key=lambda l: len(l.chars)):
        if b not in out or not b.chars or b.source != "mcfont":
            continue
        for a in out:
            if a is b or a.source != "mcfont" or a.scale != b.scale or a.y != b.y or not a.chars:
                continue
            if a.x < b.x and b.x + b.width < a.x + a.width and (b.x - a.x) % a.scale == 0:
                s = a.scale
                shift = (b.x - a.x) // s
                moved = [Char(c.char, c.x + shift, c.width, c.bold, c.color, c.glyph, c.topless) for c in b.chars]
                a.chars = respace(sorted([c for c in a.chars if c.char != " "] + [c for c in moved if c.char != " "],
                                         key=lambda c: c.x))
                a.text = "".join(c.char for c in a.chars)
                a.matched += b.matched
                a.unmatched += b.unmatched
                out.remove(b)
                break
    return out


# ------------------------------------------------------------------ classification

def classify(lines: list[Line], W: int, H: int) -> dict:
    """Group lines by where vanilla Minecraft draws things.  Heuristic but useful."""
    chat, hotbar, xp, other = [], [], None, []
    for l in lines:
        s = l.scale or 1
        gw, gh = W / s, H / s                      # GUI-unit size
        cx = (l.x + l.width / 2) / s
        cy = (l.y + l.height / 2) / s
        digits = l.text.replace(" ", "") != "" and all(c in "0123456789" for c in l.text.replace(" ", ""))
        name = nearest_color(l.chars[0].color) if l.chars and l.chars[0].color else None
        if digits and name == "xp_green" and abs(cx - gw / 2) < 20 and gh - 45 < cy < gh - 25:
            xp = l
            l.kind = "xp_level"
        elif cy > gh - 14 and abs(cx - gw / 2) < 95 and any(c.char in "0123456789" for c in l.chars):
            # stack counts: right-aligned at slot_x + 17, slots 20 GUI px apart.  A line may hold
            # several counts (plus junk from the selected-slot frame), so split into digit groups.
            tokens, cur = [], []
            for c in l.chars + [Char(" ", 0, 0, False)]:
                if c.char in "0123456789":
                    cur.append(c)
                elif cur:
                    tokens.append(cur)
                    cur = []
            for tk in tokens:
                right = l.x / s + tk[-1].x + tk[-1].width
                slot = int(round((right - (gw // 2 - 71)) / 20)) + 1
                hotbar.append({"slot": slot, "count": int("".join(c.char for c in tk)),
                               "x": int(l.x + tk[0].x * s), "y": l.y})
            l.kind = "hotbar_count"
        elif l.x / s <= 8 and cy > gh * 0.35 and l.score >= 0.9 and len(l.text) >= 3:
            l.kind = "chat"
            chat.append(l)
        else:
            other.append(l)
    hotbar.sort(key=lambda d: d["slot"])
    return {"chat": chat, "hotbar": hotbar, "xp_level": xp, "other": other}


# --------------------------------------------------------------- neural fallback

_RAPID = None
_VLM = None


# text-detector threads per process (0: one per core); batch workers use 2
DET_THREADS = int(os.environ.get("MCOCR_DET_THREADS", "0"))


def _rapid():
    global _RAPID
    if _RAPID is None:
        from rapidocr import RapidOCR
        try:    # idle threads sleep (onnxruntime spins by default)
            from rapidocr.inference_engine.onnxruntime import main as ort_main
            make = ort_main.OrtInferSession._init_sess_opts

            def no_spin(cfg):
                o = make(cfg)
                o.add_session_config_entry("session.intra_op.allow_spinning", "0")
                return o
            ort_main.OrtInferSession._init_sess_opts = staticmethod(no_spin)
        except (ImportError, AttributeError):
            pass
        params = {"Global.log_level": "error"}
        if DET_THREADS > 0:
            params["EngineConfig.onnxruntime.intra_op_num_threads"] = DET_THREADS
        _RAPID = RapidOCR(params=params)
    return _RAPID


DET_UPSCALE = int(os.environ.get("MCOCR_DET_UPSCALE", "2"))
REJECT_BOXES = os.environ.get("MCOCR_REJECT_BOXES", "1") == "1"   # read undecodable shadowed shapes too


def detect_boxes(pil: Image.Image, upscale: int = 0) -> list[tuple[float, float, float, float]]:
    """Text boxes (x, y, w, h) in image coordinates from RapidOCR's detector."""
    upscale = upscale or DET_UPSCALE
    W, H = pil.size
    big = pil.resize((W * upscale, H * upscale), Image.LANCZOS)
    res = _rapid()(np.asarray(big.convert("RGB")), use_det=True, use_cls=False, use_rec=False)
    boxes = getattr(res, "boxes", None)
    if boxes is None:
        return []
    out = []
    for box in boxes:
        xs = [p[0] / upscale for p in box]
        ys = [p[1] / upscale for p in box]
        out.append((min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)))
    return out


def read_rapidocr(crops: list[Image.Image]) -> list[tuple[str, float]]:
    out = []
    for crop in crops:
        res = _rapid()(np.asarray(crop.convert("RGB")), use_det=False, use_cls=False, use_rec=True)
        txts, scores = getattr(res, "txts", None) or (), getattr(res, "scores", None) or ()
        out.append((" ".join(txts).strip(), float(scores[0]) if len(scores) else 0.0))
    return out


def load_vlm(device: str = "auto", model_id: str = QWEN_MODEL):
    """Load the VLM once (fp16 + eager attention: about 3x faster than bf16)."""
    global _VLM
    import torch
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except Exception as e:  # the usual cause is torch and torchvision from different builds
        tv = None
        try:
            import importlib.metadata as md
            tv = md.version("torchvision")
        except Exception:
            pass
        sys.exit(f"transformers failed to import ({type(e).__name__}: {str(e)[:120]}).\n"
                 f"torch {torch.__version__} / torchvision {tv}: both must come from the same build "
                 f"(e.g. both '+rocm7.2', both '+cu130' or both '+cpu').  Reinstall them together from one "
                 f"index, e.g.\n  pip install --force-reinstall --index-url "
                 f"https://download.pytorch.org/whl/rocm7.2 torch torchvision")
    if _VLM is None:
        dev = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        if dev == "cpu":
            print(f"note: running {model_id} on the CPU (torch {torch.__version__} sees no GPU); "
                  f"expect tens of seconds per block", file=sys.stderr)
        dtype = torch.float16 if dev == "cuda" else torch.float32
        model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype, attn_implementation="eager").to(dev).eval()
        proc = AutoProcessor.from_pretrained(model_id)
        _VLM = (model, proc, dev)
    return _VLM


def read_vlm(crops: list[Image.Image], device: str = "auto", model_id: str = QWEN_MODEL,
             max_new_tokens: int = 320, prompt: str = VLM_PROMPT) -> list[tuple[str, float]]:
    """Qwen3-VL (or any transformers image-text-to-text model) reads each crop."""
    import torch
    model, proc, dev = load_vlm(device, model_id)
    out = []
    for crop in crops:
        msgs = [{"role": "user", "content": [{"type": "image", "image": crop}, {"type": "text", "text": prompt}]}]
        inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                          return_dict=True, return_tensors="pt").to(dev)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        txt = proc.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        out.append((txt.strip(), 1.0))
    return out


# instructions before the image let llama-server reuse cached prompt tokens (~27% faster), but
# it changed some correct readings ('Tracers' -> "Tracer's"), so it is off by default
LLAMA_TEXT_FIRST = os.environ.get("MCOCR_TEXT_FIRST", "0") == "1"


def read_llama(crops: list[Image.Image], url: str, max_new_tokens: int = 320, prompt: str = VLM_PROMPT,
               parallel: int = 4) -> list[tuple[str, float]]:
    """Read crops with a vision model served by llama-server (OpenAI-compatible endpoint),
    `parallel` requests at a time (start the server with -np of at least that)."""
    import base64
    import json as _json
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    import urllib.error

    def post(crop):
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        image = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}
        text = {"type": "text", "text": prompt}
        body = {"messages": [{"role": "user", "content": [text, image] if LLAMA_TEXT_FIRST else [image, text]}],
                "max_tokens": max_new_tokens, "temperature": 0, "logprobs": True}
        req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=_json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            ch = _json.load(r)["choices"][0]
        # confidence: probability of the reply's least likely token (1.0 if the server gives none)
        lps = [t["logprob"] for t in ((ch.get("logprobs") or {}).get("content") or [])]
        return ch["message"]["content"], (math.exp(min(lps)) if lps else 1.0)

    def one(crop):
        # a crop too big for the server's per-slot context comes back as HTTP 400 with both token
        # counts; image tokens grow with the area, so shrink it to fit and ask again
        for _ in range(3):
            try:
                txt, conf = post(crop)
                return txt.strip(), conf
            except urllib.error.HTTPError as e:
                msg = e.read().decode("utf-8", "replace")
                m = re.search(r"request \((\d+) tokens\) exceeds the available context size \((\d+) tokens\)", msg)
                if not m:
                    print(f"note: llama-server rejected a crop (HTTP {e.code}): {msg[:160]}", file=sys.stderr)
                    return "", 0.0
                need, have = int(m.group(1)), int(m.group(2))
                k = 0.95 * max(0.0, (have - max_new_tokens - 150) / need) ** 0.5
                if k * min(crop.size) < 28:
                    return "", 0.0
                crop = crop.resize((int(crop.width * k), int(crop.height * k)), Image.LANCZOS)
            except Exception as e:
                sys.exit(f"llama-server at {url} failed: {e}\nStart it with e.g.\n  llama-server -m Qwen3VL-8B-Instruct-Q4_K_M.gguf "
                         f"--mmproj mmproj-Qwen3VL-8B-Instruct-F16.gguf -ngl 99 --port 8080")
        print("note: a crop did not fit the llama-server context; start it with a larger -c", file=sys.stderr)
        return "", 0.0

    with ThreadPoolExecutor(max(1, parallel)) as ex:
        return list(ex.map(one, crops))


def resolve_backend(backend: str, url: str) -> str:
    """'auto' picks llama-server when one answers at url, else the transformers backend."""
    if backend != "auto":
        return backend
    import urllib.request
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=1.5) as r:
            if b"ok" in r.read():
                return "llama"
    except Exception:
        pass
    return "hf"


def clean_vlm_text(txt: str) -> str:
    """Drop lines where the model echoed its instructions instead of reading (empty crops)."""
    words = VLM_PROMPT.lower().split()
    keep = []
    for line in txt.splitlines():
        lw = line.lower().split()
        if len(lw) >= 4 and any(" ".join(lw[i:i + 4]) in " ".join(words) for i in range(len(lw) - 3)):
            continue
        keep.append(line)
    return "\n".join(keep)


def read_any(crops, backend: str, device: str, model_id: str, url: str, max_new_tokens: int = 320,
             prompt: str = VLM_PROMPT):
    reads = (read_llama(crops, url, max_new_tokens, prompt) if backend == "llama"
             else read_vlm(crops, device, model_id, max_new_tokens, prompt))
    return [(clean_vlm_text(t), c) for t, c in reads]


def neural_fallback(pil: Image.Image, engine: str, known: list[Line], device: str = "auto",
                    model_id: str = QWEN_MODEL, upscale: int = 4, backend: str = "hf",
                    url: str = "http://127.0.0.1:8080", more_boxes: tuple = ()) -> list[Line]:
    """Detect text the exact layer did not explain, crop it, upscale it, and read it."""
    W, H = pil.size
    # only confident exact lines count as explained; a shaky exact reading (resized screenshot,
    # unknown font) leaves its region to the model
    every_exact = [(k.x, k.y, k.width, k.height) for k in known]
    known = [k for k in known if k.score >= 0.95 and "\ufffd" not in k.text]
    found = list(detect_boxes(pil))
    boxes = merge_parts([part for box in found for part in unexplained_parts(box, known, W, H)])
    # extra places, read by mcrec only (the VLM reads item icons there as letters) and kept only
    # when it is sure: shapes the exact layer saw by their shadow but could not match (a glyph half
    # under an icon), and the bottom middle detected again on its own, enlarged (hotbar counts and
    # the held item's name are too small for the detector on the whole screenshot)
    probes = []
    for x, y, w, h in more_boxes:                     # glyph rows only: add detector-like margins
        probes.append(((max(0.0, x - 0.25 * h), max(0.0, y - 0.3 * h), w + 0.5 * h, 1.5 * h), "shape"))
    for fx0, fx1, fy0, fy1 in ((0.25, 0.75, 0.88, 1.0), (0.2, 0.8, 0.76, 0.92)):    # counts; item name
        sx0, sy0, sx1, sy1 = int(W * fx0), int(H * fy0), int(W * fx1), int(H * fy1)
        if sy1 - sy0 >= 24:
            probes += [((x + sx0, y + sy0, w, h), "strip") for x, y, w, h in detect_boxes(pil.crop((sx0, sy0, sx1, sy1)))]
    probes = [(b, kind) for b, kind in probes                 # not where something is read anyway
              if not any(_cover(b, o) > 0.3 for o in every_exact + [tuple(f) for f in boxes])]
    kinds = {}
    for b, kind in probes:
        kinds.setdefault(kind, []).append(b)
    probes = [(b, kind) for kind, bs in kinds.items() for b in merge_parts(bs)]
    if not boxes and not probes:
        return []

    def crop(x0, y0, x1, y1, pad_x, pad_y):
        box = (int(max(0, x0 - pad_x)), int(max(0, y0 - pad_y)), int(min(W, x1 + pad_x)), int(min(H, y1 + pad_y)))
        c = pil.crop(box)
        k = max(1, min(upscale, 2400 // max(c.width, 1), int((MAX_VLM_PIXELS[backend] / max(1, c.width * c.height)) ** 0.5)))
        return c.resize((c.width * k, c.height * k), Image.LANCZOS)

    out = []
    if engine == "rapidocr":          # single-line recogniser: one tight crop per box
        crops = [crop(x, y, x + w, y + h, 0.3 * h, 1) for x, y, w, h in boxes]
        for (x, y, w, h), (txt, conf) in zip(boxes, read_rapidocr(crops)):
            if txt:
                out.append(Line(text=txt, x=int(x), y=int(y), width=int(w), height=int(h), scale=0,
                                matched=int(conf * 100), unmatched=100 - int(conf * 100), kind="other", source=engine))
    elif engine in ("qwen", "vlm", "mcrec"):
        # mcrec first; the VLM reads what it is unsure of, one crop per box, masked to the box. Where
        # mcrec saw Minecraft text without being sure, its reading still wins when it fits the pixels
        # far better (the VLM's errors there are look-alikes: l/1, 2/3, a lost >)
        rest, mine, probe_lines, strip_rest = boxes, {}, [], []
        final: list[tuple] = []                           # boxes as read, widened ones included
        blank: list[tuple] = []
        if mcrec_available():
            import mcrec
            _, charset = mcrec.load()
            rest = []
            cut = lambda b: pil.crop((int(b[0]), int(b[1]), int(b[0] + b[2]), int(b[1] + b[3])))
            inf = mcrec.infer([cut(b) for b in boxes])
            rgb = np.asarray(pil, np.float32)
            for i, (b, (probs, p_mc)) in enumerate(zip(boxes, inf)):
                txt, conf = mcrec.ctc_decode(probs, len(probs), charset) if p_mc >= 0.5 else ("", 0.0)
                txt = fix_mcrec_digits(" ".join(txt.split()))
                if txt:
                    # the detector may have cut the line short: follow it sideways and keep the wider box when
                    # mcrec reads the same text plus more (a gap between two boxes goes to one, never both)
                    blockers = [(k.x, k.y, k.width, k.height) for k in known] + final + list(boxes[i + 1:])
                    nb = follow_line(rgb, b, blockers)
                    if nb != b:
                        (p2, mc2), = mcrec.infer([cut(nb)])
                        t2, c2 = mcrec.ctc_decode(p2, len(p2), charset)
                        t2 = fix_mcrec_digits(" ".join(t2.split()))
                        if mc2 >= 0.5 and txt in t2 and t2 != txt:
                            b, txt, conf, probs = nb, t2, c2, p2
                final.append(b)
                # dark text on a light ground (signs) has no shadow and is often in perspective: there
                # mcrec must be sure on its own and does not overrule the VLM
                dark = bool(txt) and dark_text(rgb, b)
                # one character on its own is weak evidence: icons read as 'C', armour as 'M'
                trust = MCREC_TRUST_DARK if dark or len(txt) == 1 else MCREC_TRUST
                if txt and (conf >= trust or (engine == "mcrec" and conf >= trust / 2)):
                    out.append(Line(text=txt, x=int(b[0]), y=int(b[1]), width=int(b[2]), height=int(b[3]), scale=0,
                                    matched=int(conf * 100), unmatched=100 - int(conf * 100), kind="other", source="mcrec"))
                elif float(probs[:, 0].min()) >= MCREC_EMPTY and not dark_text(rgb, b):
                    blank.append(b)                       # nothing there in any font: not worth a VLM read
                else:
                    rest.append(b)
                    if txt and not dark:
                        mine[b] = (txt, conf, probs)
            if probes:
                probes = [(b, k) for b, k in probes
                          if not any(_cover(b, f) > 0.5 for f in final if f not in rest and f not in blank)]
            if probes:
                # no font check (shapes passed the game's shadow test; counts over icons confuse it)
                for (b, kind), (probs, p_mc) in zip(probes, mcrec.infer([cut(b) for b, _ in probes])):
                    txt, conf = mcrec.ctc_decode(probs, len(probs), charset)
                    txt = fix_mcrec_digits(" ".join(txt.split()))
                    # one character on its own needs to be sure here too (a pickaxe handle reads as '2')
                    if conf >= (MCREC_TRUST_DARK if len(txt) == 1 else MCREC_TRUST) and probe_text_ok(txt, kind):
                        probe_lines.append(Line(text=txt, x=int(b[0]), y=int(b[1]), width=int(b[2]), height=int(b[3]),
                                                scale=0, matched=int(conf * 100), unmatched=100 - int(conf * 100),
                                                kind="other", source="mcrec"))
                    elif kind == "strip" and engine != "mcrec":
                        strip_rest.append(b)
        elif engine == "mcrec":
            sys.exit("--fallback mcrec needs models/mcrec.onnx and onnxruntime")
        if engine == "mcrec":
            rest = []
        src_name = "llama" if backend == "llama" else "qwen"
        if VLM_TRACE is not None:                   # for eval/vlm_boxes.py
            VLM_TRACE.append({"rest": rest, "strip": strip_rest, "mine": {repr(b): v[:2] for b, v in mine.items()}})
        crops = [box_crop(pil, b, upscale, MAX_VLM_PIXELS[backend]) for b in rest + strip_rest]
        reads = read_any(crops, backend, device, model_id, url, 200, BOX_PROMPT) if crops else []
        # an unsure reading may have its first or last letter cut by the box edge: read it again with
        # room left and right; take the new reading if the model is sure of it, or surer and it only
        # adds to the first (an unrelated reading it is barely surer of is as likely noise)
        unsure = [n for n, (t, c) in enumerate(reads[:len(rest)]) if c < VLM_RETRY and t.strip()
                  and t.strip().upper().strip(" .") not in ("NONE", "NO TEXT", "N/A")]
        if unsure:
            blockers = ([(k.x, k.y, k.width, k.height) for k in known] + [f for f in final if f not in rest + blank]
                        + list(rest) + list(strip_rest))
            wide = [box_crop(pil, widen_box(rest[n], blockers, W, 0.3), upscale, MAX_VLM_PIXELS[backend]) for n in unsure]
            for n, (t2, c2) in zip(unsure, read_any(wide, backend, device, model_id, url, 200, BOX_PROMPT)):
                t1, c1 = reads[n]
                a, b = " ".join(t2.split()), " ".join(t1.split())
                if c2 > c1 and (c2 >= 0.9 or (close_readings(a, b, 0.34) and len(a) >= len(b))):
                    reads[n] = (t2, c2)
        for n, (b, (txt, conf)) in enumerate(zip(rest + strip_rest, reads)):
            x, y, w, h = b
            txt = fix_vlm_digits(" ".join(t.strip() for t in txt.splitlines() if t.strip()))
            if txt.upper().strip(" .") in ("NONE", "NO TEXT", "N/A"):
                txt = ""
            if conf < VLM_DROP and len(txt) <= 4:   # what the model makes of icons and empty boxes
                txt = ""
            if n >= len(rest):                      # a bottom-strip box: counts, numbers, item names only
                # plain digits there are counts (at most 3); longer numbers have a point or comma
                digits = txt.replace(" ", "")
                if probe_text_ok(txt, "strip") and (len(txt) > 1 or digits.isdigit()) and not (digits.isdigit() and len(digits) > 3):
                    probe_lines.append(Line(text=txt, x=int(x), y=int(y), width=int(w), height=int(h), scale=0,
                                            matched=round(conf * 100), unmatched=100 - round(conf * 100),
                                            kind="other", source=src_name))
                continue
            src, score = src_name, round(conf * 100)
            if b in mine:
                m_txt, m_conf, probs = mine[b]
                if close_readings(m_txt, txt) and \
                        mcrec.ctc_logprob(probs, m_txt, charset) > mcrec.ctc_logprob(probs, txt, charset) + MCREC_MARGIN:
                    txt, src, score = m_txt, "mcrec", int(m_conf * 100)
            if txt:
                out.append(Line(text=txt, x=int(x), y=int(y), width=int(w), height=int(h), scale=0,
                                matched=score, unmatched=100 - score, kind="other", source=src))
    else:
        sys.exit(f"unknown fallback engine {engine}")
    # a probe where another reading already covers the spot adds nothing
    out += [p for p in probe_lines if not any(_cover((p.x, p.y, p.width, p.height), (o.x, o.y, o.width, o.height)) > 0.5
                                              for o in out)]
    out = drop_fragments([l for l in out if plausible_reading(l, W, H)])
    return sorted(out, key=lambda l: (l.y, l.x))


def probe_text_ok(t: str, kind: str) -> bool:
    """What a sure recogniser reading of an extra place must look like to be kept: plain text, not
    one letter repeated (a row of armour icons reads 'YYYYY'); in the bottom strip, a count, a number
    or words with small letters (an item's name), since icons there read as capitals."""
    if not t or not t.isascii() or not any(ch.isalnum() for ch in t) or (len(t) < 2 and not t.isdigit()):
        return False
    if len(set(t.replace(" ", ""))) == 1 and len(t) > 1:
        return False
    if kind == "strip":
        return bool(re.fullmatch(r"[\d.,:/+\-]*\d[\d.,:/+\- ]*", t)) or \
            (len(t) >= 3 and any(ch.islower() for ch in t) and sum(ch.isalpha() for ch in t) >= 3)
    return True


def _overlap(a, b) -> float:
    """Share of box a (x, y, w, h) that box b covers."""
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy / max(1e-6, a[2] * a[3])


def drop_fragments(lines: list[Line]) -> list[Line]:
    """A reading that repeats part of a longer reading of the same spot is a second box over the same
    text ('Sword' beside "Steve's Trusty Sword", the same potion timer twice)."""
    keep = []
    for l in lines:
        t = l.text.strip()
        box = (l.x, l.y, l.width, l.height)
        if not any(o is not l and t in o.text and (len(o.text) > len(t) or id(o) < id(l))
                   and _cover(box, (o.x, o.y, o.width, o.height)) >= 0.5 for o in lines):
            keep.append(l)
    return keep


def _cover(a, b) -> float:
    """How much two boxes are the same spot: the larger of the shares each covers of the other (a
    small box inside a big one counts fully)."""
    return max(_overlap(a, b), _overlap(b, a))


def widen_box(b, blockers: list[tuple], W: int, share: float) -> tuple:
    """Box b with `share` of its height added left and right, up to the nearest blocker on its row
    (text read elsewhere): the detector's box edge often cuts the first or last letter."""
    x, y, w, h = b
    x0, x1 = max(0.0, x - share * h), min(float(W), x + w + share * h)
    for bx, by, bw, bh in blockers:
        if (bx, by, bw, bh) == tuple(b) or min(y + h, by + bh) - max(y, by) <= 0.3 * min(h, bh):
            continue
        if bx + bw / 2 < x + w / 2:
            x0 = max(x0, min(x, bx + bw))
        else:
            x1 = min(x1, max(x + w, bx))
    return (x0, y, x1 - x0, h)


def dark_text(rgb: np.ndarray, b) -> bool:
    """Is the ink in box b darker than what is around it?  Brightness percentiles: with dark text
    the middle of the box sits near its bright end."""
    x, y, w, h = (int(round(v)) for v in b)
    g = rgb[max(0, y):y + h, max(0, x):x + w].mean(axis=2) if rgb.ndim == 3 else rgb[max(0, y):y + h, max(0, x):x + w]
    if g.size < 16:
        return False
    lo, mid, hi = np.percentile(g, (5, 50, 95))
    return bool(mid - lo > hi - mid)


def fix_mcrec_digits(t: str) -> str:
    """Minecraft's zero is drawn with a slash, and the recogniser sometimes names it Ø ('5Ø' for 50,
    '#Ø276'): next to a digit it is a zero."""
    return re.sub(r"(?<=\d)[Øø]|[Øø](?=\d)", "0", t)


def close_readings(a: str, b: str, share: float = 0.2) -> bool:
    """Do two readings differ by at most `share` of the characters (edit distance)?  The recogniser
    overrules the vision model only on small differences, the look-alikes it is there to fix."""
    a, b = " ".join(a.split()), " ".join(b.split())
    if not b or b in a:          # the recogniser adds what the vision model dropped ('<Steve>' + ' >!')
        return True
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] <= share * len(b)


def fix_vlm_digits(t: str) -> str:
    """The vision model reads a zero inside a number as the letter O ('H0.0%' as 'HO.0%')."""
    t = re.sub(r"(?<=\d)[Oo](?=[\d.,%])|(?<=[\d.,])[Oo](?=\d)", "0", t)
    return re.sub(r"(?<![A-Za-z])O(?=[.,]\d)|(?<=[A-Z])O(?=\.\d)", "0", t)


def follow_line(img: np.ndarray, box, blockers: list[tuple]) -> tuple:
    """Widen a box along its row while the columns beside it still hold text-like ink: the detector
    often starts a faint line late ('teve> hi' for '<Steve> hi') or stops before a last ']'. Stops
    at a gap wider than a space, at blockers (other text, as x, y, w, h) and after 2.5 box heights."""
    x, y, w, h = box
    H, W = img.shape[:2]
    r0, r1 = int(max(0, y + 0.25 * h)), int(min(H, y + 0.75 * h))
    if r1 - r0 < 2 or w < 2:
        return box
    band = img[r0:r1] if img.ndim == 3 else img[r0:r1, :, None]
    # per colour channel: green text after a white name stands out in green, not in brightness
    contrast = (band.max(axis=0) - band.min(axis=0)).max(axis=1)
    inside = contrast[int(max(0, x)):int(min(W, x + w))]
    if not len(inside):
        return box
    need = max(25.0, 0.5 * float(np.percentile(inside, 60)))
    cy = y + h / 2                             # in the way only if on this line: the boxes of the lines
    row = [b for b in blockers if b[1] < cy < b[1] + b[3]]   # above and below overlap it a little
    reach, gap = 2.5 * h, 0.45 * h

    def walk(start, step):
        last, c = None, start
        while 0 <= c < W and abs(c - start) <= reach:
            if any(b[0] <= c < b[0] + b[2] for b in row):
                break
            if contrast[c] >= need:
                last = c
            elif abs(c - (last if last is not None else start - step)) > gap:
                break
            c += step
        return last

    left, right = walk(int(x) - 1, -1), walk(int(x + w), 1)
    pad = 0.18 * h
    x0 = max(0.0, left - pad) if left is not None else x
    x1 = min(float(W), right + 1 + pad) if right is not None else x + w
    return (x0, y, x1 - x0, h)


def merge_parts(parts: list[tuple]) -> list[tuple]:
    """Boxes covering the same spot as one: neighbouring detector boxes leave the same unread gap.
    Only boxes of about the same extent merge (intersection over union), so a small HUD box inside a
    big one (3D text behind the HUD) stays its own."""
    out: list[list[float]] = []
    for x, y, w, h in sorted(parts, key=lambda p: -p[2] * p[3]):
        for q in out:
            ix = max(0, min(x + w, q[0] + q[2]) - max(x, q[0]))
            iy = max(0, min(y + h, q[1] + q[3]) - max(y, q[1]))
            if ix * iy >= 0.4 * (w * h + q[2] * q[3] - ix * iy):
                x1, y1 = max(x + w, q[0] + q[2]), max(y + h, q[1] + q[3])
                q[0], q[1] = min(x, q[0]), min(y, q[1])
                q[2], q[3] = x1 - q[0], y1 - q[1]
                break
        else:
            out.append([x, y, w, h])
    return [tuple(q) for q in out]


def superseded(l: Line, extra: list[Line]) -> bool:
    """A shaky exact line (unknown glyphs, or a weak match) that a fallback reading covers: the
    fallback read that spot as a whole, so keeping both would report the text twice."""
    if l.score >= 0.95 and "\ufffd" not in l.text:
        return False
    area = max(1, l.width * l.height)
    for e in extra:
        ix = max(0, min(l.x + l.width, e.x + e.width) - max(l.x, e.x))
        iy = max(0, min(l.y + l.height, e.y + e.height) - max(l.y, e.y))
        if ix * iy >= 0.5 * area:
            return True
    return False


def unexplained_parts(box, known: list[Line], W: int, H: int) -> list[tuple]:
    """What of a detector box still needs reading. A box confident exact lines barely cover is read
    whole; otherwise what is left beside the exact lines on their rows, and any band of rows with no
    exact line, each cut with detector-like margins (about 0.25 line heights)."""
    x, y, w, h = box
    inside = []
    covered = 0.0
    for k in known:
        ix = max(0, min(x + w, k.x + k.width) - max(x, k.x))
        iy = max(0, min(y + h, k.y + k.height) - max(y, k.y))
        if ix * iy:
            covered += ix * iy
            inside.append(k)
    if not w * h or covered / (w * h) <= 0.3:
        return [box]
    rows: list[list[Line]] = []
    for k in sorted(inside, key=lambda k: k.y):
        if rows and k.y < max(r.y + r.height for r in rows[-1]) - 0.5 * k.height:
            rows[-1].append(k)
        else:
            rows.append([k])
    parts = []
    for row in rows:
        lh = max(k.height for k in row)
        y0, y1 = min(k.y for k in row), max(k.y + k.height for k in row)
        pad = 0.25 * lh
        spans = sorted((k.x - pad, k.x + k.width + pad) for k in row)
        gaps, at = [], x
        for a, b in spans:
            if a > at:
                gaps.append((at, a))
            at = max(at, b)
        gaps.append((at, x + w))
        for a, b in gaps:
            if b - a >= 2.4 * lh:          # room for three characters or more: slivers read as fragments
                # the margins stop at exact lines above and below, so they do not leak into the reading
                beside = [k for k in known if k.x < b and k.x + k.width > a]
                top = max([0.0, y0 - 0.3 * lh] + [k.y + k.height + max(1, k.scale) for k in beside if k.y + k.height <= y0 + 0.2 * lh])
                bot = min([float(H), y1 + 0.2 * lh] + [float(k.y) for k in beside if k.y >= y1 - 0.2 * lh])
                parts.append((a, top, b - a, bot - top))
    lh = min(max(k.height for k in row) for row in rows)
    bands, at = [], y
    for row in rows:
        a, b = min(k.y for k in row) - 0.2 * lh, max(k.y + k.height for k in row) + 0.2 * lh
        if a > at:
            bands.append((at, a))
        at = max(at, b)
    bands.append((at, y + h))
    parts += [(x, a, w, b - a) for a, b in bands if b - a >= 0.9 * lh]
    return parts


MCREC_TRUST = 0.9     # recogniser confidence (lowest character probability) at which its reading is taken
MCREC_MARGIN = 6.0    # log-probability by which the recogniser's own reading must fit better than the VLM's
MCREC_TRUST_DARK = 0.97   # the same for dark text on a light ground (signs)
# VLM confidence (the probability of its least likely reply token) below which a box is read again
# with room left and right, and below which a short reading (<= 4 characters) is dropped as noise
VLM_RETRY = float(os.environ.get("MCOCR_VLM_RETRY", "0.6"))
VLM_DROP = float(os.environ.get("MCOCR_VLM_DROP", "0.5"))
# mcrec sees no character anywhere in a main box (it reads other fonts too, without trusting
# them): no text, so no VLM read. Not for dark text (small signs look blank to it) or the
# bottom strips (counts on a dimmed hotbar)
MCREC_EMPTY = float(os.environ.get("MCOCR_MCREC_EMPTY", "0.9"))
VLM_TRACE = None      # a list to record each screenshot's VLM boxes in (evaluation only)


def mcrec_available() -> bool:
    try:
        import mcrec
    except ImportError:
        return False
    return mcrec.available()


def repair_unknown(lines: list[Line], pil: Image.Image) -> None:
    """Fill the unknown glyphs (U+FFFD) of exact lines from the Minecraft-font recogniser, only where
    its reading keeps every glyph the exact layer was sure of, in order, and is itself confident."""
    todo = [l for l in lines if "\ufffd" in l.text and l.chars]
    if not todo or not mcrec_available():
        return
    import mcrec
    crops = []
    for l in todo:
        m = 0.15 * l.height
        crops.append(pil.crop((int(max(0, l.x - m)), int(max(0, l.y - m)), int(min(pil.width, l.x + l.width + m)),
                               int(min(pil.height, l.y + l.height + m)))))
    for l, (txt, conf) in zip(todo, mcrec.read(crops)):
        txt = fix_mcrec_digits(txt)
        if not txt or conf < MCREC_TRUST / 1.5:
            continue
        pat = "".join("(.{1,4}?)" if c.char == "\ufffd" else re.escape(c.char) for c in l.chars)
        mt = re.fullmatch(pat, txt)
        if mt is None:
            continue
        groups = iter(mt.groups())
        for c in l.chars:
            if c.char == "\ufffd":
                c.char = next(groups)
        l.text = "".join(c.char for c in l.chars)
        l.unmatched = 0


def box_crop(pil: Image.Image, box, upscale: int = 4, max_pixels: int = 6_000_000) -> Image.Image:
    """The pixels of one detector box on a black margin, enlarged: what a vision model reads best (a
    padded crop lets the lines above and below leak into the reading)."""
    x, y, w, h = box
    c = pil.crop((int(x), int(y), int(x + w), int(y + h)))
    m = Image.new("RGB", (c.width + 16, c.height + 4 + 16), (0, 0, 0))
    m.paste(c, (8, 10))
    k = max(1, min(upscale, 2400 // max(m.width, 1), int((max_pixels / max(1, m.width * m.height)) ** 0.5)))
    return m.resize((m.width * k, m.height * k), Image.LANCZOS)


NO_TEXT = ("no text", "does not contain", "doesn't contain", "transcribe", "the text in the image",
           "the image shows", "the image contains")
# 'there is no' is a reply about the image only next to one of these ('<Steve> there is no rain' is chat)
NO_TEXT_IF = ("there is no", "there are no")
META = ("text", "image", "visible", "readable", "legible")


def plausible_reading(l: Line, W: int, H: int) -> bool:
    """Drop what the models produce where there is no text: lines spread over a sliver of a box (a
    model looping over a 2 px detection), a row of armor or hunger icons read as 'YYYYYYYYYY',
    sentences about the image instead of its text, lines with no letter or digit, and a phrase
    repeated over and over."""
    t = l.text.strip()
    if not t or l.height < 9 or not any(ch.isalnum() for ch in t):
        return False
    low = t.lower()
    if l.source != "mcrec" and (any(p in low for p in NO_TEXT) or       # mcrec cannot write about the image
                                (any(p in low for p in NO_TEXT_IF) and any(m in low for m in META))):
        return False
    # a model stuck in a loop inside one line: far longer than a rendered line (chat wraps near 100
    # characters) and made of one phrase over and over; spam like 'DIEDIE DIEDIE' is short enough
    if len(t) > 150 and any(t.count(t[i:i + 20]) >= 3 for i in range(0, len(t) - 20, 20)):
        return False
    chars = t.replace(" ", "")
    cx, cy = (l.x + l.width / 2) / W, (l.y + l.height / 2) / H
    if len(chars) >= 5 and len(set(chars)) == 1 and cy > 0.75 and abs(cx - 0.5) < 0.3:
        return False
    return True


# --------------------------------------------------------------------------- main

_INDEX: dict = {}

def extract(path, scales=DEFAULT_SCALES, font_dir: Optional[str] = None, fallback: Optional[str] = None,
            text_colors=(), debug_dir: Optional[str] = None, device: str = "auto",
            vlm_model: str = QWEN_MODEL, glyphs: Optional[list[Glyph]] = None,
            vlm_whole: Optional[int] = None, vlm_backend: str = "hf",
            vlm_url: str = "http://127.0.0.1:8080", thorough: bool = False) -> dict:
    glyphs = glyphs if glyphs is not None else load_font(font_dir)
    if _INDEX.get("glyphs") is not glyphs:             # building it takes ~0.2 s; batch runs reuse one font
        _INDEX.update(glyphs=glyphs, index=glyph_index(glyphs))
    index = _INDEX["index"]
    pil = (path if isinstance(path, Image.Image) else Image.open(path)).convert("RGB")
    img = np.asarray(pil).astype(np.int16)
    H, W, _ = img.shape
    lines: list[Line] = []
    colors = list(text_colors) + [c for c in auto_text_colors(img) if c not in text_colors]
    # one mask per flat colour: in a union, text of one colour merges with GUI frames, icons and text
    # of the others into blobs that no longer decode (an F3 line next to a white tab list)
    flats = [("label" if c == GUI_LABEL else "flat", color_mask(img, c, tol=2 if c not in text_colors else 12))
             for c in colors]
    flats = [(name, F, likely_scales(F, scales)) for name, F in flats]
    spre, opre = shadow_pre(img), outline_pre(img)

    rejects: dict = {}

    def run(name, M, s, min_score):
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            Image.fromarray((M * 255).astype(np.uint8)).save(os.path.join(debug_dir, f"mask_{name}_x{s}.png"))
        found = find_lines(img, M, s, glyphs, index, 5, min_score,
                           rejects=rejects.setdefault(s, []) if name == "shadow" and REJECT_BOXES and fallback else None)
        for l in found:
            l.mask = name
        return found

    shadow_scales = set()
    for s in scales:
        found = run("shadow", shadow_mask(img, s, pre=spre), s, 0.6)
        shadow_scales |= {s} if found else set()
        lines += found + run("outline", outline_mask(img, s, pre=opre), s, 0.99)
    # flat colours read what the shadow passes left (text without a shadow: old F3, container
    # titles), at the scales their runs suggest or where shadowed text was found
    confident = [l for l in lines if l.score >= 0.95 and trusted(l.text) and len(l.text.strip()) >= 4]

    def unexplained(F):
        """F without the pixels of confident lines, unless F's colour carries on right beside a
        line: then that line may be a fragment of a longer one in this colour, which F should read."""
        E = np.zeros_like(F)
        for l in confident:
            y0, y1, x0, x1, m = l.y, l.y + l.height, l.x, l.x + l.width + l.scale, 6 * l.scale
            if not (F[y0:y1, max(0, x0 - m):x0].any() or F[y0:y1, x1:x1 + m].any()):
                E[y0:y1, x0:x1] = True
        return F & ~E

    flats = [(name, unexplained(F), ok) for name, F, ok in flats]
    for s in scales:
        # colours whose runs point at this scale first; once any text turns up at it (or shadowed
        # text did), the other colours are read at it too (a key name in white beside grey text)
        first = [f for f in flats if s in f[2] or s in shadow_scales or thorough]
        found = []
        for name, F, _ in first:
            found += run(name, F, s, 0.9)
        if found:
            for name, F, _ in flats:
                if all(F is not f[1] for f in first):
                    found += run(name, F, s, 0.9)
        lines += found
    lines = merge_adjacent(dedupe(lines))
    if fallback:
        repair_unknown(lines, pil)
    vlm_backend = resolve_backend(vlm_backend, vlm_url) if (fallback in ("qwen", "vlm") or vlm_whole) else vlm_backend
    # shadowed shapes that did not decode, at scales where shadowed text was found: glyphs partly
    # covered (a label half under the one below it) that the text detector does not box
    cand = [b for bs in rejects.values() for b in bs]
    extra = neural_fallback(pil, fallback, lines, device, vlm_model, backend=vlm_backend, url=vlm_url,
                            more_boxes=cand) if fallback else []
    if extra:
        lines = [l for l in lines if not superseded(l, extra)]
    groups = classify(lines, W, H)
    groups["other"] += extra
    whole = None
    if vlm_whole:
        # the whole screenshot through the VLM, no detector, no exact layer: for comparison
        k = max(1, int(vlm_whole))
        cap = MAX_VLM_PIXELS[vlm_backend]
        if W * H * k * k > cap:
            k_fit = (cap / (W * H)) ** 0.5
            print(f"note: --vlm-whole {k} would send {W * k}x{H * k} px to the model; capping at "
                  f"{k_fit:.2f}x ({cap / 1e6:.1f} Mpx) for the {vlm_backend} backend", file=sys.stderr)
            k = k_fit
        big = pil if k == 1 else pil.resize((int(W * k), int(H * k)), Image.LANCZOS)
        import time
        if vlm_backend != "llama":
            load_vlm(device, vlm_model)             # so the time below is inference only
        t = time.time()
        txt = read_any([big], vlm_backend, device, vlm_model, vlm_url, max_new_tokens=1024)[0][0]
        whole = {"model": vlm_model if vlm_backend != "llama" else f"llama-server@{vlm_url}",
                 "upscale": round(k, 2), "seconds": round(time.time() - t, 1),
                 "text": txt, "lines": [l.strip() for l in txt.splitlines() if l.strip()]}
    return {
        "vlm_whole": whole,
        "image": path if isinstance(path, str) else None, "width": W, "height": H,
        "chat": [line_dict(l) for l in groups["chat"]],
        "hotbar": groups["hotbar"],
        "xp_level": int(groups["xp_level"].text) if groups["xp_level"] else None,
        "other": [line_dict(l) for l in groups["other"]],
        "lines": [line_dict(l) for l in lines + extra],
    }


def line_dict(l: Line) -> dict:
    return {"text": l.text, "x": l.x, "y": l.y, "w": l.width, "h": l.height, "scale": l.scale,
            "kind": l.kind, "source": l.source, "confidence": round(l.score, 3),
            "colors": l.colors() if l.chars else None}


def parse_color(sx: str) -> tuple:
    sx = sx.lstrip("#")
    if sx in MC_COLORS:
        return MC_COLORS[sx]
    return tuple(int(sx[i:i + 2], 16) for i in (0, 2, 4))


_BATCH = {}


def _batch_init(font_dir, scales, colors, fallback, backend, url, device, vlm_model, thorough=False, threads=0):
    global DET_THREADS
    if threads and not os.environ.get("MCOCR_DET_THREADS"):
        DET_THREADS = threads
    _BATCH["glyphs"] = load_font(font_dir)
    _BATCH["args"] = (scales, font_dir, fallback, colors, device, vlm_model, backend, url, thorough)


def _batch_one(path):
    import time
    import traceback
    scales, font_dir, fallback, colors, device, vlm_model, backend, url, thorough = _BATCH["args"]
    t = time.time()
    try:
        r = extract(path, scales, font_dir, fallback, colors, None, device, vlm_model, _BATCH["glyphs"], None, backend, url,
                    thorough)
        r["seconds"] = round(time.time() - t, 2)
    except (Exception, SystemExit):     # SystemExit too: a dead worker would lose the image and stall the pool
        r = {"image": path, "error": traceback.format_exc()[-600:], "seconds": round(time.time() - t, 2)}
    return r


def batch_main(a, scales, colors):
    """Many images: a process pool, one JSON line per image, progress on stderr."""
    import time
    from multiprocessing import Pool
    backend = a.vlm_backend
    if a.fallback in ("qwen", "vlm") or a.vlm_whole:
        backend = resolve_backend(a.vlm_backend, a.vlm_url)
        if a.jobs > 1 and backend == "hf":
            print("note: --jobs with the transformers backend would load the model per process; using 1 job",
                  file=sys.stderr)
            a.jobs = 1
    # two detector threads per worker: one per core per worker only burns CPU
    init_args = (a.font_dir, scales, colors, a.fallback, backend, a.vlm_url, a.device, a.vlm_model, a.thorough,
                 2 if a.jobs > 1 else 0)
    out = open(a.jsonl, "w") if a.jsonl else sys.stdout
    t0 = time.time()
    done = 0
    try:
        if a.jobs > 1:
            with Pool(a.jobs, initializer=_batch_init, initargs=init_args) as pool:
                for r in pool.imap_unordered(_batch_one, a.image, chunksize=2):
                    out.write(json.dumps(r, ensure_ascii=False) + "\n")
                    out.flush()
                    done += 1
                    if done % 25 == 0:
                        print(f"{done}/{len(a.image)} images, {time.time() - t0:.0f}s", file=sys.stderr)
        else:
            _batch_init(*init_args)
            for p in a.image:
                out.write(json.dumps(_batch_one(p), ensure_ascii=False) + "\n")
                out.flush()
                done += 1
                if done % 25 == 0:
                    print(f"{done}/{len(a.image)} images, {time.time() - t0:.0f}s", file=sys.stderr)
    finally:
        if a.jsonl:
            out.close()
    print(f"{done} images in {time.time() - t0:.0f}s" + (f" -> {a.jsonl}" if a.jsonl else ""), file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="+")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    ap.add_argument("--scale", type=int, help="GUI scale (default: try 1-8 and keep the best reading)")
    ap.add_argument("--font-dir", help="directory with the font files (default: ./font next to this script)")
    ap.add_argument("--fallback", choices=["rapidocr", "qwen", "vlm", "mcrec"],
                    help="detect and read text the exact matcher did not explain: vlm (= qwen) reads each box with the "
                         "Minecraft-font recogniser (models/mcrec.onnx) and, where it is unsure, the VLM chosen by "
                         "--vlm-backend; mcrec uses the recogniser alone (CPU, no VLM); rapidocr uses RapidOCR's recogniser")
    ap.add_argument("--vlm-backend", choices=["auto", "hf", "llama"], default="auto",
                    help="llama: a GGUF model served by llama-server (see serve_vlm.sh); hf: transformers + torch "
                         "(Qwen3-VL-2B); auto (default): llama if a server answers at --vlm-url, else hf")
    ap.add_argument("--vlm-url", default="http://127.0.0.1:8080", help="llama-server base URL for --vlm-backend llama")
    ap.add_argument("--device", default="auto", help="for --fallback qwen: auto | cuda | cpu")
    ap.add_argument("--vlm-model", default=QWEN_MODEL, help="for --fallback qwen: transformers model id")
    ap.add_argument("--vlm-whole", type=int, nargs="?", const=2, default=None, metavar="UPSCALE",
                    help="also send the whole image (upscaled N x, default 2) to the VLM and show its "
                         "transcription, for comparison with the exact layer")
    ap.add_argument("--text-color", action="append", default=[], metavar="RRGGBB|name",
                    help="also look for un-shadowed text of this colour (repeatable)")
    ap.add_argument("--debug-dir", help="write foreground masks here")
    ap.add_argument("--jobs", type=int, default=1, help="process images in N parallel processes (exact layer only)")
    ap.add_argument("--jsonl", help="write one JSON object per image to this file (use with many images)")
    ap.add_argument("--thorough", action="store_true",
                    help="look for shadowless flat-colour text at every GUI scale, not only the likely ones "
                         "(about 2x slower; finds a little more, e.g. minimap labels)")
    a = ap.parse_args(argv)
    scales = (a.scale,) if a.scale else DEFAULT_SCALES
    colors = [parse_color(c) for c in a.text_color]
    glyphs = load_font(a.font_dir)
    if a.jobs > 1 or a.jsonl:
        return batch_main(a, scales, colors)
    backend = resolve_backend(a.vlm_backend, a.vlm_url) if (a.fallback in ("qwen", "vlm") or a.vlm_whole) else a.vlm_backend
    if a.vlm_backend == "auto" and backend != "auto" and (a.fallback in ("qwen", "vlm") or a.vlm_whole):
        print(f"vlm backend: {backend}" + (f" ({a.vlm_url})" if backend == "llama" else f" ({a.vlm_model})"), file=sys.stderr)
    results = [extract(p, scales, a.font_dir, a.fallback, colors, a.debug_dir, a.device, a.vlm_model, glyphs,
                       a.vlm_whole, backend, a.vlm_url, a.thorough) for p in a.image]
    if a.json:
        json.dump(results if len(results) > 1 else results[0], sys.stdout, indent=2, ensure_ascii=False)
        print()
        return
    for r in results:
        print(f"== {r['image']} ({r['width']}x{r['height']})")
        if r["chat"]:
            print("chat:")
            for l in r["chat"]:
                cols = " | ".join(f"{c['name']}:{c['text']!r}" for c in (l["colors"] or [])[:3])
                print(f"  [{l['confidence']:.2f}] {l['text']}    <{cols}>")
        if r["hotbar"]:
            print("hotbar counts: " + ", ".join(f"slot {d['slot']}={d['count']}" for d in r["hotbar"]))
        if r["xp_level"] is not None:
            print(f"xp level: {r['xp_level']}")
        if r["other"]:
            print("other text:")
            for l in r["other"]:
                print(f"  [{l['source']} {l['confidence']:.2f}] ({l['x']},{l['y']}) {l['text']}")
        if r.get("vlm_whole"):
            w = r["vlm_whole"]
            print(f"{w['model']} on the whole image at {w['upscale']}x ({w['seconds']} s), raw transcription:")
            for t in w["lines"]:
                print(f"  {t}")


if __name__ == "__main__":
    main()
