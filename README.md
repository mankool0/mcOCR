# mcOCR: all the text in Minecraft screenshots

Give it a folder of screenshots; it writes one JSON file per screenshot with every line of text it
found, where it is, its colours, and which reader produced it. It reads in three layers:

1. **Exact font matching** (CPU): Minecraft's own bitmap font, pixel for pixel. Chat, HUD, F3,
   GUI titles, hotbar counts. When it reads a line, the line is exactly right.
2. **mcrec** (CPU): a small recogniser trained for this project on the Minecraft font; it reads what
   the first layer cannot match (text over icons, resized screenshots, signs, italics).
3. **A vision model** (GPU, Qwen3-VL-8B through llama.cpp): everything else, e.g. hack-client
   menus in TrueType fonts, nametags, sign text.

Measured on 50 text-heavy screenshots it had never seen: 92 % of output lines and 97.7 % of
characters exactly right, and about 97 % of the visible text lines found. The misses are mostly
faint minimap letters, fading chat, and piles of overlapping nametags.

## What's in this folder

| | |
|---|---|
| `read_screenshots.py` | the script you run |
| `mc_ocr.py`, `mcrec.py` | the engine (`mc_ocr.py` also works on single screenshots, see below) |
| `font/` | the Minecraft font: definition and atlases (nothing is extracted from a game jar) |
| `models/` | the mcrec recognisers (`mcrec.onnx` main, `mcrec2/` second opinion) |
| `requirements.txt` | Python packages, pinned to the tested versions |
| `download_vlm.py` | downloads the vision model into `vlm/` |
| `serve_vlm.sh` | starts llama-server with it (Linux/macOS; the Windows command is below) |

## Setup

You need Python 3.12 or newer and, for the vision model, llama.cpp and a GPU with **8 GB of VRAM or more**
(AMD or NVIDIA). Without a GPU you can still run the first two layers (`--no-vlm`).

### 1. Python packages (same for AMD, NVIDIA and CPU-only)

The Python side runs on the CPU only, so there is nothing GPU-specific to install here.

With [uv](https://docs.astral.sh/uv/) (it fetches a recent Python if yours is older than 3.12):

```
uv venv --python ">=3.12" .venv
uv pip install --python .venv -r requirements.txt
```

Or with plain Python 3.12 or newer (3.12, 3.13 and 3.14 tested):

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt          # Windows: .venv\Scripts\pip install -r requirements.txt
```

Then run everything below with `.venv/bin/python` (Windows: `.venv\Scripts\python`).

### 2. llama.cpp (runs the vision model on the GPU)

You need the `llama-server` program from [llama.cpp](https://github.com/ggml-org/llama.cpp),
a build recent enough to support Qwen3-VL (late 2025 or newer; tested with build 11621). The
simplest way is a prebuilt zip from the
[releases page](https://github.com/ggml-org/llama.cpp/releases): unpack it anywhere and note
where `llama-server` (Windows: `llama-server.exe`) is.

**AMD (Radeon):**

- Linux or Windows: take the release zip with **`vulkan`** in its name. It needs the normal
  graphics driver's Vulkan support: on Linux that is Mesa (RADV), already installed on most
  distributions (check with `vulkaninfo --summary`); on Windows the AMD Adrenalin driver.
  This is what the project was tested with (an AMD card, Linux, Vulkan).
- Building it yourself instead (Linux):
  ```
  # Fedora: sudo dnf install cmake gcc-c++ vulkan-headers vulkan-loader-devel glslc
  # Ubuntu: sudo apt install cmake g++ libvulkan-dev glslc
  git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
  cmake -B build -DGGML_VULKAN=ON && cmake --build build --config Release -j 8
  # -> build/bin/llama-server
  ```
- ROCm (`-DGGML_HIP=ON`) also works but is more work to set up and not needed: Vulkan is fast
  enough here.

**NVIDIA (GeForce/RTX):**

- Windows: take the release zip with **`cuda`** in its name, plus the matching `cudart-...` zip
  (the CUDA runtime DLLs), and unpack both into the same folder. A recent NVIDIA driver is all
  else it needs.
- Linux: the **`vulkan`** zip works on NVIDIA too (with the proprietary driver, which includes
  Vulkan). For CUDA, build it (needs the CUDA toolkit):
  ```
  git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
  cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j 8
  # -> build/bin/llama-server
  ```

### 3. The vision model (6.2 GB)

```
.venv/bin/python download_vlm.py
```

downloads Qwen3-VL-8B-Instruct (Q4_K_M) and its image encoder (`mmproj`) from Hugging Face into
`vlm/`, checks their SHA-256, and continues where it stopped if interrupted (any Python 3 can run
it; it needs no packages). It uses about 7 GB of VRAM with 4 parallel slots.

For a GPU with less than 8 GB there is a smaller one: `download_vlm.py --model 2b` (Qwen3-VL-2B,
2.7 GB, about 4 GB of VRAM), then `--model 2b` when running. The pipeline was tuned and measured
with the 8B model only, so expect more mistakes from the 2B one.

To fetch the files by hand instead: they are `Qwen3VL-8B-Instruct-Q4_K_M.gguf` and
`mmproj-Qwen3VL-8B-Instruct-F16.gguf` from https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF;
put both in `vlm/`.

## Running it

The easiest way: let the script start and stop llama-server itself.

```
.venv/bin/python read_screenshots.py ~/screenshots out/ --llama-server ~/llama.cpp/build/bin/llama-server
```

Or keep llama-server running in another terminal and point the script at it:

```
./serve_vlm.sh ~/llama.cpp/build/bin/llama-server         # Linux/macOS (8b; or: ... llama-server 2b)
# Windows (one line):
#   llama-server.exe -m vlm\Qwen3VL-8B-Instruct-Q4_K_M.gguf --mmproj vlm\mmproj-Qwen3VL-8B-Instruct-F16.gguf -ngl 99 -c 8192 -np 4 --port 8080
.venv/bin/python read_screenshots.py ~/screenshots out/
```

Useful options:

| option | |
|---|---|
| `--limit 20` | do just 20 screenshots (try it out first) |
| `--model 2b` | the small vision model, for `--llama-server` to start (default `8b`). A server you started yourself runs the model you gave it; asking for the other one stops the run |
| `--combine` | at the end, also write `out/results.jsonl`: every screenshot's result, one per line |
| `--jobs N` | screenshots in parallel (default 4). The vision model is the bottleneck; 4-6 is plenty |
| `--no-vlm` | no vision model and no GPU: layers 1 and 2 only. Fast, but text in non-Minecraft fonts (most hack-client menus) is not read at all |
| `--retry-errors` | redo screenshots that failed (e.g. a broken image file) |
| `--vlm-url URL` | llama-server somewhere else (default `http://127.0.0.1:8080`) |

**Stopping and resuming.** Stop it any time (Ctrl-C, closing the terminal, a crash, a reboot).
Run the same command again and it continues: every screenshot that already has its JSON is
skipped. A JSON file only appears once its screenshot is completely done, so a stop loses at most
the few screenshots that were in progress. If llama-server dies, the run stops by itself rather
than writing half-read results; start the server again and rerun.

Screenshots are found in sub-folders too; the output mirrors the folder structure:
`screenshots/2024/shot.png` → `out/2024/shot.png.json`. Use a separate output folder for
`--no-vlm` runs: a resumed run does not redo screenshots done in the other mode.

**Speed.** About 8-9 seconds per screenshot on a current desktop CPU and GPU (so roughly 24
hours for 10,000 screenshots), less if nothing else is using the GPU. `--no-vlm` takes about
4 s per screenshot with the default 4 workers; it only uses the CPU, so there more `--jobs` does
help (each worker keeps about 1.5 cores busy). The script prints progress and an estimate of the
time left every 30 seconds.

## Output

`out/<screenshot path>.json`:

```json
{
  "image": "2024/shot.png",
  "mcocr": "2026-10-18",
  "mode": "vlm",
  "vlm": "Qwen3VL-8B-Instruct-Q4_K_M.gguf",
  "seconds": 7.4,
  "width": 1920, "height": 1080,
  "chat":   [ line, ... ],
  "hotbar": [ {"slot": 6, "count": 32, "x": 994, "y": 998}, ... ],
  "xp_level": 4,
  "other":  [ line, ... ],
  "lines":  [ line, ... ]
}
```

`mode` is `vlm` or `no-vlm`, and `vlm` names the vision model llama-server had loaded (`null` with
`--no-vlm`). `lines` holds everything found, top to bottom; `chat`, `hotbar`, `xp_level` and `other` sort the
same lines by where vanilla Minecraft draws things (a rough guide, not exact). Each `line`:

```json
{"text": "<Steve> hello", "x": 4, "y": 882, "w": 210, "h": 16, "scale": 2,
 "kind": "chat", "source": "mcfont", "confidence": 1.0,
 "colors": [{"text": "<Steve>", "rgb": [255, 255, 255], "name": "white", "bold": false}, ...]}
```

- `x, y, w, h`: the box in pixels; `scale`: the GUI scale (exact-layer lines only).
- `source`: `mcfont` (exact font match), `mcrec` (the recogniser) or `llama` (the vision model).
- `confidence`: 0-1. For `mcfont` lines it is how completely the pixels matched (1.0 = exactly);
  for `mcrec` its lowest character probability; for `llama` the probability of the least likely
  token of the model's reply. Vision-model lines below about 0.7 are often partly wrong.
- `colors`: runs of text by colour (exact-layer lines only).

A screenshot that could not be read (a broken file) gets `{"image": ..., "error": "..."}` instead.

## One screenshot at a time

```
.venv/bin/python mc_ocr.py shot.png --fallback vlm --vlm-backend llama     # readable report
.venv/bin/python mc_ocr.py shot.png --fallback vlm --vlm-backend llama --json
```

## Troubleshooting

- **"no vision model server answers"**: start llama-server (above) or pass `--llama-server PATH`.
  If the script started it and it failed, `out/llama-server.log` says why (most often: not
  enough VRAM; close other GPU programs, or use fewer slots: `--slots 2` with `--llama-server`,
  `./serve_vlm.sh PATH 8b 8080 2` otherwise).
- **Everything is slow and the GPU is idle**: llama-server is running on the CPU.
  `llama-server --list-devices` should list your card (e.g. `Vulkan0: AMD Radeon RX ...` or
  `CUDA0: NVIDIA GeForce RTX ...`); if it lists nothing, that is a CPU-only build (get the
  Vulkan or CUDA one) or the GPU driver's Vulkan/CUDA support is missing.
- **Vision-model lines all have confidence 1.0**: that llama-server build does not report token
  probabilities; the pipeline then cannot re-read or drop unsure readings. Use a newer build.

## Font files

`font/` contains Minecraft's font definition and atlases, © Mojang Studios. They are included so
the exact layer works without the game installed. This project is not affiliated with or endorsed
by Mojang or Microsoft.
