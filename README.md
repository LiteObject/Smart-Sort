# Smart-Sort

Organize a folder by **what its files are about**, not by name, date, or extension.

Smart-Sort turns each file into an embedding vector, clusters the vectors, and moves
related files into `Cluster_N_<keywords>` folders. With the multimodal backend it also
handles **images, audio, and video** in the same vector space as documents. Everything
runs locally — no API keys, no uploads.

## Quick start

```powershell
pip install -r requirements.txt

# 1. Preview what would happen (nothing is moved by default)
python smart_sort.py "D:\Downloads"

# 2. Apply it
python smart_sort.py "D:\Downloads" --apply

# 3. Changed your mind? Reverse the run
python smart_sort.py --undo ".smart_sort\manifest-20250101-120000.jsonl" --apply
```

Out of the box Smart-Sort uses a **TF-IDF fallback** that needs no model. For real
semantic grouping, install [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
and pass a GGUF embedding model:

```powershell
pip install llama-cpp-python
python smart_sort.py "D:\Docs" --model models\embeddinggemma-2.gguf --apply
```

`--backend auto` (the default) uses the GGUF model when one is given or found as
`embeddinggemma-2.gguf` in the working directory, and falls back to TF-IDF otherwise.

### Multimodal: images, audio, and video

The GGUF path is **text-only** in llama.cpp. To embed media, use the
sentence-transformers backend with [EmbeddingGemma 2](https://huggingface.co/unsloth/embeddinggemma-2),
which maps text, images, audio, and video into one 768-d space:

```powershell
# CUDA build of PyTorch (recommended; CPU-only works but is slower)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-multimodal.txt

# Preview sorting documents + media together
python smart_sort.py "D:\Downloads" --backend st --model unsloth/embeddinggemma-2 --timings
```

- `--media image,audio,video` selects which media kinds are embedded (default: all).
- `--device cuda` / `--device cpu` and `--dtype bf16` are auto-detected.
- Do not use `fp16`: EmbeddingGemma 2 produces NaN embeddings in half precision.
- On Windows, if FFmpeg is not on `PATH`, Smart-Sort exposes PyAV's bundled
  FFmpeg libraries to TorchCodec automatically (cached under
  `%LOCALAPPDATA%\smart-sort`).
- Media vectors and text vectors share one space, so a photo of a receipt can cluster
  with a receipt PDF. Media-only clusters are named by nearest text label
  (`Cluster_4_receipts`).
- Text can still come from the GGUF model, but don't mix runtimes in one clustering
  run — quantized GGUF vectors and bf16 sentence-transformers vectors are not
  perfectly comparable. Use `--backend st` for everything when media is involved.

## How it works

| Step | What happens |
| --- | --- |
| 1. Collect | Files are listed deterministically; symlinks, hidden files, `Cluster_*` folders, and common noise dirs (`.git`, `node_modules`, …) are skipped. |
| 2. Extract | Plain text is read from `.txt`/`.md`/code/CSV/…, `.pdf` (pypdfium2's C++ PDFium engine, with pypdf fallback), and `.docx` (python-docx). Unknown extensions are accepted only when the bytes look like text. PDF/DOCX parsing can use worker processes (`--jobs`), but the automatic mode stays sequential unless a batch is genuinely large enough to amortize worker startup. Images/audio/video are passed to the multimodal backend when enabled. Binary files that no backend can read are skipped individually instead of aborting the run. |
| 3. Embed | Text is split into up to 4 chunks of 4 000 chars — sampling the start *and* end of long documents — and chunks are mean-pooled and L2-normalized per document. Media files get one vector each. Embeddings are cached in `.smart_sort/`. |
| 4. Cluster | K-means with `n_init=10` and a fixed seed over all documents and media. `--k` sets the count, `--auto-k` chooses it by silhouette score, otherwise `sqrt(n/2)` is used (capped at 20 and at the file count). |
| 5. Name | Folder names get keywords from the TF-IDF terms that most distinguish each cluster (`Cluster_3_late_fee`). Media-only clusters get the nearest label from a built-in vocabulary (`receipts`, `screenshots`, `wallpapers`, …). `--no-labels` keeps plain `Cluster_N`. |
| 6. Move | Only with `--apply`. Collisions get ` (2)` suffixes; nothing is overwritten. Each successful move is appended to a JSONL manifest immediately, so a crash still leaves a valid undo record. |

## Safety model

- **Dry run by default.** Files are only moved when `--apply` is passed.
- **Undo.** Every applied run writes `.smart_sort/manifest-<timestamp>.jsonl`.
  `--undo <manifest>` prints the reverse plan; add `--apply` to execute it. SHA-256
  checksums are verified before a file is moved back, and emptied cluster folders are removed.
- **No overwrites.** Destination name clashes become `name (2).ext`.
- **No symlink traversal**, no hidden/system files (Windows attributes, `desktop.ini`, Office `~$` lock files; dot-prefixed files are skipped only on non-Windows), no state folder re-ingestion.
- Files that cannot be read are reported and skipped; the rest of the run continues.

## Options

| Option | Description |
| --- | --- |
| `--apply` | Perform the moves (otherwise preview only). |
| `--undo MANIFEST` | Reverse a previous run (dry run unless `--apply`). |
| `--k N` | Number of clusters. |
| `--auto-k` | Pick `k` by silhouette score. |
| `-r`, `--recursive` | Include files in subfolders. |
| `--backend {auto,gguf,tfidf,st}` | Embedding backend (`st` = multimodal sentence-transformers). |
| `--model PATH_OR_ID` | GGUF file path, or HF id/local path for `st` (default `unsloth/embeddinggemma-2`). |
| `--media KINDS` | Media kinds for `st` (default `image,audio,video`). |
| `--device DEVICE` | `auto`, `cpu`, `cuda`, `cuda:0` (default auto). |
| `--dtype DTYPE` | `auto`, `bf16`, `fp32` (default auto; avoid `fp16`). |
| `--batch-size N` | Encode batch size for `st` (default 32). |
| `--text-prompt NAME` | Task prompt for text embeddings (e.g. `Document`, `SearchQuery`). |
| `--n-ctx N` | GGUF context size (default 2048). |
| `--max-chars N` | Characters per chunk (default 4000). |
| `--max-chunks N` | Chunks sampled per document (default 4). |
| `--max-pdf-pages N` | Pages read per PDF (default 20). |
| `--min-chars N` | Skip files with less text (default 20). |
| `--jobs N` | Worker processes for PDF/DOCX extraction; 0 = auto (only when estimated parse time exceeds worker startup), 1 = sequential, N = force N workers. |
| `--timings` | Print per-phase durations (collect/extract/embed/cluster/apply). |
| `--seed N` | Random seed (default 0). |
| `--no-labels` | Plain `Cluster_N` folder names. |
| `--no-cache` | Ignore the embedding cache. |
| `--exclude PATTERN` | Extra name glob to skip (repeatable). |
| `-v`, `--verbose` | List every planned move. |

## Development

```powershell
python -m unittest discover -s tests -v
```

The tests use a deterministic fake embedder plus the TF-IDF backend, so they need
no model download. Use `--timings` on a real folder to see where time is actually
spent before considering any performance rewrite.

To compare PDF engines, set `SMART_SORT_PDF_ENGINE=legacy` to use the old pypdf
path (default is PDFium via pypdfium2).

## Limitations

- Text extraction: scanned/image-only PDFs yield nothing without OCR (they can still be
  embedded as images if saved as such).
- The GGUF backend is text-only; media requires `--backend st` and the multimodal model.
- TF-IDF fallback groups by shared vocabulary, not meaning.
- Media decoding depends on `PyAV`/`soundfile` (installed with the multimodal extra) and
  on codecs the OS can decode; unsupported files are skipped individually.
- Truly binary formats (archives, installers, executables, unknown formats) are not
  embedded by any backend and stay in place.
- Each file joins exactly one cluster; mixed-topic documents get one averaged vector.
- Clustering is unsupervised, so cluster numbering is arbitrary between runs.
- `.smart_sort/` holds the cache and manifests — leave it in place to keep undo working.