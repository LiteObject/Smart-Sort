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

# 3. Changed your mind? Use the manifest path printed by the applied run
python smart_sort.py --undo "D:\Downloads\.smart_sort\manifest-<run-id>.jsonl" --apply
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

## Organize into existing and new folders

The opt-in `--organize` mode previews loose files in the selected root against
eligible top-level folders. It uses up to 25 deterministically selected reference
files per folder, including existing `Cluster_*` folders. Reference files are
never moved. Folders without usable references cannot receive automatic matches.

Scores are cosine similarities, **not probabilities**. No thresholds are assumed:
without explicit thresholds files remain marked for review. The values below are
illustrative, not calibrated recommendations:

```powershell
# Preview suggestions and write a new plan (the parent directory must exist)
python smart_sort.py "D:\Docs" --organize --backend tfidf `
  --match-threshold 0.6 --match-margin 0.1 --new-group-threshold 0.8 `
  --save-plan "D:\Plans\sort.json"

# Edit the JSON plan: review destinations and set "approved": true for each move.
# For a review item with an extractable file, you can also set a safe destination.

# Preview approved moves, then apply them
python smart_sort.py "D:\Docs" --apply-plan "D:\Plans\sort.json"
python smart_sort.py "D:\Docs" --apply-plan "D:\Plans\sort.json" --apply
```

An existing-folder suggestion must meet the specified match threshold and, when
multiple folders are available, the margin over the next-best score. Unmatched
files can form new folders only when every pair meets the new-group threshold
and the group contains at least `--min-group-size` files (default two).
Ambiguous files, outliers, and files without usable embeddings remain for review.
New-group clustering is capped at 1,000 unmatched files; larger batches are left
for review rather than clustered.

Every decision is saved with `"approved": false`; `--organize --apply` is not
allowed. Only manually approved entries with content fingerprints can move.
Before applying, Smart-Sort checks that incoming files, sampled reference files,
category listings, and destinations still match the preview. If they changed,
create a new plan. The usual manifest and undo command are produced.

To read incoming files from a separate folder inside the root, use
`--source "Inbox"`; add `--recursive` to include its subfolders. The source's
top-level folder is not used as a reference category. Hidden, excluded, state,
and prior `Cluster_*` folders cannot be selected as sources. The legacy
clustering mode and its `--recursive`, `--k`, and `--auto-k` behavior remain
unchanged.

Grouping quality and the thresholds have **not** been benchmarked on a
representative held-out dataset. Review every proposed destination and keep a
backup of important files.

### Evaluate a held-out preview

`benchmark_sort.py` evaluates an **unapproved** preview against a local labels
file. Its sample gate requires at least 200 held-out files, ten existing
categories, and 40 unrelated or ambiguous files. It reports existing-folder
precision and recall and new-group pairwise precision and recall, overall and
by modality. The proposed targets are 95%/70% and 90%/50%, respectively.

The following abbreviated example illustrates the labels format; it is not a
large enough evaluation set:

```json
{
  "type": "smart-sort-labels",
  "version": 1,
  "dataset_id": "held-out-example",
  "split": "held-out",
  "calibration_case_ids": ["calibration-document-1"],
  "cases": [
    {"case_id": "held-1", "source": "invoice.txt", "kind": "text",
     "expected": "existing", "folder": "Invoices"},
    {"case_id": "held-2", "source": "meal.txt", "kind": "text",
     "expected": "new", "group": "Recipes"},
    {"case_id": "held-3", "source": "unknown.txt", "kind": "text",
     "expected": "unrelated"}
  ]
}
```

`source` values must match every decision in the saved plan. `kind` is `text`,
`image`, `audio`, or `video`; `expected` is `existing`, `new`, `unrelated`, or
`ambiguous`. A `new` label needs a group shared by at least two files to measure
pairwise recall. Case IDs are checked against the supplied calibration IDs, but
you must ensure they identify genuinely separate, representative files.

```powershell
python benchmark_sort.py --plan "D:\Plans\sort.json" --labels "D:\Plans\labels.json"
# Add --json for counts, per-modality results, and gate status.
```

The evaluator exits 0 when the sample and overall targets pass, 1 when they do
not, and 2 for invalid input. A passing synthetic or narrow dataset is not a
real-world guarantee and does not enable automatic application.

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
- `--device cuda` / `--device cpu` are auto-detected. Automatic dtype uses bf16
  only when the selected CUDA device reports native support; otherwise it uses fp32.
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
- GGUF models and media are content-hashed for cache identity. Multimodal caching
  is disabled for local models or remote models without a verifiable immutable
  revision, avoiding reuse of vectors from different model contents.

## How it works

| Step | What happens |
| --- | --- |
| 1. Collect | Files are listed deterministically; symlinks, Windows reparse points (including junctions), hidden files, `Cluster_*` folders, and common noise dirs (`.git`, `node_modules`, …) are skipped. |
| 2. Extract | Plain text is read from `.txt`/`.md`/code/CSV/…, `.pdf` (pypdfium2's C++ PDFium engine, with pypdf fallback), and `.docx` (python-docx). Unknown extensions are accepted only when the bytes look like text. PDF/DOCX parsing can use worker processes (`--jobs`), but the automatic mode stays sequential unless a batch is genuinely large enough to amortize worker startup. Images/audio/video are passed to the multimodal backend when enabled. Binary files that no backend can read are skipped individually instead of aborting the run. |
| 3. Embed | Text is split into up to 4 chunks of 4 000 chars — sampling the start *and* end of long documents — and chunks are mean-pooled and L2-normalized per document. Media files get one vector each. Embeddings are cached in `.smart_sort/`. |
| 4. Cluster | K-means with `n_init=10` and a fixed seed over all documents and media. `--k` sets the count, `--auto-k` chooses it by silhouette score, otherwise `sqrt(n/2)` is used (capped at 20 and at the file count). |
| 5. Name | Folder names get keywords from the TF-IDF terms that most distinguish each cluster (`Cluster_3_late_fee`). Media-only clusters get the nearest label from a built-in vocabulary (`receipts`, `screenshots`, `wallpapers`, …). `--no-labels` keeps plain `Cluster_N`. |
| 6. Move | Only with `--apply`. Planned collisions get ` (2)` suffixes. A move intent is flushed to a JSONL manifest before each no-overwrite move, then a completion record is appended. |

## Safety model

- **Dry run by default.** Files are only moved when `--apply` is passed.
- **Undo and recovery.** Every applied run creates a uniquely named manifest under
  `.smart_sort/` and prints its absolute path. `--undo <manifest>` previews the
  reverse plan; add `--apply` to restore files, including unambiguously completed
  moves interrupted before their completion record. SHA-256 checksums are checked
  first, and existing version-1 manifests remain supported.
- **Interrupted copies.** If both source and destination exist after a failed or
  interrupted copy, Smart-Sort reports the ambiguity and leaves both untouched.
  Inspect both files manually before deciding what to keep; do not assume the
  destination is a complete copy.
- **No overwrites.** Planned name clashes become `name (2).ext`. If a destination
  appears after planning, that move fails instead of replacing it.
- **One applied operation per root.** A process-held lock prevents concurrent
  applied runs or undos for the same root. The lock is released when the process
  exits, including after a crash. It does not prevent other programs from changing
  files or directories; avoid doing so while applying or undoing a run.
- **No link traversal.** Symlinks and Windows reparse points are excluded from
  discovery and rejected in the selected root and move paths. Paths are checked
  immediately before moves, but a separate process replacing a directory between
  the check and the filesystem operation can still race those checks. Hidden/system
  files (Windows attributes, `desktop.ini`, Office `~$` lock files; dot-prefixed
  files only on non-Windows) and the state folder are not re-ingested.
- Manifest records are flushed with `fsync`; recovery after power loss still depends
  on the filesystem and storage honoring flushes. Keep backups of important files.
- Files that cannot be read are reported and skipped; the rest of the run continues.

## Options

| Option | Description |
| --- | --- |
| `--apply` | Perform the moves (otherwise preview only). |
| `--undo MANIFEST` | Reverse a previous run (dry run unless `--apply`). |
| `--organize` | Preview loose files against existing folders and possible new groups; never moves files directly. |
| `--source FOLDER` | Incoming folder inside the root for `--organize`; use `--recursive` for its subfolders. |
| `--match-threshold N` | Explicit minimum existing-folder similarity for `--organize`. |
| `--match-margin N` | Explicit minimum gap between the best two existing-folder scores. Required with `--match-threshold`. |
| `--new-group-threshold N` | Explicit minimum pairwise similarity for proposed new groups. |
| `--min-group-size N` | Minimum size for a proposed new group (default 2). |
| `--save-plan PATH` | Save an organizing preview to a new JSON file for review. |
| `--apply-plan PATH` | Preview or, with `--apply`, execute individually approved decisions from a saved plan. |
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
- The organizing workflow's scores and thresholds have not been calibrated on
  representative held-out data. Automated suggestions are therefore preview-only.
- Each file joins exactly one cluster; mixed-topic documents get one averaged vector.
- Clustering is unsupervised, so cluster numbering is arbitrary between runs.
- `.smart_sort/` holds the cache and manifests — leave it in place to keep undo working.