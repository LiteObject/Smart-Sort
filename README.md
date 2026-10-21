# Smart-Sort

Organize a folder by **what its files are about**, not by name, date, or extension.

Smart-Sort reads each file's text, turns it into an embedding vector, clusters the
vectors, and moves related files into `Cluster_N_<keywords>` folders. Everything runs
locally — no API keys, no uploads.

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
Use `--backend tfidf` or `--backend gguf` to force a choice.

## How it works

| Step | What happens |
| --- | --- |
| 1. Collect | Files are listed deterministically; symlinks, hidden files, `Cluster_*` folders, and common noise dirs (`.git`, `node_modules`, …) are skipped. |
| 2. Extract | Plain text is read from `.txt`/`.md`/code/CSV/…, `.pdf` (pypdf), and `.docx` (python-docx). Unknown extensions are accepted only when the bytes look like text. PDF and DOCX parsing is dispatched to worker processes (`--jobs`) since it is CPU-bound. Binary files are skipped individually instead of aborting the run. |
| 3. Embed | Content is split into up to 4 chunks of 4 000 chars — sampling the start *and* end of long documents — and each chunk is embedded. Chunks are mean-pooled and L2-normalized per document. Embeddings of unchanged chunks are cached in `.smart_sort/`. |
| 4. Cluster | K-means with `n_init=10` and a fixed seed. `--k` sets the count, `--auto-k` chooses it by silhouette score, otherwise `sqrt(n/2)` is used (capped at 20 and at the file count). |
| 5. Name | Folder names get keywords from the TF-IDF terms that most distinguish each cluster (`Cluster_3_late_fee`), with a numeric fallback. `--no-labels` keeps plain `Cluster_N`. |
| 6. Move | Only with `--apply`. Collisions get ` (2)` suffixes; nothing is overwritten. Each successful move is appended to a JSONL manifest immediately, so a crash still leaves a valid undo record. |

## Safety model

- **Dry run by default.** Files are only moved when `--apply` is passed.
- **Undo.** Every applied run writes `.smart_sort/manifest-<timestamp>.jsonl`.
  `--undo <manifest>` prints the reverse plan; add `--apply` to execute it. SHA-256
  checksums are verified before a file is moved back, and emptied cluster folders are removed.
- **No overwrites.** Destination name clashes become `name (2).ext`.
- **No symlink traversal**, no hidden files, no state folder re-ingestion.
- Files that cannot be read are reported and skipped; the rest of the run continues.

## Options

| Option | Description |
| --- | --- |
| `--apply` | Perform the moves (otherwise preview only). |
| `--undo MANIFEST` | Reverse a previous run (dry run unless `--apply`). |
| `--k N` | Number of clusters. |
| `--auto-k` | Pick `k` by silhouette score. |
| `-r`, `--recursive` | Include files in subfolders. |
| `--backend {auto,gguf,tfidf}` | Embedding backend. |
| `--model PATH` | GGUF embedding model. |
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

## Limitations

- Text-based only: scanned/image-only PDFs yield nothing without OCR.
- TF-IDF fallback groups by shared vocabulary, not meaning; use a GGUF model for quality.
- Each file joins exactly one cluster; mixed-topic documents get one averaged vector.
- Clustering is unsupervised, so cluster numbering is arbitrary between runs; naming
  keywords make folders recognizable but are not curated labels.
- `.smart_sort/` holds the cache and manifests — leave it in place to keep undo working.