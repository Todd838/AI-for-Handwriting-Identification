# AnyScript Writer Retrieval (GLM-OCR + Unsloth + Triplet Loss)

This project provides an end-to-end training and retrieval pipeline for the AnyScript challenge:

- Triplet dataset sampling from `author/book/page` structure
- Vision backbone loading with Unsloth (`FastVisionModel`) when available
- Writer embedding head + triplet loss training
- Optional LoRA adaptation support
- FAISS index building and retrieval evaluation

`ANYSCRIPT_FILTERED_ARCHIVE` env overrides the default archive path in `scripts/data_anyscript.py` (useful on other machines / Colab).

## DeepSeek-OCR 2 (Unsloth + Transformers)

- **Config / prompts / recommended decoding:** `scripts/deepseek_ocr2_config.py` (`RECOMMENDED_DECODING`, prompt presets, infer `base_size` / `image_size` presets).
- **Load + `infer` wrapper:** `scripts/deepseek_ocr2.py` (`snapshot_download`, Unsloth `FastVisionModel` with `auto_model=AutoModel`, optional Transformers fallback, `run_deepseek_infer`).
- **OCR CLI:** `scripts/deepseek_ocr2_infer.py` — run `model.infer(tokenizer, ...)` on an image (see `--help`).
- **Writer retrieval (triplet / FAISS / submission):** pass `--backbone deepseek_ocr2` and `--model_name` (HF id or local path), or rely on **auto** when `model_name` contains `DeepSeek` and `OCR`. Optional: `--deepseek_download --deepseek_local_dir ./weights` to snapshot `unsloth/DeepSeek-OCR-2`. Training still uses **pooled vision features** + triplet loss (not OCR text loss). If feature extraction fails on your checkpoint revision, use GLM-OCR or extend `extract_pooled_features` for that architecture.

## Expected dataset layout

```text
anyscript/
  author_0001/
    book_01/
      page_001.jpg
      page_002.jpg
  author_0002/
    ...
```

## Quick start (Colab)

**Opening from GitHub in Colab:** the file browser only lists **`.ipynb`** files in the repo. Use path `colab_quickstart.ipynb`, or ignore that UI and run `git clone https://github.com/Todd838/AI-for-Handwriting-Identification.git` in a code cell (see notebook). The **Actions** tab on GitHub is unrelated to Colab.

If training reports **0 pages** under `--data_root`, images may live deeper (e.g. `MyDrive/data/...` after `tar` extract). Run `python scripts/inspect_anyscript_layout.py /content/drive/MyDrive/AnyScriptFiltered` (see Colab notebook cell) and set `DATA_ROOT` to the suggested path.

```bash
pip install -r requirements.txt
python scripts/train_triplet_unsloth.py \
  --data_root /content/drive/MyDrive/anyscript \
  --model_name THUDM/glm-ocr \
  --output_dir /content/drive/MyDrive/anyscript_runs/run1
```

Then build an index (images are read **in batches from disk**, so huge galleries do not load all pixels into RAM at once; reduce `--batch_size` if the GPU runs out of memory):

```bash
python scripts/build_faiss_index.py \
  --data_root /content/drive/MyDrive/anyscript \
  --checkpoint /content/drive/MyDrive/anyscript_runs/run1/best.pt \
  --index_out /content/drive/MyDrive/anyscript_runs/run1/faiss.index \
  --meta_out /content/drive/MyDrive/anyscript_runs/run1/meta.npy
```

`export_embeddings_split.py` and checkpoint mode in `export_anyscript_submission.py` use the same streaming embed path.

Evaluate retrieval:

```bash
python scripts/eval_retrieval.py \
  --index_path /content/drive/MyDrive/anyscript_runs/run1/faiss.index \
  --meta_path /content/drive/MyDrive/anyscript_runs/run1/meta.npy \
  --query_embeddings /content/drive/MyDrive/anyscript_runs/run1/query_embs.npy \
  --query_meta /content/drive/MyDrive/anyscript_runs/run1/query_meta.npy
```

## ICDAR 2026 AnyScript — platform submission (CSV)

The competition expects a **dense similarity table**: for **each** official query document you must emit one CSV row per **training** gallery item (not only top‑K), with your model’s similarity score.

| Track | Queries | Gallery | Row meaning |
| --- | --- | --- | --- |
| **Intra-book** | Held-out query **pages** | All training **pages** | Page-to-page scores |
| **Extra-book** | Held-out query **books** | All training **books** | Book-to-book scores |

**Columns (exact names, in order):** `query_document_id`, `retrieved_document_id`, `similarity_score`.

**IDs:** Use the **exact** query and training identifiers from the challenge README / dataset package (e.g. `query_page_001`, `train_page_1234` — illustrations only; your release may differ). Your retrieval or embedding code must map filesystem paths or internal keys to those official strings.

**Helper in-repo:** `scripts/data_anyscript.py` holds paths, records, submission CSV helpers, and id resolution **without importing PyTorch** (so `python export_anyscript_submission.py --help` works in a numpy-only env). Training/embedding use `scripts/data_anyscript_vision.py` (`TripletPageDataset`, `default_transform`).

**Id map templates:** generate JSON keys that match `page_relative_key` / `book_key`, then replace values with ids from the challenge package:

```bash
cd scripts
python make_id_map_template.py --data_root /path/to/train_root --granularity page --out_json train_ids_template.json
python make_id_map_template.py --data_root /path/to/query_root --granularity page --out_json query_ids_template.json
```

### Export script (`scripts/export_anyscript_submission.py`)

Produces the **dense** CSV (every query × every gallery item) with cosine similarity scores.

**From checkpoint** (separate query and training trees on disk):

```bash
cd scripts
python export_anyscript_submission.py --mode intra_book \
  --checkpoint /path/to/best.pt \
  --gallery_data_root /path/to/train_root \
  --query_data_root /path/to/query_pages_root \
  --query_ids_json /path/to/query_ids.json \
  --gallery_ids_json /path/to/train_ids.json \
  --out_csv /path/to/submission_intra.csv
```

`--mode extra_book` aggregates **L2-normalized mean** page embeddings per `(author_id, book_id)` before scoring.

**From precomputed embeddings** (e.g. after editing `export_embeddings_split` to use the full training set):

```bash
python export_anyscript_submission.py --mode intra_book \
  --embeddings_dir /path/to/embs_folder \
  --gallery_key_root /path/to/train_root \
  --query_key_root /path/to/query_root \
  --query_ids_json /path/to/query_ids.json \
  --gallery_ids_json /path/to/train_ids.json \
  --out_csv submission.csv
```

`--gallery_key_root` / `--query_key_root` must match the roots used so `page_relative_key(root, page_path)` equals the keys in your id JSON (paths use forward slashes).

**ID JSON formats** (either file can use either shape):

1. **List** — length must match embedding row order (query order = `query_meta` rows; gallery = `gallery_meta` rows; for extra-book, books sorted by `author_id/book_id`). With `export_embeddings_split.py`, records are **sorted** before shuffling and the default **`--shuffle_seed 42`** makes that order reproducible across runs; use **`--random_shuffle`** only if you accept non-reproducible splits (then prefer dict id JSON).
2. **Object** — maps each key to the platform id, e.g. `"some_author/some_book/page_01.jpg": "query_page_001"` or `"some_author/some_book": "train_book_042"`.

For dry runs without official ids: add `--allow_synthetic_ids` (emits placeholders like `query_page_000000`, `train_page_000001`).

## Notes

- `--model_name` can be changed to the newest GLM OCR checkpoint.
- If Unsloth fast vision loader does not support the specific checkpoint revision, the code falls back to `transformers`.
- This code is retrieval-focused; OCR generation is not used during training.
