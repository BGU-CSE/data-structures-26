# Final Project: Semantic Movie Search
### Building a Vector Database with Locality-Sensitive Hashing (LSH)

> **Course:** Data Structures · BGU CS · 2026  
---

## About the Project

*"Can you propose a movie that is funny, happens in space, but also a bit philosophical?"*

How do we turn free text into a movie recommendation? This project asks you to build the data structure that makes it possible.

You will implement a **Locality-Sensitive Hashing (LSH) Index** — a modern Vector Database, the essential data structure powering cutting-edge AI and semantic search systems. Traditional data structures fail when handling the high-dimensional data used by neural networks; LSH provides an algorithmic solution to the *curse of dimensionality*, allowing us to efficiently find similar vectors without a slow brute-force scan.

**What your program does:**
1. Accepts a free-text search query (a description of the movie you want)
2. Returns the *k* closest movies from a database of ~5,000 films — instantly

Behind the scenes, the search is purely geometric. Your query is transformed into a **384-dimensional vector** (embedding) by a pre-trained AI model. Each movie in the database is already stored as a similar 384-dimensional vector. The goal is to find the *k* nearest vectors to your query, measured by **cosine similarity**.

### How LSH Works

LSH uses **random hyperplanes** (a flat boundary through the origin — a line in 2D, a flat plane in 3D, a 383-dimensional subspace in our 384-dimensional space) to partition the high-dimensional space. Each vector is converted into a short binary **hash key** by checking which side of each hyperplane it falls on:

```
bit_i = 1  if  dot(vector, plane_i) >= 0  else  0
key   = sum(bit_i * 2**i  for i in range(num_bits))
```

Similar vectors tend to land in the same bucket, which dramatically reduces the number of comparisons needed.

A single table is often not enough — a hyperplane may fall directly between two similar movies. To compensate, every movie is inserted into **all L tables**. Each table has its own independently-generated hyperplanes, so the same movie gets a different key in each table. At query time, buckets from all L tables are **unioned** into one candidate pool, then re-ranked by precise cosine similarity.

📺 **[Watch this video for intuition on LSH](https://youtu.be/-nMatLpm_0A?si=PWl9JTs8bjgO7or0)**

---

## What You Need to Implement

| Component | Description |
|---|---|
| `cosine_similarity(a, b)` | Measures closeness between two vectors (range −1 to 1) |
| `LSHIndex` | Splits the vector space into buckets using random hyperplanes |
| `search(...)` | Encodes a query, looks up candidates in the index, returns top-k results |

You may build any interface on top (CLI, GUI, web) — that part is up to you.

### AI Policy
- ✅ AI tools are permitted for boilerplate code (file parsing, UI)
- ❌ The LSH index, cosine similarity function, and search logic must be written **from scratch** (no external libraries that implement them)
- ✅ Guidance and feedback from AI is OK

---

## Provided Files

| File | Contents |
|---|---|
| `settings.json` | All config: model name, LSH parameters (`num_tables`, `num_bits`, `seed`), top-k |
| `tmdb_data.json` | Movie metadata: title, overview, genres, release date, etc. |
| `tmdb_vectors.npy` | `np.ndarray` of shape `(n, 384)`, float32, unit-normalized — one row per movie |
| `search_template.py` | Starter template — copy and rename to `search.py`, then fill in the TODOs |
| `test_local.py` | Pre-submission checker — run locally before uploading to VPL |
| `requirements.txt` | Exact library versions compatible with `test_local.py` |

> **Important:** Open and read these files before and while reading the instructions.

---

## Your Submission: `search.py`

Create a file called `search.py`. The grader imports it as a module and calls your functions directly — no setup is needed. Use **Python 3.12**.

### LSH Parameters

Your `search.py` must read parameters from `settings.json`:

```
num_tables = 20    num_bits = 8    hyperplane_seed = 42
```

### Functions to Implement

| Name | Parameters | Returns |
|---|---|---|
| `cosine_similarity(a, b)` | Two 1-D float32 numpy arrays | `float` in [−1, 1]; `0.0` if either is all-zeros |
| `LSHIndex(vectors, num_tables, num_bits, **kwargs)` | Movie vectors array + LSH params | `LSHIndex` object with `self.planes` and `self.tables` |
| `search(query, index, encoder, movies, k=5)` | Query string + built index + encoder + movie list | List of k movie dicts sorted by score, each with `"title"` and `"score"` |

### LSHIndex Attributes

| Attribute | Type | Description |
|---|---|---|
| `self.planes` | `np.ndarray` float32, shape `(num_tables, num_bits, d)` | Random hyperplane normals for all tables |
| `self.tables` | `list` of `num_tables` dicts | `{int key → [idx, idx, ...]}` — bucket contents per table |

### VPL Upload Rules

1. Only upload `search.py`
2. Do **not** upload `settings.json`, `tmdb_data.json`, `tmdb_vectors.npy`, `vpl_test.py`, or `search_template.py`
3. `search.py` must be importable without side effects (no printing, file loading, or model loading at module level)

### Available Libraries on VPL

```
# Safe to import
numpy, sentence_transformers, json, heapq, math, collections, itertools

# Will crash your submission — do not import
sklearn, scipy, torch, tensorflow, keras, faiss, pandas
```

### Suppressing Console Warnings

Paste this at the very top of `search.py` to silence HuggingFace noise:

```python
import os, warnings
from transformers import logging as hf_logging
os.environ['TOKENIZERS_PARALLELISM']          = 'false'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
warnings.filterwarnings('ignore')
hf_logging.set_verbosity_error()
```

---

## Testing Locally

Before submitting to VPL, run the provided checker:
**use python 3.12**
```bash
pip install -r requirements.txt
python test_local.py
```

| Section | What it checks | Data files needed |
|---|---|---|
| Section 1 | `cosine_similarity` | None |
| Section 2 | `LSHIndex` structure | None |
| Section 3 | Full search recall | `tmdb_data.json` + `tmdb_vectors.npy` |

Sections are graded **independently** — a crash in one section does not affect the others.

Make sure both data files exist at the paths defined in `settings.json`, or update `settings.json` to match your local folder layout.

---

Good luck! 🎬
