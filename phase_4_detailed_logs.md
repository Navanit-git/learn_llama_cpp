# Phase 4 Detailed Logs: GGUF Quantization Variant Exploration

Date started: 2026-02-25

This document tracks Phase 4 execution: identifying available quantized GGUF variants, selecting practical candidates for this machine, and benchmarking quality/speed trade-offs.

## Status: 🔄 In Progress

Current checkpoint: model variant + size discovery completed for 4 repos:
- `unsloth/Qwen3.5-27B-GGUF`
- `unsloth/Ministral-3-14B-Instruct-2512-GGUF`
- `unsloth/Qwen3-14B-GGUF`
- `unsloth/Qwen3.5-0.8B-GGUF`

---

## Goal of this first Phase 4 step

Avoid downloading full model repositories and instead list available `.gguf` files + their sizes first, then pick targeted variants.

---

## Methods used

### Method 1 — Hugging Face Hub API (tree listing)

Used Hugging Face model tree API to enumerate files in the repo without downloading weights.

Command run from workspace root:

```bash
curl -s https://huggingface.co/api/models/unsloth/Qwen3.5-27B-GGUF/tree/main | grep -oP '"path":"\K[^"]+\.gguf'
```

Why this method:
- Calls metadata API only (file index), not model blobs
- Fast and bandwidth-light
- Ideal when repos are very large

### Method 2 — Structured parse for names + sizes (all requested repos)

Used `python3` JSON parsing against the HF API with `?recursive=1` to retrieve complete `.gguf` file paths and exact byte sizes.

Command run from workspace root:

```bash
python3 - <<'PY'
import json, urllib.request
repos = [
	'unsloth/Qwen3.5-27B-GGUF',
	'unsloth/Ministral-3-14B-Instruct-2512-GGUF',
	'unsloth/Qwen3-14B-GGUF',
]
for repo in repos:
	url = f'https://huggingface.co/api/models/{repo}/tree/main?recursive=1'
	data = json.load(urllib.request.urlopen(url))
	rows = [
		(x['path'], int(x.get('size', 0)))
		for x in data
		if x.get('type') == 'file' and x.get('path', '').endswith('.gguf')
	]
	for name, size in sorted(rows):
		print(repo, name, size)
PY
```

Why this method:
- Gives exact size in bytes directly from API metadata
- Handles nested paths (for example BF16 shards in subfolders)
- Produces a repeatable, machine-readable output for logging

---

## Results — `.gguf` names and sizes

### Repo: `unsloth/Qwen3.5-27B-GGUF` (20 `.gguf` files)

| Variant | Size (bytes) | Size (GiB) |
|---|---:|---:|
| `mmproj-BF16.gguf` | 931146176 | 0.87 |
| `mmproj-F16.gguf` | 927607232 | 0.86 |
| `mmproj-F32.gguf` | 1842940352 | 1.72 |
| `Qwen3.5-27B-Q2_K.gguf` | 10493060480 | 9.77 |
| `Qwen3.5-27B-Q3_K_M.gguf` | 13505115520 | 12.58 |
| `Qwen3.5-27B-Q3_K_S.gguf` | 12289422720 | 11.45 |
| `Qwen3.5-27B-Q4_0.gguf` | 15677408640 | 14.60 |
| `Qwen3.5-27B-Q4_1.gguf` | 17182934400 | 16.00 |
| `Qwen3.5-27B-Q4_K_M.gguf` | 16740812160 | 15.59 |
| `Qwen3.5-27B-Q4_K_S.gguf` | 15769159040 | 14.69 |
| `Qwen3.5-27B-Q5_K_M.gguf` | 19608995200 | 18.26 |
| `Qwen3.5-27B-Q5_K_S.gguf` | 18889000320 | 17.59 |
| `Qwen3.5-27B-Q6_K.gguf` | 22453933440 | 20.91 |
| `Qwen3.5-27B-Q8_0.gguf` | 28595762560 | 26.63 |
| `Qwen3.5-27B-UD-Q4_K_XL.gguf` | 16729015680 | 15.58 |
| `Qwen3.5-27B-UD-Q5_K_XL.gguf` | 19600147840 | 18.25 |
| `Qwen3.5-27B-UD-Q6_K_XL.gguf` | 23064053120 | 21.48 |
| `Qwen3.5-27B-UD-Q8_K_XL.gguf` | 32395212160 | 30.17 |

### Repo: `unsloth/Ministral-3-14B-Instruct-2512-GGUF` (29 `.gguf` files)

| Variant | Size (bytes) | Size (GiB) |
|---|---:|---:|
| `Ministral-3-14B-Instruct-2512-BF16.gguf` | 27020865952 | 25.17 |
| `Ministral-3-14B-Instruct-2512-IQ4_NL.gguf` | 7805711040 | 7.27 |
| `Ministral-3-14B-Instruct-2512-IQ4_XS.gguf` | 7432155840 | 6.92 |
| `Ministral-3-14B-Instruct-2512-Q2_K.gguf` | 5246530240 | 4.89 |
| `Ministral-3-14B-Instruct-2512-Q2_K_L.gguf` | 5403816640 | 5.03 |
| `Ministral-3-14B-Instruct-2512-Q3_K_M.gguf` | 6682096320 | 6.22 |
| `Ministral-3-14B-Instruct-2512-Q3_K_S.gguf` | 6074905280 | 5.66 |
| `Ministral-3-14B-Instruct-2512-Q4_0.gguf` | 7805711040 | 7.27 |
| `Ministral-3-14B-Instruct-2512-Q4_1.gguf` | 8581657280 | 7.99 |
| `Ministral-3-14B-Instruct-2512-Q4_K_M.gguf` | 8239067840 | 7.67 |
| `Ministral-3-14B-Instruct-2512-Q4_K_S.gguf` | 7834546880 | 7.30 |
| `Ministral-3-14B-Instruct-2512-Q5_K_M.gguf` | 9620566720 | 8.96 |
| `Ministral-3-14B-Instruct-2512-Q5_K_S.gguf` | 9383817920 | 8.74 |
| `Ministral-3-14B-Instruct-2512-Q6_K.gguf` | 11088409280 | 10.33 |
| `Ministral-3-14B-Instruct-2512-Q8_0.gguf` | 14359311040 | 13.37 |
| `Ministral-3-14B-Instruct-2512-UD-IQ1_M.gguf` | 3671126720 | 3.42 |
| `Ministral-3-14B-Instruct-2512-UD-IQ1_S.gguf` | 3441750720 | 3.21 |
| `Ministral-3-14B-Instruct-2512-UD-IQ2_M.gguf` | 4912132800 | 4.57 |
| `Ministral-3-14B-Instruct-2512-UD-IQ2_XXS.gguf` | 4057379520 | 3.78 |
| `Ministral-3-14B-Instruct-2512-UD-IQ3_XXS.gguf` | 5493437120 | 5.12 |
| `Ministral-3-14B-Instruct-2512-UD-Q2_K_XL.gguf` | 5527106240 | 5.15 |
| `Ministral-3-14B-Instruct-2512-UD-Q3_K_XL.gguf` | 6931092160 | 6.46 |
| `Ministral-3-14B-Instruct-2512-UD-Q4_K_XL.gguf` | 8366207680 | 7.79 |
| `Ministral-3-14B-Instruct-2512-UD-Q5_K_XL.gguf` | 9640104640 | 8.98 |
| `Ministral-3-14B-Instruct-2512-UD-Q6_K_XL.gguf` | 12124533440 | 11.29 |
| `Ministral-3-14B-Instruct-2512-UD-Q8_K_XL.gguf` | 17116738240 | 15.94 |
| `mmproj-BF16.gguf` | 879257760 | 0.82 |
| `mmproj-F16.gguf` | 878053536 | 0.82 |
| `mmproj-F32.gguf` | 1755867296 | 1.64 |

### Repo: `unsloth/Qwen3-14B-GGUF` (26 `.gguf` files)

| Variant | Size (bytes) | Size (GiB) |
|---|---:|---:|
| `Qwen3-14B-BF16.gguf` | 29543424160 | 27.51 |
| `Qwen3-14B-IQ4_NL.gguf` | 8541363584 | 7.95 |
| `Qwen3-14B-IQ4_XS.gguf` | 8135040384 | 7.58 |
| `Qwen3-14B-Q2_K.gguf` | 5753984384 | 5.36 |
| `Qwen3-14B-Q2_K_L.gguf` | 5936307584 | 5.53 |
| `Qwen3-14B-Q3_K_M.gguf` | 7321313664 | 6.82 |
| `Qwen3-14B-Q3_K_S.gguf` | 6657106304 | 6.20 |
| `Qwen3-14B-Q4_0.gguf` | 8543001984 | 7.96 |
| `Qwen3-14B-Q4_1.gguf` | 9389522304 | 8.74 |
| `Qwen3-14B-Q4_K_M.gguf` | 9001753984 | 8.38 |
| `Qwen3-14B-Q4_K_S.gguf` | 8573476224 | 7.98 |
| `Qwen3-14B-Q5_K_M.gguf` | 10514570624 | 9.79 |
| `Qwen3-14B-Q5_K_S.gguf` | 10263895424 | 9.56 |
| `Qwen3-14B-Q6_K.gguf` | 12121938304 | 11.29 |
| `Qwen3-14B-Q8_0.gguf` | 15698534784 | 14.62 |
| `Qwen3-14B-UD-IQ1_M.gguf` | 4064814464 | 3.79 |
| `Qwen3-14B-UD-IQ1_S.gguf` | 3827369344 | 3.56 |
| `Qwen3-14B-UD-IQ2_M.gguf` | 5425956224 | 5.05 |
| `Qwen3-14B-UD-IQ2_XXS.gguf` | 4470113664 | 4.16 |
| `Qwen3-14B-UD-IQ3_XXS.gguf` | 6013732224 | 5.60 |
| `Qwen3-14B-UD-Q2_K_XL.gguf` | 6067584384 | 5.65 |
| `Qwen3-14B-UD-Q3_K_XL.gguf` | 7593917824 | 7.07 |
| `Qwen3-14B-UD-Q4_K_XL.gguf` | 9159818624 | 8.53 |
| `Qwen3-14B-UD-Q5_K_XL.gguf` | 10546437504 | 9.82 |
| `Qwen3-14B-UD-Q6_K_XL.gguf` | 13285990784 | 12.37 |
| `Qwen3-14B-UD-Q8_K_XL.gguf` | 18754560384 | 17.47 |

### Repo: `unsloth/Qwen3.5-0.8B-GGUF` (25 `.gguf` files)

| Variant | Size (bytes) | Size (GiB) |
|---|---:|---:|
| `Qwen3.5-0.8B-BF16.gguf` | 1516744736 | 1.41 |
| `Qwen3.5-0.8B-IQ4_NL.gguf` | 506859776 | 0.47 |
| `Qwen3.5-0.8B-IQ4_XS.gguf` | 492605696 | 0.46 |
| `Qwen3.5-0.8B-Q3_K_M.gguf` | 470167808 | 0.44 |
| `Qwen3.5-0.8B-Q3_K_S.gguf` | 440750336 | 0.41 |
| `Qwen3.5-0.8B-Q4_0.gguf` | 507154688 | 0.47 |
| `Qwen3.5-0.8B-Q4_1.gguf` | 535171328 | 0.50 |
| `Qwen3.5-0.8B-Q4_K_M.gguf` | 532517120 | 0.50 |
| `Qwen3.5-0.8B-Q4_K_S.gguf` | 508104960 | 0.47 |
| `Qwen3.5-0.8B-Q5_K_M.gguf` | 590057728 | 0.55 |
| `Qwen3.5-0.8B-Q5_K_S.gguf` | 568889600 | 0.53 |
| `Qwen3.5-0.8B-Q6_K.gguf` | 639029504 | 0.60 |
| `Qwen3.5-0.8B-Q8_0.gguf` | 811843840 | 0.76 |
| `Qwen3.5-0.8B-UD-IQ2_M.gguf` | 371933440 | 0.35 |
| `Qwen3.5-0.8B-UD-IQ2_XXS.gguf` | 338227456 | 0.31 |
| `Qwen3.5-0.8B-UD-IQ3_XXS.gguf` | 398237952 | 0.37 |
| `Qwen3.5-0.8B-UD-Q2_K_XL.gguf` | 417718528 | 0.39 |
| `Qwen3.5-0.8B-UD-Q3_K_XL.gguf` | 492216576 | 0.46 |
| `Qwen3.5-0.8B-UD-Q4_K_XL.gguf` | 558772480 | 0.52 |
| `Qwen3.5-0.8B-UD-Q5_K_XL.gguf` | 606585088 | 0.56 |
| `Qwen3.5-0.8B-UD-Q6_K_XL.gguf` | 771092736 | 0.72 |
| `Qwen3.5-0.8B-UD-Q8_K_XL.gguf` | 1186443520 | 1.10 |
| `mmproj-BF16.gguf` | 207346528 | 0.19 |
| `mmproj-F16.gguf` | 204987232 | 0.19 |
| `mmproj-F32.gguf` | 402381664 | 0.37 |

---

## Observations for this machine (15 GB RAM / 4 GB VRAM)

- 27B quants are heavy for this system; practical first tests are `Q2_K` and `Q3_K_*`.
- 14B repos are much more realistic for repeated benchmarking on this hardware.
- For 14B, `Q4_K_M`/`Q5_K_M` are likely the quality/speed sweet spot to evaluate first.
- `Q8_0` variants are likely to be RAM-assisted and slower, but useful as quality anchors.

---

## Recommended next actions (Phase 4 continuation)

1. Pick one 14B repo as the benchmark baseline (`Qwen3-14B` or `Ministral-3-14B`).
2. Benchmark 3 tiers first: `Q3_K_M`, `Q4_K_M`, `Q5_K_M`.
3. Optionally benchmark `Q2_K` (speed floor) and `Q8_0` (quality ceiling).
4. Pull only one file at a time via `--hf-repo` + `--hf-file`.
5. Record RAM/VRAM + prefill/generation t/s in the benchmark table.

---

## Phase 4 checklist (initial)

- [x] Locate multi-quant GGUF repos on Hugging Face
- [x] Enumerate model file names without full download
- [x] Capture GGUF variant sizes for all selected repos
- [ ] Select first benchmark candidate variant
- [ ] Run first benchmark
- [ ] Log first benchmark results

---

## Automation script for Ministral (Q4 -> Q8)

Created script:

`./phase4_ministral_q4_to_q8_cycle.sh`

What it does per variant (`Q4_0`, `Q4_1`, `Q4_K_M`, `Q4_K_S`, `Q5_K_M`, `Q5_K_S`, `Q6_K`, `Q8_0`):
1. Downloads one model file from HF (`--hf-repo` + `--hf-file`)
2. Runs `llama-bench` speed tests (prefill + generation)
3. Starts `llama-server` for a smoke check and writes a dedicated server log for that variant
4. Appends results to a summary CSV
5. Deletes that model from cache (unless `KEEP_MODELS=1`)

### Run command

```bash
./phase4_ministral_q4_to_q8_cycle.sh
```

### Useful overrides

```bash
# Keep downloaded models after each test
KEEP_MODELS=1 ./phase4_ministral_q4_to_q8_cycle.sh

# Change benchmark settings
THREADS=10 PROMPT_TOKENS=512 GEN_TOKENS=128 REPETITIONS=3 ./phase4_ministral_q4_to_q8_cycle.sh

# Custom output location
OUTPUT_ROOT=/home/nav_wsl/code/learn_llama_cpp/archieve/phase4/ministral_q4_to_q8 ./phase4_ministral_q4_to_q8_cycle.sh
```

### Output files

- Summary table (CSV):
	- `archieve/phase4/ministral_q4_to_q8/results/summary_<timestamp>.csv`
- Run log:
	- `archieve/phase4/ministral_q4_to_q8/results/run_<timestamp>.log`
- Benchmark logs:
	- `archieve/phase4/ministral_q4_to_q8/bench_logs/<variant>.log`
	- `archieve/phase4/ministral_q4_to_q8/bench_logs/<variant>.json`
- `llama-server` logs (different log per GGUF format):
	- `archieve/phase4/ministral_q4_to_q8/server_logs/<variant>.log`


