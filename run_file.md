# Run + Edit Guide (SH Files)

Use this quick guide to edit and run the main scripts.

## 1) One script for everything

Main file: `model_one_shot.sh`

Edit these 2 lines for your model:
- `MODEL_REPO`
- `MODEL_FILE`

Run:

```bash
cd /home/nav_wsl/code/learn_llama_cpp
./model_one_shot.sh
```

Optional action modes:

```bash
ACTION=download ./model_one_shot.sh
ACTION=check ./model_one_shot.sh
ACTION=smoke ./model_one_shot.sh
```

## 2) Regression only (generic)

Main file: `model_regression_benchmark.sh`

Run with explicit model:

```bash
MODEL_REPO="unsloth/Qwen3.5-0.8B-GGUF" \
MODEL_FILE="Qwen3.5-0.8B-Q4_K_M.gguf" \
MODE=smoke MAX_CASES=20 \
./model_regression_benchmark.sh
```

Common modes:
- `MODE=smoke` (quick)
- `MODE=full` (default)
- `MODE=exhaustive` (long run)

## 3) How to edit any `.sh` quickly

Use VS Code:

```bash
code model_one_shot.sh
```

Or terminal editor:

```bash
nano model_one_shot.sh
```

After editing, validate syntax:

```bash
bash -n model_one_shot.sh
```

If script is new or not executable:

```bash
chmod +x model_one_shot.sh
```

## 4) Legacy scripts

Older model-specific helper scripts were moved to:

- `archieve/legacy_scripts/`



# start
./qwen35_server.sh > qwenterminal_v1.log 2>&1 &
echo $! > qwen35.pid

# …leave terminal/close it …

# later, reopen or in another terminal:
kill $(<qwen35.pid)

# review what happened
grep -i \"tokens per second\" qwenterminal_v1.log
less qwenterminal_v1.log

# wait for server to come up
until curl -s http://127.0.0.1:8080/health | grep -q '"status":"ok"'; do sleep 1; done && echo "Server ready"

# send a chat completion
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen35",
    "messages": [{"role": "user", "content": "explain transformer in details atleast 4 paragraph "}],
    "max_tokens": 64
  }' | python3 -m json.tool

  curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen35","messages":[{"role":"user","content":"ping"}],"max_tokens":16}' \
  | python3 -m json.tool