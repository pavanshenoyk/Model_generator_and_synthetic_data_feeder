# Model Generator

Train small decoder-only transformer models from scratch on **synthetic data written by a larger LLM**.

You describe the data you want and give a few samples of its format. A teacher model generates thousands of examples, and a small transformer (the student) trains on them. Training repeats on fresh data until the student passes a held-out test set, and every trained model is saved as a versioned **agent**.

The model code lives in [TransformerAgentModels](https://github.com/pavanshenoyk/TransformerAgentModels), a decoder-only transformer written from scratch in PyTorch.

## Two versions: pick your teacher

The training logic is identical. Only the teacher differs.

| | **API version** | **Local version** |
|---|---|---|
| Script | `model_generator_api.py` | `model_generator_local.py` |
| Teacher | OpenAI API | Local LLM served by Ollama |
| Cost | Pay per token, with a built-in spend cap | Free (your own hardware) |
| Needs | API key | Ollama + GPU with enough VRAM |
| Speed | Fast | Slower, depends on your GPU |
| Sample file | Uploaded once and reused | Pasted into every prompt, so keep it short |
| Extra safeguards | Spend cap | Server check at startup, teacher unloaded from VRAM before training |

Use **API** for quality and speed, **Local** for zero cost and full privacy.

## How it works

```
 your prompt + sample file
            |
            v
   Teacher LLM [API: OpenAI | Local: Ollama]
   --> batches of examples, separated by ===
            |
            v
   Student transformer trains on a fresh batch each cycle
            |
            v
   Test on a held-out, teacher-generated set
            |
            v
   Passed N times in a row?  --yes--> save final agent, done
            |no
            v
   Next cycle (new training data, hourly checkpoints saved)
```

1. **Test set:** generated once up front. It is regenerated only after you pass it.
2. **Each cycle:** a fresh batch of training examples is generated, the student trains on it, then it is scored on the test set.
3. **Score:** next-token accuracy (%) on the test set.
4. **Stop condition:** accuracy stays at or above the threshold for `required_successes` cycles in a row.
5. **Truncation handling:** if the teacher's output is cut off, the last (incomplete) example is dropped.
6. **Checkpoints:** a versioned agent is saved every `checkpoint_hours`, and you are asked whether to continue every `halt_hours`.
7. **[API] Spend cap:** running cost is tracked after every call, and the run stops when `spend_cap_usd` is reached.
8. **[Local] GPU sharing:** the teacher is unloaded from VRAM before every training phase, so the student gets the full GPU.

## Requirements

**Both versions**
- Python 3.10+
- A GPU is strongly recommended for training (CPU works but is slow)

**[API] only**
- An OpenAI API key with access to the model set in `gpt_config["model"]`

**[Local] only**
- [Ollama](https://ollama.com/download) installed and running (a recent version, since the script turns off the model's reasoning mode)
- Enough VRAM to run the teacher and, separately, the student
- The teacher model pulled locally (default: `qwen3:8b`)

## Setup

**Steps 1 to 4 are the same for both versions.**

1. **Clone this repo**
```bash
   git https://github.com/pavanshenoyk/Model_generator_and_synthetic_data_feeder
   cd Model_generator_and_synthetic_data_feeder
```

2. **Install PyTorch** using the command for your CUDA version from https://pytorch.org. Installing it first avoids getting the CPU-only build.

3. **Install the model package** ([TransformerAgentModels](https://github.com/pavanshenoyk/TransformerAgentModels)):
```bash
   pip install "TransformerAgentModels @ git+https://github.com/pavanshenoyk/TransformerAgentModels"
```

4. **Install the remaining dependencies**
```bash
   pip install -r requirements.txt
```
   `requirements.txt` covers both versions. `openai` is only used by the API script and `requests` only by the local script.

**Step 5 depends on your version.**

### 5a. [API] Set your API key

Set it as an environment variable. Never hardcode it or commit it.

```bash
# macOS / Linux
export OPENAI_API_KEY="your-key-here"

# Windows (PowerShell)
$env:OPENAI_API_KEY="your-key-here"
```

### 5b. [Local] Start Ollama and pull the teacher

```bash
ollama pull qwen3:8b
ollama serve        # skip if Ollama already runs as a background service
```

If Ollama runs on another machine, point the script at it (the default is `http://localhost:11434`):

```bash
# macOS / Linux
export OLLAMA_HOST="http://<server-ip>:11434"

# Windows (PowerShell)
$env:OLLAMA_HOST="http://<server-ip>:11434"
```

## Prepare your inputs

You need two things before running. Both versions use the same format.

**1. A sample file** (any text file) showing the exact format of one or more examples. Separate examples with `===` on its own line:

```
Q: What is 2 + 3?
A: 5
===
Q: What is 10 - 4?
A: 6
```

- **[API]** The file is uploaded once and reused for every call.
- **[Local]** The whole file is pasted into every prompt, so keep it short (a few examples). It has to fit inside `num_ctx` together with your prompt and the generated output.

**2. A data-generation prompt**, a short instruction to the teacher, for example:

> Generate arithmetic questions and answers. Vary the numbers and the operations.

- **[API]** The script only sends your prompt plus "Generate N different examples", so your prompt must tell the teacher to separate examples with a line containing only `===` and to follow the sample file's format.
- **[Local]** The script already adds the format and separator instructions, so your prompt only needs to describe the data.

The separator is set by `basic_config["data_splitter"]` and must match in your prompt and sample file.

## Usage

Run the script for the teacher you picked:

```bash
# API version (OpenAI teacher)
python Generator.py

# Local version (Ollama teacher)
python Generator_v2.py
```

You'll be asked for:

| Prompt | What to enter |
|---|---|
| `Enter model name` | Name for this model, used in folder and file names |
| `Select option` | `1` start from scratch, `2` continue from an existing agent (see [Execution modes](#execution-modes)) |
| `enter the version number` | (option 2 only) the version to resume from, e.g. `3` for `v3` |
| `enter the prompt for data generation` | Your teacher prompt |
| `samples of the format of data you require` | Path to your sample file |

Then it runs unattended. Each cycle prints the teacher token counts, the loss and the test accuracy. **[API]** also prints the running spend. **[Local]** first checks that Ollama is reachable and the model is installed.

**First run tip:** do a cheap dry run before a long run.
- **[API]** Set `train_examples=64`, `examples_per_call=16` and `spend_cap_usd=0.5`.
- **[Local]** Set `train_examples=64` and `test_examples=8`.
- **Both:** run one cycle, then open the saved samples to check the teacher follows your format.

## Execution modes

After entering the model name, you pick one of two modes. This works the same in both versions.

### 1. Start from scratch

Builds a brand-new, randomly initialized model using the sizes in `MODEL_ARGS`.

```
Enter model name: mathbot
Select option: 1
```

- Saved agents are numbered from `v1`: `agents/mathbot/v1/`, `v2/`, and so on.
- The model size comes from `MODEL_ARGS` at the top of the script.
- Use a **new model name**. Starting from scratch with a name that already has saved agents overwrites `v1`.

### 2. Continue from an existing agent

Loads a saved agent and keeps training it.

```
Enter model name: mathbot
Select option: 2
enter the version number: v3
```

- The model name must match the folder under `agents/`, and the version must exist (`agents/mathbot/v3/agent.pt`).
- The model size comes from the saved agent, so `MODEL_ARGS` is ignored.
- Training moves to the GPU automatically if one is available.
- New checkpoints continue from the next number (`v4`, `v5`, ...), so the version you loaded is never overwritten.
- You are asked for the data prompt and sample file again, so you can continue on the same data or on new data.
- The optimizer state is not saved, so loss may jump briefly on the first cycle.
- Generated samples and logs are appended to the existing files for that model name, not replaced.

## Configuration

All settings are at the top of each script.

### Teacher settings (differ by version)

**[API] `gpt_config`**

| Setting | Meaning |
|---|---|
| `model` | Teacher model name. Set it to a model your account has access to |
| `api_key` | Read from `OPENAI_API_KEY` |
| `reasoning_effort` | Teacher reasoning level (lower is cheaper). Check the docs for allowed values |
| `max_output_tokens` | Must fit `examples_per_call` examples plus reasoning tokens |
| `spend_cap_usd` | Hard stop for teacher spend |
| `price_in` / `price_out` | USD per 1M tokens, used for the spend estimate. Check them against current pricing |
| `accuracy_threshold` | Test accuracy (%) needed to count as a success |
| `required_successes` | Consecutive passing cycles needed to finish |

**[Local] `ollama_config`**

| Setting | Meaning |
|---|---|
| `host` | Ollama server URL (from `OLLAMA_HOST`, default `http://localhost:11434`) |
| `model` | Teacher model name, must match `ollama list` exactly |
| `think` | Reasoning mode. Off is faster and gives cleaner structured output |
| `num_ctx` | Context window. Must fit the sample file, your prompt and the output |
| `num_predict` | Max tokens generated per call. Must fit `examples_per_call` examples |
| `temperature` / `top_p` | Sampling settings. Higher temperature gives more varied examples |
| `timeout_s` | Per-call timeout in seconds |
| `accuracy_threshold` | Test accuracy (%) needed to count as a success |
| `required_successes` | Consecutive passing cycles needed to finish |

### Data settings (same in both)

| Setting | Meaning |
|---|---|
| `data_splitter` | Separator between examples (default `===`) |
| `train_examples` | Fresh training examples per cycle |
| `test_examples` | Examples in the test set |
| `examples_per_call` | Examples requested per teacher call. **[API]** default 32. **[Local]** default 4, since small batches finish more reliably on a local model |

### Training settings (same in both)

| Setting | Meaning |
|---|---|
| `batch_size`, `lr`, `epochs_per_cycle`, `grad_clip` | Standard training settings |
| `checkpoint_hours` | Save a versioned agent this often |
| `halt_hours` | Ask whether to continue after this many hours |
| `MODEL_ARGS` | Student size: `d_model`, `d_ff`, `head`, `seq_length`, `dropout`, `num_layers` |

## Output

Same in both versions:

```
agents/<model_name>/v1/agent.pt     saved models (checkpoints and final)
training_sample/<model_name>.txt    every training example generated
testing_sample/<model_name>.txt     every test example generated
training_history/<model_name>_training_log.csv   loss per cycle
testing_history/<model_name>_testing_log.csv     accuracy per cycle
```

## Notes and limitations

**Both versions**
- **No format validation:** examples are not checked against your sample format, so a malformed example from the teacher gets trained on. Check the saved samples after the first cycle.
- **Accuracy target:** next-token accuracy is easy to reach on rigid, structured data and may never reach 80% on free-form text. Set `accuracy_threshold` to match your data.
- **Train/test overlap:** there is no dedup, and both sets come from the same teacher and prompt, so near-duplicates can inflate accuracy.
- **Saved agents are pickled** with `torch.save`. Don't rename or move the `agent` or model classes after saving agents, or loading will break.
- **Interrupting:** if the run crashes or is stopped between checkpoints, progress since the last hourly checkpoint is lost.

**[API] only**
- **Cost:** each cycle makes about `train_examples / examples_per_call` calls, and the sample file is sent with every call. Watch the spend after the first cycle.
- **Spend cap:** hitting it stops the run, so set it above what a full run needs.
- **Truncation:** if you see the truncation warning often, `max_output_tokens` is too low for `examples_per_call`. You pay for output you throw away.

**[Local] only**
- **Speed:** generation dominates each cycle. A smaller teacher model or a lower `examples_per_call` speeds up each call.
- **VRAM:** the teacher and student share one GPU. The teacher is unloaded before training, but a large `num_ctx` increases its memory use.

## Security

- Never commit API keys. **[API]** Keep the key in an environment variable only.
- Add this to `.gitignore`:

```
agents/
training_sample/
testing_sample/
training_history/
testing_history/
.env
```

## License

MIT

