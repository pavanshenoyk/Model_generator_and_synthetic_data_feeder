import os
import re
import time
import requests
import torch
import torch.nn.functional as F
import pandas as pd
from TransformerAgentModels.models import decoder_only as Decoder_only
from TransformerAgentModels.agent import agent

basic_config = {
    "location": os.getcwd(),
    "model_name": "",
    "model_type": "decoder_only",
    "file": "",
    "samples_text": "",
    "count": 1,
    "data_splitter": "==="}

ollama_config = {
    "host": os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
    "model": "qwen3:8b",
    "prompt": "",
    "think": False,              # reasoning off: faster, and cleaner structured output
    "num_ctx": 12288,            # must fit samples file + prompt + examples_per_call outputs
    "num_predict": 3000,         # max tokens generated per call
    "temperature": 0.9,          # higher = more varied examples
    "top_p": 0.95,
    "timeout_s": 1800,
    "accuracy_threshold": 80.0,
    "required_successes": 2}

train_test_config = {
    "train_examples": 1024,
    "test_examples": 32,
    "examples_per_call": 4      # local model: small batches stay in context and finish reliably
}

data_config = {
    "batch_size": 16,
    "epochs_per_cycle": 1,
    "lr": 3e-4,
    "grad_clip": 1.0,
    "checkpoint_hours": 1,
    "halt_hours": 12
}

# placeholder sizes -- tune for your GPU
MODEL_ARGS = dict(d_model=256, d_ff=1024, head=8, seq_length=256, dropout=0.1, num_layers=4)


def check_ollama():
    try:
        tags = requests.get(f"{ollama_config['host']}/api/tags", timeout=10).json()
    except requests.RequestException as e:
        raise RuntimeError(f"Can't reach Ollama at {ollama_config['host']}: {e}")
    names = [m["name"] for m in tags.get("models", [])]
    if ollama_config["model"] not in names:
        raise RuntimeError(f"Model {ollama_config['model']} not found. Installed: {names}")


def unload_teacher():
    """Free the teacher's VRAM so training gets the GPU."""
    try:
        requests.post(f"{ollama_config['host']}/api/generate",
                      json={"model": ollama_config["model"], "keep_alive": 0}, timeout=60)
    except requests.RequestException:
        pass


def _split_re():
    return re.compile(rf"^\s*{re.escape(basic_config['data_splitter'])}\s*$", re.MULTILINE)


def call_teacher(k):
    text = (
        f"{ollama_config['prompt']}\n\n"
        f"Here are samples of the required format:\n<samples>\n{basic_config['samples_text']}\n</samples>\n\n"
        f"Generate {k} different examples in the same format as the samples. "
        f"Put a line containing only {basic_config['data_splitter']} between examples and output nothing else."
    )

    payload = {
        "model": ollama_config["model"],
        "messages": [{"role": "user", "content": text}],
        "stream": False,
        "think": ollama_config["think"],
        "options": {
            "num_ctx": ollama_config["num_ctx"],
            "num_predict": ollama_config["num_predict"],
            "temperature": ollama_config["temperature"],
            "top_p": ollama_config["top_p"],
        },
    }
    r = requests.post(f"{ollama_config['host']}/api/chat", json=payload,
                      timeout=ollama_config["timeout_s"])
    r.raise_for_status()
    data = r.json()
    print(f"  teacher: {data.get('prompt_eval_count', '?')} in / {data.get('eval_count', '?')} out tokens")

    out = data["message"]["content"]
    if data.get("done_reason") == "length":
        print("Warning: teacher output was truncated, dropping the last example")
        parts = _split_re().split(out)
        out = f"\n{basic_config['data_splitter']}\n".join(parts[:-1])  # "" if only one part
    return out


def parse_examples(text):
    return [p.strip() for p in _split_re().split(text) if p.strip()]


def save_sample(ex, save):
    """Append one accepted example to training_sample/<model_name>.txt or testing_sample/<model_name>.txt."""
    if save == "training": directory = os.path.join(basic_config["location"], "training_sample")
    else: directory = os.path.join(basic_config["location"], "testing_sample")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{basic_config['model_name']}.txt")
    with open(path, "a", encoding="utf-8") as f:
        f.write(ex + f"\n{basic_config['data_splitter']}\n")


def collect(count, save):
    examples, misses = [], 0
    while len(examples) < count:
        k = min(train_test_config["examples_per_call"], count - len(examples))
        got = parse_examples(call_teacher(k))[:k]
        if not got:
            misses += 1
            if misses >= 3:
                raise RuntimeError("teacher model returned no parsable examples 3 times in a row")
            continue
        misses = 0
        examples.extend(got)

        for ex in got:
            save_sample(ex, save)
        print(f"  collected {len(examples)}/{count}")
    return examples


def generate_data(kind):
    if kind == "testing":
        return collect(train_test_config["test_examples"], save=kind)

    n = train_test_config["train_examples"]
    return collect(n, save=kind)


def log(data, action):
    if action == "training": folder = "training_history"
    else: folder = "testing_history"
    directory = os.path.join(basic_config["location"], folder)
    os.makedirs(directory, exist_ok=True)

    file_path = os.path.join(directory, f"{basic_config['model_name']}_{action}_log.csv")

    df = pd.DataFrame([data])
    if os.path.exists(file_path):
        df.to_csv(file_path, mode="a", header=False, index=False)
    else:
        df.to_csv(file_path, index=False)


def make_windows(model, examples, shuffle=False):
    """Tokenize examples (EOS-terminated), pack into one stream, cut into
    (x, y) windows where y is x shifted by one token."""
    stream = []
    for ex in examples:
        stream.extend(model.encode(ex, add_eos=True)[0].tolist())
    if len(stream) < 2:
        raise ValueError("not enough data to build a training window")

    seq = model.seq_length
    if shuffle:
        # random offset so windows differ between epochs/cycles
        stream = stream[torch.randint(0, min(seq, max(1, len(stream) - seq - 1)), (1,)).item():]

    t = torch.tensor(stream, dtype=torch.long)
    n = (len(t) - 1) // seq
    if n == 0:
        return t[:-1].unsqueeze(0), t[1:].unsqueeze(0)

    x = torch.stack([t[i * seq:(i + 1) * seq] for i in range(n)])
    y = torch.stack([t[i * seq + 1:(i + 1) * seq + 1] for i in range(n)])
    return x, y


def train(model, examples, optimizer):
    """One training cycle over freshly generated data. Returns mean loss."""
    model.train()
    losses = []
    bs = data_config["batch_size"]

    for _ in range(data_config["epochs_per_cycle"]):
        x, y = make_windows(model, examples, shuffle=True)
        order = torch.randperm(x.shape[0])

        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            xb, yb = x[idx], y[idx].to(model.device)

            logits = model(xb)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), data_config["grad_clip"])
            optimizer.step()
            losses.append(loss.item())

    return sum(losses) / len(losses)


@torch.no_grad()
def test(model, examples):
    """Next-token accuracy (%) over the test set."""
    model.eval()
    x, y = make_windows(model, examples)
    bs = data_config["batch_size"]
    correct = total = 0

    for i in range(0, x.shape[0], bs):
        xb, yb = x[i:i + bs], y[i:i + bs].to(model.device)
        pred = model(xb).argmax(dim=-1)
        correct += (pred == yb).sum().item()
        total += yb.numel()

    return 100.0 * correct / total


def create_agent(model):
    version = f"v{basic_config['count']}"

    agent_ = agent(
        model=model,
        name=basic_config["model_name"],
        version=version
    )

    agent_path = os.path.join(basic_config["location"], "agents", basic_config["model_name"], version)
    os.makedirs(agent_path, exist_ok=True)
    torch.save(agent_, os.path.join(agent_path, "agent.pt"))

    basic_config["count"] += 1
    return agent_


def main():
    print("MODEL GENERATOR")
    basic_config["model_name"] = input("Enter model name: ").strip()
    print("\n1. Start from scratch")
    print("2. Continue from existing agent")
    choice = input("Select option: ")

    if choice == "1":
        model = Decoder_only(**MODEL_ARGS)
    elif choice == "2":
        version = int(input("enter the version number: v"))
        path = os.path.join(basic_config["location"], "agents", basic_config["model_name"], f"v{version}", "agent.pt")
        loaded = torch.load(path, weights_only=False, map_location="cpu")
        model = loaded.model
        model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(model.device)
        basic_config["count"] = version + 1   # don't overwrite the version we loaded
    else:
        print("Invalid choice")
        return

    ollama_config["prompt"] = input("enter the prompt for data generation :")
    basic_config["file"] = input("samples of the format of data you require: ")
    with open(basic_config["file"], "r", encoding="utf-8") as f:
        basic_config["samples_text"] = f.read()
    check_ollama()

    optimizer = torch.optim.AdamW(model.parameters(), lr=data_config["lr"])

    testing_data = generate_data("testing")
    unload_teacher()

    consecutive_success = 0
    cycle = 0

    version_start = time.time()
    halt_start = time.time()

    while consecutive_success < ollama_config["required_successes"]:
        cycle += 1

        training_data = generate_data("training")
        unload_teacher()   # give the GPU back before training
        loss = train(model, training_data, optimizer)
        log({"cycle": cycle, "loss": loss, "time": time.time()}, "training")

        accuracy = test(model, testing_data)
        log({"cycle": cycle, "accuracy": accuracy, "time": time.time()}, "testing")
        print(f"\nCycle {cycle} | loss {loss:.4f} | test accuracy {accuracy:.2f}%")

        if accuracy >= ollama_config["accuracy_threshold"]:
            consecutive_success += 1
            print(f"Threshold reached ({consecutive_success}/{ollama_config['required_successes']} consecutive successes)")

            # new test set only after passing, and only if we're not done
            if consecutive_success < ollama_config["required_successes"]:
                testing_data = generate_data("testing")
                unload_teacher()
        else:
            consecutive_success = 0
            print("Threshold not reached.")

        if time.time() - version_start >= data_config["checkpoint_hours"] * 3600:
            saved = create_agent(model)
            print(f"\nAgent {saved.version} checkpoint created.")
            version_start = time.time()

        if time.time() - halt_start >= data_config["halt_hours"] * 3600:
            print("\n================================")
            print(f"{data_config['halt_hours']}-HOUR TRAINING HALT")
            print("================================")

            if input("Continue training? (y/n): ").lower() != "y":
                saved = create_agent(model)
                print(f"Training stopped by user. Agent {saved.version} saved.")
                return saved

            halt_start = time.time()

    print("\n================================")
    print("TRAINING COMPLETE")
    print("================================")

    final_agent = create_agent(model)
    print(f"Final agent {final_agent.version} created.")
    return final_agent


if __name__ == "__main__":
    main()