import os
import re
import time
import torch
import torch.nn.functional as F
import pandas as pd
import openai
from TransformerAgentModels.models import decoder_only as Decoder_only
from TransformerAgentModels.agent import agent

#the above thing is temporary
basic_config = {
    "location": os.getcwd(),
    "model_name": "",
    "model_type": "decoder_only",
    "file": "",
    "file_id": None, 
    "count": 1,
    "data_splitter":"==="} 

gpt_config = {
    "model": "gpt-6-luna",
    "prompt": "",
    "api_key": os.environ.get('OPENAI_API_KEY'),
    "reasoning_effort": "low",      # try "none" to cut cost; verify allowed values in the docs
    "max_output_tokens": 32000,     # must fit examples_per_call examples plus any reasoning tokens
    "spend_cap_usd": 10.0,          # hard stop, raise it whenever you want
    "price_in": 0.10,               # $ per 1M input tokens
    "price_out": 0.50,              # $ per 1M output tokens
    "accuracy_threshold": 80.0,
    "required_successes": 2}

train_test_config = {
    "train_examples": 1024,
    "test_examples": 32,
    "examples_per_call": 32
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

_client = openai.OpenAI(api_key=gpt_config["api_key"])

spend = {"in": 0, "out": 0}


def track_spend(response):
    """Accumulate token usage, print running cost, stop if the cap is hit."""
    u = response.usage
    spend["in"] += u.input_tokens
    spend["out"] += u.output_tokens
    cost = (spend["in"] * gpt_config["price_in"] + spend["out"] * gpt_config["price_out"]) / 1e6
    print(f"  teacher spend so far: ${cost:.3f}")
    if cost >= gpt_config["spend_cap_usd"]:
        raise RuntimeError(f"Spend cap of ${gpt_config['spend_cap_usd']} reached, stopping")


def call_teacher(k):
    text = (
        f"{gpt_config['prompt']}\n\n"
        f"Generate {k} different examples.\n\n"
    )

    if basic_config["file_id"] is None:
        with open(basic_config["file"], "rb") as f:
            basic_config["file_id"] = _client.files.create(file=f, purpose="user_data").id

    response = _client.responses.create(
        model=gpt_config["model"],
        reasoning={"effort": gpt_config["reasoning_effort"]},
        max_output_tokens=gpt_config["max_output_tokens"],
        input=[{
            "role": "user",
            "content": [
                {"type": "input_file", "file_id": basic_config["file_id"]},
                {"type": "input_text", "text": text},
            ],
        }],
    )
    track_spend(response)
    out=response.output_text
    if response.status == "incomplete":
        print("Warning: teacher output was truncated, last example may be cut off")
        out = out.rsplit(f"\n{basic_config['data_splitter']}", 1)[0] if f"\n{basic_config['data_splitter']}" in out else out
    return out


def parse_examples(text):
    pattern = rf"^\s*{re.escape(basic_config['data_splitter'])}\s*$"
    parts = re.split(pattern, text, flags=re.MULTILINE)
    return [p.strip() for p in parts if p.strip()]


def save_sample(ex,save):
    """Append one accepted training example to training_sample/<model_name>.txt."""
    if save=="training":directory = os.path.join(basic_config["location"], "training_sample")
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
            save_sample(ex,save)
    return examples


def generate_data(kind):
    if kind == "testing":
        return collect(train_test_config["test_examples"], save=kind)

    n = train_test_config["train_examples"]
    return collect(n, save=kind)
            



def log(data,action):
    if action == "training": folder="training_history"
    else: folder="testing_history"
    directory = os.path.join(basic_config["location"],folder)
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

    gpt_config["prompt"] = input("enter the prompt for data generation :")
    basic_config["file"] = input("samples of the format of data you require: ")
    basic_config["file_id"] = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=data_config["lr"])

    testing_data = generate_data("testing")

    consecutive_success = 0
    cycle = 0

    version_start = time.time()
    halt_start = time.time()

    while consecutive_success < gpt_config["required_successes"]:
        cycle += 1

        training_data = generate_data("training")
        loss = train(model, training_data, optimizer)
        log({"cycle": cycle, "loss": loss, "time": time.time()},"training")

        accuracy = test(model, testing_data)
        log({"cycle": cycle, "accuracy": accuracy, "time": time.time()},"testing")
        print(f"\nCycle {cycle} | loss {loss:.4f} | test accuracy {accuracy:.2f}%")

        if accuracy >= gpt_config["accuracy_threshold"]:
            consecutive_success += 1
            print(f"Threshold reached ({consecutive_success}/{gpt_config['required_successes']} consecutive successes)")

            # new test set only after passing, and only if we're not done
            if consecutive_success < gpt_config["required_successes"]:
                testing_data = generate_data("testing")
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