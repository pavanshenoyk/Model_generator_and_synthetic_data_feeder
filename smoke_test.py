import os, random
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "dummy")  # OpenAI() just needs some value to construct

import Generator as G

TEMPLATES = [
    "Input: add 2 and 3\nOutput: 5",
    "Input: add 4 and 4\nOutput: 8",
    "Input: say hello\nOutput: hello",
    "Input: say bye\nOutput: bye",
    "Input: count to three\nOutput: 1 2 3",
]

def fake_teacher(k, incomplete):
    return "\n===\n".join(random.choice(TEMPLATES) for _ in range(k))

G.call_teacher = fake_teacher
G.MODEL_ARGS = dict(d_model=64, d_ff=128, head=4, seq_length=64, dropout=0.1, num_layers=2)
G.train_test_config.update(train_examples=64, test_examples=16, examples_per_call=16)
G.data_config.update(epochs_per_cycle=30, lr=1e-3, checkpoint_hours=0.0003)  # ~1s checkpoints

def run(answers):
    with patch("builtins.input", side_effect=iter(answers)):
        return G.main()

a = run(["smoke", "1", "any prompt", "unused.txt"])
print("first run done:", a.version)

b = run(["smoke", "2", a.version[1:], "any prompt", "unused.txt"])
print("resume ok:", b.version)