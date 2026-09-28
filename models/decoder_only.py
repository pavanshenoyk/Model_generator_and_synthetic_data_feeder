import math
import torch
import torch.nn as nn
import tiktoken

EOT = "<|endoftext|>"


class DataEmbedding(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        # small init so that (embedding * sqrt(d_model)) has ~unit variance,
        # and so the tied output layer doesn't start with huge logits
        nn.init.normal_(self.embedding.weight, mean=0.0, std=d_model ** -0.5)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)

        x = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
        y = 10000 ** ((torch.arange(0, d_model, 2)) / d_model)

        pe[:, 0::2] = torch.sin(x / y)
        pe[:, 1::2] = torch.cos(x / y)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        if x.shape[1] > self.seq_len:
            raise ValueError(
                f"input length {x.shape[1]} exceeds seq_length {self.seq_len}; "
                "crop or chunk the input first"
            )
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff)
        self.layer2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.layer2(self.dropout(torch.relu(self.layer1(x))))


class SelfAttention(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0
        self.d_k = d_model // h
        self.dropout_p = dropout
        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)
        self.WO = nn.Linear(d_model, d_model)

    def forward(self, k, q, v):
        batch, seq_len = q.shape[0], q.shape[1]
        q = self.WQ(q)
        k = self.WK(k)
        v = self.WV(v)

        q = q.view(q.shape[0], q.shape[1], self.h, self.d_k).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.h, self.d_k).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], self.h, self.d_k).transpose(1, 2)

        x = nn.functional.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )

        x = x.transpose(1, 2).contiguous().view(batch, seq_len, self.h * self.d_k)
        return self.WO(x)


class NormalizationFunc(nn.Module):

    def __init__(self, d_model: int, eps: float = 10 ** -6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, correction=0)

        return self.alpha * ((x - mean) / (torch.sqrt(variance + self.eps))) + self.bias


class ContextAddition(nn.Module):

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.normalize = NormalizationFunc(d_model)

    def forward(self, x, sublayer):
        return self.normalize(x + self.dropout(sublayer))


class DecoderBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, h: int, dropout: float):
        super().__init__()
        self.self_attention = SelfAttention(d_model, h, dropout)
        self.residual_norm1 = ContextAddition(d_model, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.residual_norm2 = ContextAddition(d_model, dropout)

    def forward(self, x):
        x1 = self.self_attention(x, x, x)
        x2 = self.residual_norm1(x, x1)
        x3 = self.feed_forward(x2)
        x4 = self.residual_norm2(x2, x3)
        return x4


class Decoder(nn.Module):

    def __init__(self, d_model: int, d_ff: int, h: int, dropout: float, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(d_model, d_ff, h, dropout) for _ in range(num_layers)])
        self.normalize = NormalizationFunc(d_model)

    def forward(self, x):
        for block in self.layers:
            x = block(x)
        return self.normalize(x)


class Transformer(nn.Module):

    def __init__(self, d_model: int, d_ff: int, head: int, seq_length: int, dropout: float, num_layers: int):
        super().__init__()
        self.tokenizer = tiktoken.get_encoding("gpt2")
        vocab_size = self.tokenizer.n_vocab
        self.eos_id = self.tokenizer.eot_token
        self.seq_length = seq_length
        self.decoder = Decoder(d_model, d_ff, head, dropout, num_layers)
        self.data_embedder = DataEmbedding(d_model, vocab_size)
        self.positional_encoder = PositionalEncoding(d_model, seq_length, dropout)
        self.linear = nn.Linear(d_model, vocab_size)
        # weight tying: output projection shares the embedding matrix
        self.linear.weight = self.data_embedder.embedding.weight
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        embedded = self.data_embedder(x)
        pos = self.positional_encoder(embedded)
        out = self.decoder(pos)
        out = self.linear(out)
        return out

    def encode(self, text, add_eos: bool = False):
        ids = self.tokenizer.encode(text, allowed_special={EOT})
        if add_eos:
            ids.append(self.eos_id)
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    def decode(self, ids):
        return self.tokenizer.decode(ids.view(-1).tolist())

    def generate(self, text: str, max_tokens: int = 50, temperature: float = 0.0, top_k: int = None):
        """temperature=0 -> greedy. Stops early when the model emits EOS."""
        was_training = self.training
        self.eval()

        ids = self.encode(text)
        if ids.numel() == 0:
            # empty prompt: start from the EOS token as a BOS
            ids = torch.tensor([[self.eos_id]], dtype=torch.long)
        prompt_len = ids.shape[1]

        with torch.no_grad():
            for _ in range(max_tokens):
                inputs = ids[:, -self.seq_length:]
                next_logits = self.forward(inputs)[0, -1, :]

                if temperature and temperature > 0:
                    next_logits = next_logits / temperature
                    if top_k:
                        kth = torch.topk(next_logits, min(top_k, next_logits.numel())).values[-1]
                        next_logits = next_logits.masked_fill(next_logits < kth, float("-inf"))
                    next_id = torch.multinomial(torch.softmax(next_logits, dim=-1), 1)
                else:
                    next_id = torch.argmax(next_logits, keepdim=True)

                ids = torch.cat([ids, next_id.view(1, 1).to(ids.device)], dim=1)
                if next_id.item() == self.eos_id:
                    break

        self.train(was_training)

        if ids.shape[1] > prompt_len and ids[0, -1].item() == self.eos_id:
            ids = ids[:, :-1]
        return self.decode(ids)