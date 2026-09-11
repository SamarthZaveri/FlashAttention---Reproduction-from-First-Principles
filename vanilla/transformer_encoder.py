"""
Phase 3 — Full Transformer Encoder from scratch, trained end-to-end.

Builds a complete encoder layer (Multi-Head Attention + residual + LayerNorm,
Feed-Forward + residual + LayerNorm) on top of Phase 2's MultiHeadAttention,
then trains it on a small synthetic sentiment-style classification task to
confirm the whole stack is correct end-to-end (not just numerically matching
a reference layer-by-layer, but actually learnable).

The task: sequences of token ids where a handful of "positive" tokens and
"negative" tokens are sprinkled among neutral filler tokens; label = whether
positive tokens outnumber negative ones. This requires the model to
aggregate information across positions (i.e. actually use attention), while
being fully synthetic so the script has zero external data dependencies.

Run:
    python transformer_encoder.py
Expect:
    Training loss decreasing and final val accuracy printed (should reach
    >90% — if it doesn't, something upstream in Phase 1/2 is broken).
"""

import torch
import torch.nn as nn

from multihead_attention import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-norm residual attention block.
        x = x + self.dropout(self.attn(self.norm1(x)))
        # Pre-norm residual feed-forward block.
        x = x + self.ff(self.norm2(x))
        return x


class TinyTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, num_classes, max_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, token_ids):
        batch, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        x = self.token_emb(token_ids) + self.pos_emb(positions)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        pooled = x.mean(dim=1)  # mean-pool over sequence
        return self.classifier(pooled)


def make_synthetic_sentiment_dataset(num_examples, seq_len, vocab_size, device, seed):
    """Token ids 0/1 are reserved as 'negative'/'positive' signal tokens;
    everything else is neutral filler. Label = 1 if positives > negatives."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    NEG, POS = 0, 1
    x = torch.randint(2, vocab_size, (num_examples, seq_len), generator=gen)

    num_signal = max(2, seq_len // 8)
    for i in range(num_examples):
        positions = torch.randperm(seq_len, generator=gen)[:num_signal]
        n_pos = torch.randint(0, num_signal + 1, (1,), generator=gen).item()
        for j, pos in enumerate(positions):
            x[i, pos] = POS if j < n_pos else NEG

    labels = (x == POS).sum(dim=1) > (x == NEG).sum(dim=1)
    return x.to(device), labels.long().to(device)


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab_size, seq_len, embed_dim, num_heads, ff_dim, num_layers = 50, 32, 64, 4, 128, 2
    num_classes = 2

    x_train, y_train = make_synthetic_sentiment_dataset(2000, seq_len, vocab_size, device, seed=0)
    x_val, y_val = make_synthetic_sentiment_dataset(400, seq_len, vocab_size, device, seed=1)

    model = TinyTransformerClassifier(
        vocab_size, embed_dim, num_heads, ff_dim, num_layers, num_classes, max_len=seq_len
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    batch_size = 64
    num_epochs = 15
    num_batches = len(x_train) // batch_size

    for epoch in range(num_epochs):
        model.train()
        perm = torch.randperm(len(x_train), device=device)
        total_loss = 0.0
        for b in range(num_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            xb, yb = x_train[idx], y_train[idx]

            logits = model(xb)
            loss = loss_fn(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            val_acc = (val_logits.argmax(dim=-1) == y_val).float().mean().item()

        print(f"epoch {epoch+1:2d}/{num_epochs} | train_loss {total_loss/num_batches:.4f} | val_acc {val_acc:.3f}")

    print(f"\nFinal val accuracy: {val_acc:.3f}")
    assert val_acc > 0.85, "encoder failed to learn a trivially separable task — check Phases 1/2"
    print("OK: from-scratch Transformer encoder trains end-to-end and learns the task.")
