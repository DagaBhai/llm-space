import torch
import torch.nn as nn
import torch.optim as optim

from transformer import TinyGPT
from tokenizer import Tokenizer

with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

vocabulary = text.split()[:20000]

text = " ".join(vocabulary)

tokenizer = Tokenizer(vocabulary)
data = tokenizer.encode(text)

block_size = 64

model = TinyGPT(
    vocab_size=len(tokenizer.vocab),
    max_seq_len=block_size,
    d_model=128
)

optimizer = optim.AdamW(
    model.parameters(),
    lr=3e-4
)

criterion = nn.CrossEntropyLoss()

model.train()

epochs = 1000

x = data[:block_size].unsqueeze(0)
y = data[1:block_size + 1].unsqueeze(0)

for epoch in range(epochs):

    optimizer.zero_grad()

    logits = model(x)

    loss = criterion(
        logits.view(-1, len(tokenizer.vocab)),
        y.view(-1)
    )

    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

def generate(model, tokenizer, prompt, max_new_tokens=20):
    model.eval()

    tokens = tokenizer.encode(prompt)
    tokens = tokens.unsqueeze(0)

    for _ in range(max_new_tokens):
        x = tokens[:, -block_size:]

        with torch.no_grad():
            logits = model(x)

        logits = logits[:, -1, :]

        next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.unsqueeze(1)

        tokens = torch.cat((tokens, next_token), dim=1)

    model.train()

    return tokenizer.decode(tokens[0])

print(generate(model, tokenizer, "hello", 20))
