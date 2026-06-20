import torch
from model import model, tokenizer

def make_hook(alpha, direction_vector):
    def hook(module, input, output):
        vec = direction_vector.to(device=output[0].device, dtype=output[0].dtype)
        output[0] += alpha * vec
        return output
    return hook

def output(prompt, alpha, direction =  torch.load("directions\sentiment_direction.pt"), layer = 6):
    h = model.transformer.h[layer].register_forward_hook(make_hook(alpha, direction_vector = direction))
    inputs = tokenizer(prompt, return_tensors="pt")
    out = model.generate(**inputs)
    h.remove()
    return tokenizer.decode(out[0])

if __name__ == "__main__":
    prompt = "The weather today is"
    print(output(prompt, alpha = 15))
    print(output(prompt, alpha = -15))
