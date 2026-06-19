import torch
from model import model, tokenizer
from extract import direction

def make_hook(direction_vector, alpha):
    def hook(module, input, output):
        vec = torch.tensor(direction_vector, device=output[0].device, dtype=output[0].dtype)
        output[0] += alpha * direction_vector
        return output
    return hook

prompt = "The weather today is"

for alpha in [0,2,5,15,-15]:
    h = model.transformer.h[8].register_forward_hook(make_hook(direction,alpha))
    inputs = tokenizer(prompt, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=20)
    print(f"alpha={alpha}:", tokenizer.decode(out[0]))
    h.remove()