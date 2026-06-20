import torch
from model import model, tokenizer

def load_prompts(path1, path2):
    positive = []
    negative = []

    with open(path1,"r",encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            positive.append(line)

    with open(path2,"r",encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            negative.append(line)
    return positive, negative

def extractor(prompts, layer):
    all_acts = []
    for prompt in prompts:
        acts = {}
        def hook(module,input,output):  #The PyTorch Forward Hook - A forward_hook is a listener attached to a specific layer. When data passes through that layer, the hook triggers automatically.
            acts["h"] = output[0].detach() 
        h = model.transformer.h[layer].register_forward_hook(hook)
        inputs = tokenizer(prompt, return_tensors = "pt") #Tokenizing and Running the Model
        with torch.no_grad():
            model(**inputs)
        h.remove() #hooks would keep accumulating every time you run the function, slowing down your system or crashing your memory.
        all_acts.append(acts["h"][-1,:]) #In general [batch_size, sequence_length, hidden_dimension] - 0: Selects the first (and only) batch. , -1: Selects the very last token in the sequence,: Selects all features across the hidden dimension size (e.g., 4096 dimensions for Llama-7B).
    return torch.stack(all_acts)


def compute_direction(positive, negative, layer=6):
    #passing positive and negative examples for activation extraction
    pos_act = extractor(positive, layer)
    neg_act = extractor(negative, layer)
    '''
    This is Contrastive Activation Addition (CAA)
    '''
    #computing the mean and dif between positive and negative act
    direction = pos_act.mean(dim=0) - neg_act.mean(dim=0)
    direction = direction / direction.norm()

    return direction


if __name__ == "__main__":
    positive, negative = load_prompts("data/positive.txt", "data/negative.txt")
    direction = compute_direction(positive, negative, layer=6, )
    torch.save(direction, "directions/sentiment_direction.pt")
    print("Direction saved.")