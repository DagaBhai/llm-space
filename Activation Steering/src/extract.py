import torch
from model import model, tokenizer

#examples
Positive = ["I love this film", "This was wonderful", "I'm so happy", "Brilliant performance", 
            "An absolute masterpiece", "The customer service was incredibly helpful", 
            "Highly recommend this product to everyone", "The food tasted fresh and delicious", 
            "What a beautiful and sunny day", "I feel incredibly motivated and energized", 
            "The team executed the project flawlessly", "This book is deeply inspiring and well-written", 
            "The interface is smooth and intuitive", "They showed immense kindness to the strangers", 
            "A perfect ending to a great week", "The melody is uplifting and catchy", 
            "I am genuinely grateful for your support", "Everything went exactly according to plan", 
            "The hotel room was spotless and luxurious", "She gave a warm and welcoming speech", 
            "This software saved me hours of work", "The atmosphere was vibrant and full of life", 
            "An innovative solution to a complex problem", "I have full confidence in this decision", 
            "They celebrated a magnificent victory"]

Negative = ["I hate this film", "This was terrible", "I'm so sad", "Dreadful performance", 
            "A complete waste of time", "The customer service was rude and useless", "Avoid this product at all costs", 
            "The food was bland and undercooked", "What a gloomy and depressing day", "I feel completely exhausted and drained", 
            "The team completely botched the project", "This book is boring and riddled with errors", 
            "The interface is clunky and confusing", "They were incredibly cruel to the strangers", 
            "A disastrous end to a terrible week", "The melody is grating and annoying", "I am deeply disappointed by your lack of support", 
            "Everything went completely wrong", "The hotel room was filthy and cramped", "She gave a cold and dismissive speech", 
            "This software caused me hours of extra work", "The atmosphere was dull and dead", "A lazy solution to a simple problem", 
            "I have serious doubts about this decision", "They suffered a crushing defeat"]

def extractor(prompts, layer):
    all_acts = []
    for prompt in prompts:
        acts = {}
        def hook(m,input,output):  #The PyTorch Forward Hook - A forward_hook is a listener attached to a specific layer. When data passes through that layer, the hook triggers automatically.
            acts["h"] = output[0].detach() 
        h = model.transformer.h[layer].register_forward_hook(hook)
        inputs = tokenizer(prompt, return_tensors = "pt") #Tokenizing and Running the Model
        print(inputs)
        with torch.no_grad():
            model(**inputs)
        h.remove() #hooks would keep accumulating every time you run the function, slowing down your system or crashing your memory.
        print(acts["h"])
        all_acts.append(acts["h"][-1,:]) #In general [batch_size, sequence_length, hidden_dimension] - 0: Selects the first (and only) batch. , -1: Selects the very last token in the sequence,: Selects all features across the hidden dimension size (e.g., 4096 dimensions for Llama-7B).
    return torch.stack(all_acts)

#passing positive and negative examples for activation extraction
pos_act = extractor(Positive, 6)
neg_act = extractor(Negative, 6)

'''
This is Contrastive Activation Addition (CAA)
'''
#computing the mean and dif between positive and negative act
direction = pos_act.mean(dim=0) - neg_act.mean(dim=0)
direction = direction / direction.norm() #normalizing

        
