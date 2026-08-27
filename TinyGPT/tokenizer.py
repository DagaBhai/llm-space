import torch

class tokenizer():
    def __init__(self, vocabulary, unk_token="<unk>", pad_token="<pad>"):
        self.unk_token = unk_token
        self.pad_token = pad_token
        
        self.vocab = [self.pad_token, self.unk_token] + sorted(list(set(vocabulary)))
        self.char_to_int = {}
        self.int_to_char = {}

        for i,word in enumerate(self.vocab):
            self.char_to_int[word] = i
            self.int_to_char[i] = word

        self.pad_id = self.char_to_int[self.pad_token]
        self.unk_id = self.char_to_int[self.unk_token]

    def encode(self, text, return_tensors=True):
        tokens = text.split(" ")
        ids = [self.char_to_int.get(token, self.unk_id) for token in tokens]
        if return_tensors:
            return torch.tensor(ids, dtype=torch.long)
        return ids

    def decode(self, tensors):
        if isinstance(tensors, torch.Tensor):
            tensors = tensors.tolist()
        return " ".join([self.int_to_char.get(i, self.unk_token) for i in tensors])
