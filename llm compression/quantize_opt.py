import torch
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

'''

def quantize_symmetric_per_tensor(x, bits=8, data_type=torch.int8):
    #find the maximum abs value for the entire tensor
    x_max = torch.max(torch.abs(x)).item()

    #defualt int8 bound [-127,127]
    q_max = 2**(bits-1) -1

    scale = x_max/q_max if x_max > 0 else 0

    # -q_max = q_max using the symmetric clipping
    x_q = torch.clamp(torch.round(x/scale),-q_max, q_max)

    return x_q.to(data_type).detach(), scale

def dequantize_symmetric_per_tensor(x, scale):
    return x.float() * scale
'''
    
class QuantizedLinear(nn.Module):
    def __init__(self, x, bias=None, bits=8, data_type = torch.int8):
        super().__init__()

        self.scales, self.q_max = self._compute_scale(x, bits)
        self.bias = bias
        q_weight = self._quantize_symmetric_weight(x).to(data_type).detach()
        self.register_buffer("weight", q_weight)
        self.register_buffer("scale", torch.tensor(self.scales))

    def _compute_scale(self, x, bits):
        x_max = torch.max(torch.abs(x)).item()
        q_max = 2 ** (bits - 1) -1
        scale = x_max/q_max if x_max > 0 else 1
        return scale, q_max
    
    def _quantize_symmetric_weight(self, x):
        return torch.clamp(torch.round(x/self.scales),-self.q_max,self.q_max)
    
    def forward(self, x):
        w = self.weight.float() * self.scales
        return F.linear(x, w, self.bias)

    
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path="facebook/opt-125m",
                                            device_map="auto", 
                                            cache_dir="./cache")
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="facebook/opt-125m",
                                          cache_dir="./cache")

print("model loaded")

#not necessary
q_proj = []
k_proj = []
v_proj = []
out_proj = []
fc_proj = []

'''
Fully Connected layer. In the context of "between kvqout" (Queries, Keys, Values, and output), 
the FC layers refer to the linear feed-forward projections used to process the outputs of the attention mechanism
'''

for name, module in model.named_modules():

    if isinstance(module, nn.Linear):
        print(f"Quantizing: {name}\n")
        print("Before:", module.weight.data.dtype, "\n")
        quant_layer = QuantizedLinear(module.weight.data, module.bias)
        module.weight = nn.Parameter(quant_layer.weight.float(), requires_grad=False)
        print("After:", quant_layer.weight.dtype, "\n")
        print(f"Quantizing: {name} done\n")

output_dir = Path(r"..\quantized_model")
file_name = "quantized_int8_opt128.pt"
output_dir.mkdir(parents=True, exist_ok=True)

save_path =  output_dir / file_name

torch.save(model.state_dict(), save_path)
print(f"Model successfully saved to: {save_path}")
