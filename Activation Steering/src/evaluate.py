from extract import load_prompts, compute_direction
from steering import make_hook, output
import pandas as pd

eval_set_path = [
    [r"evaluation dataset\france_n_japan\france.txt", r"evaluation dataset\france_n_japan\japan.txt"],
    [r"evaluation dataset\positive_n_negative\positive.txt", r"evaluation dataset\positive_n_negative\negative.txt"],
    [r"evaluation dataset\refusal_n_compliance\compliance.txt", r"evaluation dataset\refusal_n_compliance\refusal.txt"]
    ]

prompt = [
    "The most famous landmark in my country is",        # France vs Japan
    "How was your day today?",                          # Positive vs Negative (keep)
    "My response to your request is",                   # Compliance vs Refusal
]

alphalst = [20,10,15]

direcion_alpha_output = [[0,0] for _ in range(len(eval_set_path))] 

for i, paths in enumerate(eval_set_path):
    positve, negative = load_prompts(path1=paths[0],path2=paths[1])
    computed_direction = compute_direction(positve, negative, layer=6)
    positve_alpha = output(prompt[i], alphalst[i], direction=computed_direction)
    negative_alpha = output(prompt[i], -alphalst[i], direction=computed_direction)
    direcion_alpha_output[i][0] = positve_alpha
    direcion_alpha_output[i][1] = negative_alpha
    
print(direcion_alpha_output)

Data = {
    "prompt" : prompt,
    "positive alpha" : [direcion_alpha_output[i][0] for i in range(len(direcion_alpha_output))],
    "negative alpha" : [direcion_alpha_output[i][1] for i in range(len(direcion_alpha_output))],
}

df = pd.DataFrame(Data)
df.to_csv("evaluation_outputs.csv", index=False)
