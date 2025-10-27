#%% packages
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#%% test
sample_sentence = "the man ate the pizza because it smelled delicious"


#%% Get word encodings and attention weights from BERT
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased", output_attentions=True)
inputs = tokenizer(sample_sentence, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    outputs = model(**inputs)
word_encodings = outputs.last_hidden_state
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

#%% Get attention weights from all layers and heads
attention_weights = outputs.attentions  # Shape: (num_layers, batch_size, num_heads, seq_len, seq_len)

# # Use attention from the last layer, first head (or average across heads)
last_layer_attention = attention_weights[-1][0]  # Shape: (num_heads, seq_len, seq_len)
# # Average across all attention heads
avg_attention = last_layer_attention.mean(dim=0)  # Shape: (seq_len, seq_len)



#%% Create a clean list of tokens (remove special tokens)
clean_tokens = []
for i, token in enumerate(tokens):
    if not token.startswith('[') and not token.startswith('<'):
        clean_tokens.append(token)

# Create attention matrix for clean tokens only
clean_attention_matrix = []
for i, token in enumerate(tokens):
    if not token.startswith('[') and not token.startswith('<'):
        row = []
        for j, target_token in enumerate(tokens):
            if not target_token.startswith('[') and not target_token.startswith('<'):
                row.append(avg_attention[i, j].item())
        clean_attention_matrix.append(row)

clean_attention_matrix = np.array(clean_attention_matrix)

# Find the index of "it" in clean_tokens
it_idx = clean_tokens.index("it")

# Create the heatmap showing only the "it" row
plt.figure(figsize=(10, 3))
sns.heatmap(clean_attention_matrix[it_idx:it_idx+1], 
            xticklabels=clean_tokens,
            yticklabels=["it"],
            annot=True,
            fmt='.3f', 
            cmap='Blues',
            cbar_kws={'label': 'Attention Weight'})

plt.xlabel('Target Words')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %%
