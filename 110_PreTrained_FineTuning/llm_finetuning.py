#%% packages
from datasets import load_dataset
from transformers import GemmaForSequenceClassification, GemmaTokenizer
import huggingface_hub
import torch
from collections import Counter
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

#%% delete GPU memory
torch.cuda.empty_cache()

#%% hyperparameters
EPOCHS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
LR = 0.001

#%% Load pre-trained model and tokenizer
model_name = "google/gemma-3-270m-it"
model = GemmaForSequenceClassification.from_pretrained(model_name)
model.to(DEVICE)
tokenizer = GemmaTokenizer.from_pretrained(model_name)


#%% Load the dataset
dataset = load_dataset("Deysi/spam-detection-dataset")

# Map string labels to integer class ids
label2id = {"not_spam": 0, "spam": 1}
id2label = {v: k for k, v in label2id.items()}
dataset = dataset.map(
    lambda batch: {"label": [label2id[label] for label in batch["label"]]},
    batched=True,
)

#%% check the class distribution by counting the 0 and 1s

print(Counter(dataset["train"]["label"]))


#%% Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Remove text and other columns not needed for training to streamline
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"],
)

# Ensure model is aware of label mappings
model.config.label2id = label2id
model.config.id2label = id2label



#%% Create data loaders
train_loader = DataLoader(tokenized_datasets["train"], batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(tokenized_datasets["test"], batch_size=BATCH_SIZE, shuffle=False)



# Define loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

#%% Training loop
model.train()
train_losses = []
for epoch in range(EPOCHS):
    train_loss_epoch = 0
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        # Move only tensors to device (after ensuring labels are tensors via set_format)
        batch = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        outputs = model(**batch)
        # The outputs from the model contain the logits and other information
        loss = loss_fn(outputs.logits, batch["labels"])
        
        loss.backward()
        optimizer.step()
        train_loss_epoch += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}: Training batch {batch_idx + 1}/{len(train_loader)+1}: Loss: {loss.item()}")
    train_losses.append(train_loss_epoch/len(train_loader))
    print(f"Epoch {epoch+1}/{EPOCHS}: Loss: {train_losses[-1]}")
    
# %% Evaluate the model
model.eval()
predicted_labels_all, true_labels_all = [], []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        true_labels = batch["labels"].numpy().tolist()
        
        pred_labels = model(input_ids, attention_mask=attention_mask)
        logits = pred_labels.logits
        predicted_labels = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        
        predicted_labels_all.extend(predicted_labels)
        true_labels_all.extend(true_labels)

#%% add confusion matrix
cm = confusion_matrix(true_labels_all, predicted_labels_all)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Vorhergesagte Labels')
plt.ylabel('Tatsächliche Labels')
plt.title('Konfusionsmatrix')
plt.show()




#%%
accuracy = accuracy_score(true_labels_all, predicted_labels_all)
#%%
print(f"Validierungsgenauigkeit: {(accuracy*100):.1f}%")



# %%
