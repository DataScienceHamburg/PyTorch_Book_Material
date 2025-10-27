#%% packages
import os
import torch
import torch.nn.functional as F
import seaborn as sns
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GATConv
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

#%% Load Dataset
dataset_path = 'data/Planetoid'
if not os.path.exists(dataset_path):
    dataset = Planetoid(root=dataset_path, name='Cora', transform=NormalizeFeatures())
else:
    dataset = Planetoid(root=dataset_path, name='Cora', transform=NormalizeFeatures(), force_reload=False)
data = dataset[0]
data = data.to(device) 

print(f'Anzahl der Graphen: {len(dataset)}')
print(f'Anzahl der Merkmale: {dataset.num_features}')
print(f'Anzahl der Klassen: {dataset.num_classes}')
print(f'Graph-Objekt: {data}')
print(f'Ist ein gerichteter Graph: {data.is_directed()}')
print(f'Anzahl der Knoten: {data.num_nodes}')
print(f'Anzahl der Kanten: {data.num_edges}')
print(f'Anzahl der Trainingsknoten: {data.train_mask.sum()}')
print(f'Anzahl der Validierungsknoten: {data.val_mask.sum()}')
print(f'Anzahl der Testknoten: {data.test_mask.sum()}')
print(f"Anzahl der Knoten im Graphen: {data.num_nodes}")

#%% get number of elements per class
# Get class distribution
class_counts = torch.bincount(data.y)
for class_idx, count in enumerate(class_counts):
    print(f'Klasse {class_idx}: {count.item()} Knoten')

# Calculate dummy classifier accuracy (majority class)
majority_class = torch.argmax(class_counts)
dummy_predictions = torch.full_like(data.y, majority_class)
dummy_acc = (dummy_predictions[data.test_mask] == data.y[data.test_mask]).float().mean()
print(f'\nGenauigkeit des Dummy-Klassifikators: {dummy_acc:.4f}')



#%% Model Class
class GAT(torch.nn.Module):
    def __init__(self, num_features, num_hidden, num_classes, heads=8):
        super().__init__()
        self.conv1 = GATConv(num_features, num_hidden, heads=heads, dropout=0.6)
        self.conv2 = GATConv(heads * num_hidden, num_classes, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x

#%% Model Instance, Optimizer, and Loss Function
model = GAT(num_features=dataset.num_features, num_hidden=8, num_classes=dataset.num_classes, heads=8).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Note on loss: For multi-class classification, CrossEntropyLoss is standard.
# However, for models returning log-softmax, Negative Log Likelihood Loss (NLLLoss) is more appropriate.
criterion = torch.nn.NLLLoss()

#%% Training and Evaluation Functions
def train(model, data, optimizer, criterion):
    """Performs a single training step."""
    model.train()  
    optimizer.zero_grad() 
    X_train = data.x
    y_true = data.y
    y_pred = model(X_train, data.edge_index)  
    train_mask = data.train_mask
    loss = criterion(y_pred[train_mask], y_true[train_mask])  
    loss.backward()  
    optimizer.step() 
    return loss.item()

def test(model, data, mask):
    """Evaluates the model on a specific mask (e.g., test or validation)."""
    model.eval()  
    with torch.no_grad():
        X_test = data.x
        y_true = data.y
        logits = model(X_test, data.edge_index)
        y_pred = logits[mask].argmax(dim=1)  
        correct = (y_pred == y_true[mask]).sum().item()
        total = mask.sum().item()
        accuracy = correct / total
    return accuracy

#%% Training Loop
num_epochs = 200
loss_list = []
train_acc_list = []
val_acc_list = []

for epoch in range(num_epochs):
    epoch_loss = train(model, data, optimizer, criterion)
    train_acc = test(model, data, data.train_mask)
    val_acc = test(model, data, data.val_mask)

    loss_list.append(epoch_loss)
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)

    if (epoch + 1) % 20 == 0:
        print(f'Epoche: {epoch + 1:03d} | Verlust: {epoch_loss:.4f} | '
              f'Training Acc: {train_acc:.4f} | Validierungs-Accuracy: {val_acc:.4f}')

#%% Evaluation on the Test Set
final_test_acc = test(model, data, data.test_mask)
print(f'\nFinale Test-Accuracy: {final_test_acc:.4f}')

#%% Plotting the training loss and accuracy
# Plot the loss and accuracy curves during training
# Set up the plot style
sns.set_style("whitegrid")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot loss
sns.lineplot(data=loss_list, ax=ax1, color='blue', label='Training Loss')
ax1.set_ylabel('Verlust [-]')
ax1.set_title('Modell-Training-Performance')

# Plot accuracies
sns.lineplot(data=train_acc_list, ax=ax2, color='orange', linestyle='--', label='Train Accuracy')
sns.lineplot(data=val_acc_list, ax=ax2, color='green', linestyle='--', label='Validation Accuracy') 
ax2.set_xlabel('Epoche [-]')
ax2.set_ylabel('Accuracy [-]')
plt.tight_layout()
plt.show()
#%% visualize the embeddings
# Visualize the node embeddings using t-SNE
model.eval()
with torch.no_grad():
    # Get the final node embeddings (the output of the model)
    z = model(data.x, data.edge_index).detach().cpu().numpy()
    z_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(z)

# Visualize the embeddings using Seaborn
sns.set_style("white")
plt.figure(figsize=(10, 8))

# Color the nodes by their ground-truth class
sns.scatterplot(
    x=z_tsne[:, 0], y=z_tsne[:, 1],
    hue=data.y.cpu().numpy(),
    palette=sns.color_palette("hsv", n_colors=dataset.num_classes),
    s=50,
    alpha=0.8
)
plt.title('t-SNE Visualisierung der Knoten-Embeddings')
plt.xlabel('t-SNE Komponente 1')
plt.ylabel('t-SNE Komponente 2')
plt.legend(title='Klasse')
plt.show()