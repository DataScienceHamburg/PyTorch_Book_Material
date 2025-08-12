#%%
import torch
from models.Model1 import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

#%% function to show model parameter distribution
def show_model_parameters(model):
    params = []
    for param in model.parameters():
        params.append(param.detach().numpy().flatten())
    g = sns.histplot(params, kde=True)
    # add title
    g.set_title('Model Parameter Distribution')
    g.set_xlabel('Parameter Value')
    g.set_ylabel('Frequency')
    return g

# %% create model instance
model = LinearRegression(37, 1)

#%% show model parameters
show_model_parameters(model)
#%% load model weights
model.load_state_dict(torch.load('models/Model1.pth'))
# %%
show_model_parameters(model)
