#%%
from torchvision import transforms
from PIL import Image
# %% import image
img = Image.open('kiki.jpg')
img

# %% compose a series of steps
preprocess_steps = transforms.Compose([
    transforms.Resize((100, 100)),  # better (300, 300)
    # transforms.RandomRotation(90),
    # transforms.CenterCrop((300, 200)),
    transforms.Grayscale(),
    # transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # ImageNet values
])
x = preprocess_steps(img)
x.shape

# %% get the mean and std of given image
x.mean([1, 2]), x.std([1, 2])

#%% show the tensor as heatmap
import seaborn as sns
sns.heatmap(x.squeeze().numpy(), cmap='gray', linewidths=0.5, linecolor='black')
# %%
