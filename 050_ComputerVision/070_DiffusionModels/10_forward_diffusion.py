#%% packages
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
#%%
def load_image(path, size=(64, 64)):
    """Loads and resizes an image, then normalizes it to [0, 1]."""
    image = Image.open(path).convert("RGB").resize(size)
    return np.array(image) / 255.0

def add_noise(image, noise_level):
    """Adds Gaussian noise to the image."""
    noise = np.random.normal(0, noise_level, image.shape)
    noised_image = image + noise
    return np.clip(noised_image, 0, 1)

def forward_diffusion(image, steps, max_noise=1.0):
    """Applies forward diffusion (adding noise step by step)."""
    for step in range(steps):
        noise_level = (step + 1) / steps * max_noise
        image = add_noise(image, noise_level)
    return image

#%% Test

# Path to your image file (use a small one to start!)
image_path = 'data/kiki.jpg'  # Change this!

original = load_image(image_path, size=(128, 128))
steps = 100  # Diffusion steps
noised = forward_diffusion(image=original, steps=steps, max_noise=0.2)

#%% Visualise results
sns.set_style("whitegrid")

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
with sns.axes_style("white"):
    axs[0].imshow(original)
    axs[0].set_title("Original", fontsize=14)
    axs[0].axis("off")
    
    axs[1].imshow(noised)
    axs[1].set_title(f"Noised after {steps} steps", fontsize=14)
    axs[1].axis("off")

sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.show()

# %%
