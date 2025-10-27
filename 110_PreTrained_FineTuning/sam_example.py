#%%
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

#%%
predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-tiny")
#%% load image
image_name = "AiFuture.jpg"
point_coords = np.array([[264, 395]])
point_labels = np.array([1], dtype=np.int32)  # 1 = foreground
#%%
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    image_path = os.path.join(os.path.dirname(__file__), image_name)
    image_rgb = np.array(Image.open(image_path).convert("RGB"))
    predictor.set_image(image_rgb)

    height, width = image_rgb.shape[:2]

    masks, _, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels)

# %% visualise image with masks
plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)  # Show original image

# Overlay mask with semi-transparent red color
mask_overlay = np.zeros_like(image_rgb, dtype=np.uint8)
mask_overlay[masks[1].astype(bool)] = [255, 0, 0]  # Red color for the mask
plt.imshow(mask_overlay, alpha=0.35)  # 0.5 transparency

# Plot point coordinates
plt.scatter(point_coords[:, 0], point_coords[:, 1], c='yellow', s=100, marker='*', label='Selected Point')

plt.axis('off')
plt.show()

# %%
