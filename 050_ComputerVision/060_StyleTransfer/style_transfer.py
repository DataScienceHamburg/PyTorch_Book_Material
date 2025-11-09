
#%% packages
import numpy as np
from PIL import Image
import torch
from torch.optim import Adam
from torch.nn.functional import mse_loss
from torchvision import transforms, models
import matplotlib.pyplot as plt


#%% hyperparameters
EPOCHS = 500
CONTENT_WEIGHT = 1 # Or whatever factor you deem appropriate
STYLE_WEIGHT = 2000000
VGG_MEAN = [0.485, 0.456, 0.406]
VGG_STD = [0.229, 0.224, 0.225]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#%% 
vgg = models.vgg19(pretrained=True).features.to(DEVICE)
for param in vgg.parameters():
    param.requires_grad = False
vgg.eval()
# %% image transformations
preprocess_steps = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=VGG_MEAN, std=VGG_STD)
])
content_img = Image.open('hamburg.jpg').convert('RGB')
content_img = preprocess_steps(content_img)

content_img = torch.unsqueeze(content_img, 0).to(DEVICE)
print(content_img.shape)
#%%
style_img = Image.open('The_Great_Wave_off_Kanagawa.jpg').convert('RGB')
style_img = preprocess_steps(style_img)
style_img = torch.unsqueeze(style_img, 0).to(DEVICE)
print(style_img.shape)

#%%
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be un-normalized.
        Returns:
            Tensor: UnNormalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

unnormalize_transform = UnNormalize(mean=VGG_MEAN, std=VGG_STD)

def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unnormalize_transform(image)
    image.clamp_(0, 1)
    # Convert from (C,H,W) to (H,W,C) for matplotlib
    image = image.permute(1, 2, 0).numpy()
    return image
# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Show content image in first subplot
content_image = imshow(content_img)
ax1.imshow(content_image)
ax1.set_title('Reference Image')
ax1.axis('off')

# Show style image in second subplot
style_image = imshow(style_img)
ax2.imshow(style_image)
ax2.set_title('Style Image')
ax2.axis('off')

plt.show()

# %% feature extraction 
STYLE_LOSS_LAYERS = { '0': 'conv1_1', 
                '5': 'conv2_1',  
                '10': 'conv3_1', 
                '19': 'conv4_1', 
                '21': 'conv4_2', 
                '28': 'conv5_1'}
CONTENT_LOSS_LAYER = 'conv4_2'

def extract_features(x, model):
    features = {}   
    for name, layer in model._modules.items():
        x = layer(x)
        
        if name in STYLE_LOSS_LAYERS:
            features[STYLE_LOSS_LAYERS[name]] = x   
            
    return features

content_img_features = extract_features(content_img, vgg)
style_img_features   = extract_features(style_img, vgg)
# %%
def calc_gram_matrix(tensor):
    _, C, H, W = tensor.size()
    tensor = tensor.view(C, H * W)    
    gram_matrix = torch.mm(tensor, tensor.t())
    gram_matrix = gram_matrix.div(C * H * W)  # normalization required
    return gram_matrix

style_features_gram_matrix = {layer: calc_gram_matrix(style_img_features[layer]) for layer in style_img_features}


# %%
weights = {'conv1_1': 1.0, 'conv2_1': 0.8, 'conv3_1': 0.6,
           'conv4_1': 0.4, 'conv5_1': 0.2}

generated_image = content_img.clone().requires_grad_(True).to(DEVICE)

optimizer = Adam([generated_image], lr=0.003)
# %% train the model
for epoch in range(1, EPOCHS):
    
    target_features = extract_features(generated_image, vgg)

    content_loss = mse_loss(target_features[CONTENT_LOSS_LAYER], content_img_features[CONTENT_LOSS_LAYER])
    
    style_loss = 0
    for layer in weights:
  
        target_feature = target_features[layer]
        target_gram_matrix = calc_gram_matrix(target_feature)
        style_gram_matrix = style_features_gram_matrix[layer]
        
        layer_loss = mse_loss (target_gram_matrix, style_gram_matrix) * weights[layer]
        

        style_loss += layer_loss  
    
    total_loss = STYLE_WEIGHT * style_loss + CONTENT_WEIGHT * content_loss
    
    if epoch % 100 == 0:
        # save the output image
        output_image = imshow(generated_image.detach()) # Detach and unnormalize for display
        plt.imsave(f"output_epoch_{epoch}.jpg", output_image)

        
    optimizer.zero_grad()
    
    total_loss.backward()
    
    optimizer.step()

