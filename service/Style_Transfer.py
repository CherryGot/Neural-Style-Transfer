from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import torch.optim as optim
from torchvision import transforms, models
import torchvision



parser = argparse.ArgumentParser()
parser.add_argument("--content", type=str, required=True, help="Path to content image")
parser.add_argument("--style", type=str, required=True, help="Path to style image")
parser.add_argument("--epochs",type=int, required=True, help="No of steps.")
args = parser.parse_args()

vgg = models.vgg19(pretrained=True).features
for param in vgg.parameters():
    param.requires_grad_(False)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device=torch.device("cpu")
vgg.to(device)

def load_image(img_path, max_size=512, shape=None):
    ''' Load in and transform an image, making sure the image
       is <= 400 pixels in the x-y dims.'''
    
    image = Image.open(img_path).convert('RGB')
    
    # large images will slow down processing
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape
        
    in_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor()])

    # discard the transparent, alpha channel and add the batch dimension
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    
    return image



# load in content and style image
content = load_image(args.content).to(device)
# Resize style to match content, makes code easier
style = load_image(args.style, shape=content.shape[-2:]).to(device)


def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1', 
                  '10': 'conv3_1', 
                  '19': 'conv4_1',
                  '21': 'conv4_2',  #This layer is for content representation
                  '28': 'conv5_1'}
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
            
    return features

def gram_matrix(tensor):
    
    _, d, h, w = tensor.size()
    
    tensor = tensor.view(d, h * w)
    
    gram = torch.mm(tensor, tensor.t())
    
    return gram 

content_features = get_features(content, vgg)
style_features = get_features(style, vgg)
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

target = content.clone().requires_grad_(True).to(device)

style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.675,
                 'conv3_1': 0.55,
                 'conv4_1': 0.325,
                 'conv5_1': 0.1}

content_weight = 1  # alpha
style_weight = 1e-3  # beta


# iteration hyperparameters
#optimizer = optim.Adam([target], lr=0.1)
optimizer = optim.LBFGS([target], lr=0.1)
steps = args.epochs
for ii in range(1, steps+1):
    
    # get the features from your target image

    def closure():
        target_features = get_features(target, vgg)
        optimizer.zero_grad()
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
    
        style_loss = 0
        for layer in style_weights:
        # get the "target" style representation for the layer
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            _, d, h, w = target_feature.shape
        # get the "style" style representation
            style_gram = style_grams[layer]
        # the style loss for one layer, weighted appropriately
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
        # add to the style loss
            style_loss += layer_style_loss / (d * h * w)
        total_loss = content_weight * content_loss + style_weight * style_loss
        total_loss.backward()
        return total_loss
    
    # update your target image
    #optimizer.zero_grad()
    #total_loss.backward()
    #optimizer.step()
    optimizer.step(closure)
torchvision.utils.save_image(target[0],"output.jpg")





