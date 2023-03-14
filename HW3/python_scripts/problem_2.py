#Part 1
import torch
import sys
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import List, Tuple

class ImageClassifier:
    """Image classifier based on a pre-trained PyTorch model.

    Args:
        model_name (str): Name of the pre-trained PyTorch model to use.
        pretrained (bool, optional): Whether to use pre-trained weights for the model. Defaults to True.
    """

    def __init__(self, model_name: str, pretrained: bool = True):
        """Initialize the ImageClassifier object.

        Loads the specified pre-trained PyTorch model and sets it to evaluation mode.

        Args:
            model_name (str): Name of the pre-trained PyTorch model to use.
            pretrained (bool, optional): Whether to use pre-trained weights for the model. Defaults to True.
        """
        # Load the model and set it to evaluation mode
        self.model = torch.hub.load('pytorch/vision:v0.9.0', model_name, pretrained=pretrained)
        self.model.eval()

        # Define the preprocessing steps to apply to the input image
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def classify_image(self, image_path: str, topk: int = 3) -> Tuple[List[str], List[float]]:
        """Classify an input image and return the top-k categories and their respective probabilities.

        Opens the input image at the specified path and applies the preprocessing steps
        defined in the ImageClassifier object. Passes the preprocessed image through the
        PyTorch model to obtain the output probabilities for each category. Returns the
        top-k categories and their respective probabilities.

        Args:
            image_path (str): Path to the input image file.
            topk (int, optional): Number of top categories to return. Defaults to 3.

        Returns:
            Tuple[List[str], List[float]]: Tuple containing two lists: the top-k categories and their respective probabilities.
        """
        # Open the input image and apply the preprocessing steps
        input_image = Image.open(image_path)
        org = Image.open(image_path)
        input_tensor = self.preprocess(org)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

        # Pass the preprocessed image through the model and obtain the output
        with torch.no_grad():
            output = self.model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Load the list of categories from the text file
        with open("imagenet_classes.txt", "r") as f:
            categories = [s.strip() for s in f.readlines()]

        # Obtain the top-k categories and their respective probabilities
        topk_prob, topk_catid = torch.topk(probabilities, topk)
        topk_categories = [categories[topk_catid[i]] for i in range(topk_prob.size(0))]
        topk_probabilities = [topk_prob[i].item() for i in range(topk_prob.size(0))]

        # Return the top-k categories and their respective probabilities as a tuple of two lists
        return topk_categories, topk_probabilities

#Calls fpr part 1

# Create an instance of the ImageClassifier class
classifier = ImageClassifier(model_name='resnet18')

# Classify an input image and print the top-k categories and their respective probabilities
image_path = "peppers.jpg"
topk_categories, topk_probabilities = classifier.classify_image(image_path)
for i in range(len(topk_categories)):
    print(topk_categories[i], topk_probabilities[i])



#Part 2
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from typing import List
import torch

class ConvolutionalLayerVisualizer:
    """
    A class to visualize the filters and feature maps of convolutional layers in a PyTorch model.
    """
    
    def __init__(self, model: nn.Module):
        """
        Initializes the ConvolutionalLayerVisualizer object.

        Args:
            model (nn.Module): The PyTorch model to visualize.
        """
        self.model = model
        self.model_weights = []
        self.conv_layers = []
        self.counter = 0
        self.device = 'cpu'  # default device
        
    def set_device(self, device: str):
        """
        Sets the device to use for computations (e.g. 'cpu' or 'cuda').

        Args:
            device (str): The name of the device.
        """
        self.device = device
        
    def get_conv_layers(self) -> List[nn.Conv2d]:
        """
        Finds all convolutional layers in the model and saves their weights and module objects.

        Returns:
            List[nn.Conv2d]: A list of all convolutional layers in the model.
        """
        model_children = list(self.model.children())
        for i in range(len(model_children)):
            if type(model_children[i]) == nn.Conv2d:
                self.counter += 1
                self.model_weights.append(model_children[i].weight)
                self.conv_layers.append(model_children[i])
            elif type(model_children[i]) == nn.Sequential:
                for j in range(len(model_children[i])):
                    for child in model_children[i][j].children():
                        if type(child) == nn.Conv2d:
                            self.counter += 1
                            self.model_weights.append(child.weight)
                            self.conv_layers.append(child)
        print(f"Total convolutional layers: {self.counter}")
        return self.conv_layers
        
    def plot_filters(self, layer: int, figsize: tuple = (20, 17)):
        """
        Plots the filters for a given convolutional layer.

        Args:
            layer (int): The index of the convolutional layer to visualize.
            figsize (tuple, optional): The size of the figure. Defaults to (20, 17).
        """
        plt.figure(figsize=figsize)
        for i, filter in enumerate(self.model_weights[layer]):
            plt.subplot(8, 8, i+1) 
            plt.imshow(filter[0, :, :].detach().cpu(), cmap='gray')
            plt.axis('off')
        plt.show()
        
    def visualize_layers(self, input_batch: torch.Tensor, figsize: tuple = (30, 30)):
        """
        Computes and plots the feature maps for all convolutional layers in the model.

        Args:
            input_batch (torch.Tensor): The input batch to feed through the model.
            figsize (tuple, optional): The size of the figure. Defaults to (30, 30).
        """
        input_batch = input_batch.to(self.device)
        outputs = [self.conv_layers[0](input_batch)]
        for i in range(1, len(self.conv_layers)):
            outputs.append(self.conv_layers[i](outputs[-1]))
        for idx in range(len(outputs)):
            plt.figure(figsize=figsize)
            layer_data = outputs[idx][0, :, :, :]
            layer_data = layer_data.data.cpu()
            print(layer_data.size()) # [64, 112, 112]
            for i, filter in enumerate(layer_data):
                if i == 64: 
                    break
                plt.subplot(8, 8, i + 1)
                plt.imshow(filter, cmap='gray')
                plt.axis("off")
            print(f"Saving layer {idx} feature maps...")
            plt.show()
        plt.close()

#Calls for part 2
from torchvision.models import resnet18

model = resnet18(pretrained=True)

visualizer = ConvolutionalLayerVisualizer(model)
visualizer.set_device('cuda')  # use GPU device
conv_layers = visualizer.get_conv_layers()

input_tensor = torch.rand(1, 3, 224, 224).to('cuda')
visualizer.plot_filters(0)
visualizer.visualize_layers(input_tensor)