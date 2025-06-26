import torch
import torch.nn as nn
import torchvision

class VGG16FeaturesExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.vgg16 = torchvision.models.vgg16(pretrained=True)
        self.fine_tune()
    
    
    def forward(self, x):
        # Shape of x: (batch_size, channels, height, width)
        x = self.vgg16(x)
        return x


    def fine_tune(self):
        for param in self.vgg16.parameters():
            param.requires_grad = False
        
        self.vgg16.classifier = nn.Sequential(*[self.vgg16.classifier[i] for i in range(4)]) # Keeping only till classifier(3) layer. 

class MobileNetFeaturesExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mobilenet = torchvision.models.mobilenet_v2(pretrained=True)
        self.fine_tune()
    
    def forward(self, x):
        # Shape of x: (batch_size, channels, height, width)
        x = self.mobilenet(x)
        return x

    def fine_tune(self):
        for param in self.mobilenet.parameters():
            param.requires_grad = False
        
        # Keep only the features part, remove the classifier
        self.mobilenet.classifier = nn.Sequential()

class InceptionFeaturesExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Inception v3 requires aux_logits=True during initialization
        self.inception = torchvision.models.inception_v3(pretrained=True, aux_logits=True)
        self.fine_tune()
        # Set to evaluation mode to disable batch norm updates
        self.inception.eval()
    
    def forward(self, x):
        # Shape of x: (batch_size, channels, height, width)
        # Inception v3 expects input size of 299x299
        if x.size(2) != 299 or x.size(3) != 299:
            x = torch.nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Disable gradient computation and batch norm updates
        with torch.no_grad():
            # Get both main and auxiliary outputs
            if self.inception.training:
                x, aux = self.inception(x)
            else:
                x = self.inception(x)
            # We only need the main output
            return x

    def fine_tune(self):
        for param in self.inception.parameters():
            param.requires_grad = False
        
        # Keep only the features part, remove the classifier
        self.inception.fc = nn.Sequential()
