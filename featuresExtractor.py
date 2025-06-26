from models import InceptionFeaturesExtractor
import os
import numpy as np
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob
from PIL import Image

# Extract the features from the video frames (assuming that video frames already extracted)

class features:
    def __init__(self, sourcePath, destPath, diFeature) -> None:
        self.sourcePath = sourcePath
        self.destPath   = destPath
        self.diFeature = diFeature

    def extractFeatures(self):
        if self.diFeature['model'] == 'inception':
            model = InceptionFeaturesExtractor()
        else:
            raise ValueError('Not defined model !')
        
        print(model)

        # The dataset should have the signer\cat[train\test]\class\sample\sampleFrames

        for signer in os.listdir(self.sourcePath):
            signerFolder = os.path.join(self.sourcePath, signer)
            if os.path.isdir(signerFolder):
                signerFolderDest = os.path.join(self.destPath, signer)
                self.createFolder(signerFolderDest)

                for cat in os.listdir(signerFolder):
                    signCat = os.path.join(signerFolder, cat)
                    destCat = os.path.join(signerFolderDest, cat)
                    self.createFolder(destCat)

                    for sign in os.listdir(signCat): 
                        signFolder = os.path.join(signCat, sign)
                        signFolderDest = os.path.join(destCat, sign)
                        self.createFolder(signFolderDest)

                        # Get all subfolders in the sign folder
                        for subfolder in os.listdir(signFolder):
                            subfolder_path = os.path.join(signFolder, subfolder)
                            if os.path.isdir(subfolder_path):
                                print(f'Processing folder: {subfolder_path}')
                                
                                # Get all jpg files in the subfolder
                                jpg_files = sorted(glob.glob(os.path.join(subfolder_path, "*.jpg")))
                                
                                if not jpg_files:
                                    continue
                                
                                # Create npy file path directly in the sign folder
                                npy_file = os.path.join(signFolderDest, f"{subfolder}.npy")
                                
                                if os.path.exists(npy_file):
                                    continue
                                
                                # Initialize array to store features
                                all_features = []
                                
                                # Process each image in the folder
                                transform = transforms.Compose([
                                    transforms.Resize(self.diFeature['tuInputShape'][0:2]),
                                    transforms.ToTensor()
                                ])
                                
                                for jpg_file in tqdm(jpg_files, desc=f"Processing {subfolder}"):
                                    image = Image.open(jpg_file)
                                    image = transform(image).unsqueeze(0)  # Add batch dimension
                                    
                                    # Extract features
                                    features = model(image)
                                    features_np = features.detach().numpy()
                                    all_features.append(features_np)
                                
                                # Stack all features and save
                                if all_features:
                                    all_features = np.vstack(all_features)
                                    np.save(npy_file, all_features)

    def createFolder(self, folderPath):
        if os.path.exists(folderPath) == False:
            os.mkdir(folderPath)




