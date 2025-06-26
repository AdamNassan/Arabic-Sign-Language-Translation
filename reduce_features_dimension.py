import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

class FeatureDimensionReducer:
    def __init__(self, sourcePath, destPath):
        self.sourcePath = sourcePath
        self.destPath = destPath
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Create average pooling layer to reduce from 2048 to 1024
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def reduce_dimensions(self):
        # The dataset should have the signer\cat[train\test]\class\sample.npy structure
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

                        # Process all .npy files in the sign folder
                        npy_files = [f for f in os.listdir(signFolder) if f.endswith('.npy')]
                        
                        for npy_file in tqdm(npy_files, desc=f"Processing {os.path.join(signer, cat, sign)}"):
                            source_path = os.path.join(signFolder, npy_file)
                            dest_path = os.path.join(signFolderDest, npy_file)
                            
                            if os.path.exists(dest_path):
                                continue
                            
                            # Load features
                            features = np.load(source_path)
                            
                            # Convert to tensor
                            features_tensor = torch.from_numpy(features).to(self.device)
                            
                            # Reshape for 1D average pooling (batch_size, channels, length)
                            features_tensor = features_tensor.view(features_tensor.shape[0], 1, -1)
                            
                            # Apply average pooling
                            reduced_features = self.pool(features_tensor)
                            
                            # Reshape back to original format
                            reduced_features = reduced_features.squeeze(1)
                            
                            # Convert back to numpy and save
                            reduced_features_np = reduced_features.cpu().numpy()
                            np.save(dest_path, reduced_features_np)

    def createFolder(self, folderPath):
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)

if __name__ == "__main__":
    # Update these paths according to your setup
    source_path = "features/color"  # folder containing 2048D features
    dest_path = "features/color_1024d"  # folder where 1024D features will be saved
    
    reducer = FeatureDimensionReducer(source_path, dest_path)
    reducer.reduce_dimensions()
    print("Feature dimension reduction completed!")
