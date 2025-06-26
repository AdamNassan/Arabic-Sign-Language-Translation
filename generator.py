# %%
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import torchvision.transforms as transforms
import glob
from PIL import Image as pil
from sklearn.model_selection import train_test_split

# %%
# This cutomDataset to read images per class (not vidoe frames)
class CustomImageDataset(Dataset):
    def __init__(self, dataPath, transformer = None, fileExtention='jpg'):
        self.path = Path(dataPath)
        self.transform = transformer
        self.fileExt = fileExtention
        self.filenames = pd.DataFrame(sorted(glob.glob(dataPath + "/*/*/*."+fileExtention)), columns=["sPath"])
        self.labels = self.filenames.sPath.apply(lambda s: s.split("/")[-3]) 
        self.classes = sorted(list(self.labels.unique()))
        self.nClasses = len(self.classes)
        print(f'Found {len(self.filenames)} images belong to {self.nClasses} classes')
    
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        #print("Index: ", str(index))  
        #print(self.labels)
        
        image_filepath = self.filenames.iloc[index].sPath
        #print(image_filepath)
        image = pil.open(image_filepath)
        label = self.labels.iloc[index]
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image) 
        #vector = torch.from_numpy(np.load(fn))
        return image, label


# %%



# %%
# This cutomDataset to read images per class (not video frames) according to the csv list
class CustomCSVDataset(Dataset):
    def __init__(self, csvDataPath, captionToIndex, captions, transformer = None, cat ='train', split = 0.8, valAvailable = False):
        self.path = csvDataPath
        self.transform = transformer
        self.fileData = pd.read_csv(csvDataPath, dtype = str, names=["index","sentId","sPath","framesNo","signerID", "caption", "procCaption"])
        self.filenames = self.fileData.sPath.apply(self.append_ext)
        self.labels = self.fileData["sentId"]
        self.classes = sorted(list(self.labels.unique()))
        self.nClasses = len(self.classes)
        print(f'Found {len(self.filenames)} samples belong to {self.nClasses} classes')
        self.captionToIndex = captionToIndex
        self.captions = captions
        self.nFramesNorm = 80  # Normalized number of frames
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if valAvailable:
            train_data, val_data, y_train, y_val = train_test_split(self.filenames, self.labels, test_size=split, random_state=42, stratify=self.labels)   
            if cat == 'train':
                self.filenames = train_data
                self.labels = y_train
                self.filenames.reset_index(drop=True, inplace=True)
                print(f'Found {len(self.filenames)} {cat} samples with {len(self.labels)} labels')
            elif cat == 'val':
                self.filenames = val_data
                self.labels = y_val
                self.filenames.reset_index(drop=True, inplace=True)
                print(f'Found {len(self.filenames)} {cat} samples with {len(self.labels)} labels')
            else:
                print('Data cat is invalid !!!')

    def __len__(self):
        return len(self.filenames)
    
    def append_ext(self, filePath):
        return filePath + '.npy'
            
    def __getitem__(self, index):
        sample_filepath = self.filenames[index]
        data = np.load(sample_filepath)
        label = self.labels.iloc[index]

        # Ensure data has exactly nFramesNorm frames
        if data.shape[0] < self.nFramesNorm:
            # Pad with zeros if fewer frames
            pad_size = self.nFramesNorm - data.shape[0]
            data = np.pad(data, ((0, pad_size), (0, 0)), mode='constant')
        elif data.shape[0] > self.nFramesNorm:
            # Truncate if more frames
            data = data[:self.nFramesNorm]

        # Convert data to tensor and move to GPU
        data = torch.from_numpy(data).float().to(self.device)

        if self.transform is not None:
            data = self.transform(data)

        try:
            tokenized_caption = self.captions[int(label) - 1]
        except:
            print(f'Error processing label {label}')
            print('Captions:', self.captions)
            print('Captions shape:', len(self.captions))
            raise

        mapped_caption = []
        for tok in tokenized_caption:
            mapped_caption.append(self.captionToIndex[tok])

        # Convert caption to tensor and move to GPU
        return data, torch.tensor(mapped_caption, device=self.device)


'''
# %%
print('hi')
transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor()
                        ])
train_image_paths = "/home/eye/ArSL-Continuous/80/color/01/test"    
train_dataset = CustomImageDataset(train_image_paths,transform)
#print('The shape of tensor for 50th image in train dataset: ',train_dataset[49][0].shape)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#batch of image tensor
next(iter(train_loader))[0].shape
'''
'''
# %%
transform = None
train_image_paths = "/home/eye/ArSL-Continuous/80/features/images/vgg/color/01.csv"    
train_dataset = CustomCSVDataset(train_image_paths,transform)
#print('The shape of tensor for 50th image in train dataset: ',train_dataset[49][0].shape)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#batch of image tensor
next(iter(train_loader))[0].shape

# %%


'''