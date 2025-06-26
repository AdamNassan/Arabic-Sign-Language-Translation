#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchmetrics import WordErrorRate as WER
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import random
import codecs
import nltk
import glob

from generator import CustomCSVDataset
import encoderDecoderModel as EncDecModel
import attentionModel as encDecMode

# Force CUDA usage if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("Using CPU for computation")

def freeze_encoder_layers(encoder, num_layers_to_freeze):
    """Freeze the first num_layers_to_freeze layers of the encoder."""
    for name, param in encoder.named_parameters():
        if 'lstm.weight_ih_l0' in name or 'lstm.weight_hh_l0' in name or 'lstm.bias_ih_l0' in name or 'lstm.bias_hh_l0' in name:
            param.requires_grad = False
    print(f"Froze first {num_layers_to_freeze} layers of encoder")

def unfreeze_encoder_layers(encoder):
    """Unfreeze all encoder layers."""
    for param in encoder.parameters():
        param.requires_grad = True
    print("Unfroze all encoder layers")

def get_optimizer_with_different_lrs(model, encoder_lr=1e-4, decoder_lr=1e-3):
    """Create optimizer with different learning rates for encoder and decoder."""
    encoder_params = list(map(id, model.encoder.parameters()))
    decoder_params = filter(lambda p: id(p) not in encoder_params, model.parameters())
    
    optimizer = optim.Adam([
        {'params': model.encoder.parameters(), 'lr': encoder_lr},
        {'params': decoder_params, 'lr': decoder_lr}
    ])
    return optimizer

def FindMaxLength(lst):
    maxList = max(lst, key = lambda i: len(i))
    maxLength = len(maxList)
    return maxLength

def readCaptions(filepath):
    # Add all the captions.
    captions = []
    f = open(filepath, "r", encoding='utf-8')
    for line in f.readlines():
        lineWords = line.split()
        lineWords.insert(0, '<SOS>')
        lineWords.insert(len(lineWords), '<EOS>')
        captions.append(lineWords)


    sentenceMaxLengh = FindMaxLength(captions)
    print(sentenceMaxLengh)    
    
    # padding
    paddedCaptions = []
    for caption in captions:
        if len(caption) < sentenceMaxLengh:
            caption = caption + ['<PAD>'] * (sentenceMaxLengh - len(caption))
        paddedCaptions.append(caption)
    print(paddedCaptions)
    captions = paddedCaptions
    
    vocab = set() # Total unique words including <SOS>, <EOS>, <PAD> forms the vocab. 
    for caption in captions:
        print(caption, f'len = {len(caption)}')
        for token in caption:
            vocab.add(token)
    print(f'\nVocab:\n{vocab} len = {len(vocab)}')

    # Mapping string/word to an index.
    captionToIndex = {
    '<PAD>': 0,
    '<SOS>': 1,
    '<EOS>': 2,
    }

    temp = {}
    idx = 3 # Since indices 0,1,2 are already reversed for tokens <PAD>, <SOS>, <EOS> respectively.
    for caption in captions:
        for tok in caption:
            if tok not in ['<PAD>', '<SOS>', '<EOS>'] and tok not in temp:
                temp[tok] = idx
                idx += 1
                
    captionToIndex.update(temp)
    print(f'\nString-to-index mapping:\n{captionToIndex}\n')

    # Mapping index to string/word.
    indexToCaption = {value : key for (key, value) in captionToIndex.items()}
    print(f'\nIndex-to-string mapping:\n{indexToCaption}\n')

    return captions, captionToIndex, indexToCaption






class VideoCaptionDataset(Dataset):
    def __init__(self, dataPath, captionToIndexndex, captions) -> None:
        #super().__init__()
        self.path = dataPath #Path(dataPath)
        self.filenames = pd.DataFrame(sorted(glob.glob(dataPath + "/*/*.npy")), columns=["sPath"])
        self.sentenceIndices = self.filenames.sPath.apply(lambda s: s.split("/")[-2]) 
        self.classes = sorted(list(self.sentenceIndices.unique()))
        self.nClasses = len(self.classes)
        
        print(f'Found {len(self.filenames)} files belong to {self.nClasses} sentences')
    
        #self.df = train_df
        self.captionToIndex = captionToIndexndex
        self.captions = captions


    def __len__(self):
        return len(self.filenames)
    

    def __getitem__(self, index):
        # Reshaping the data to (frames, extracted_features) i.e. (80, 4096) in our case.
        sample_filepath = self.filenames.iloc[index].sPath
        #print(sample_filepath)
        sample = np.load(sample_filepath)
        sample = torch.tensor(sample).float()
        
        ##sample = self.df.iloc[index, :-1].to_numpy().reshape(80, 4096) 
        ##sample = torch.tensor(sample).float()
        
        # Label corresponds to the last column of the df. This is just a number from 1.0 - 9.0 with 0.0 signifying label 10.0
        #label = self.df.iloc[index, -1]
#         if label == 0.0:
#             label = 10.0
        label = self.sentenceIndices.iloc[index] 
        
        tokenized_caption = self.captions[int(label) - 1]
        
        mapped_caption = []
        # Convert the sentences to their mapping through captionToIndex.
        for tok in tokenized_caption:
            mapped_caption.append(captionToIndex[tok])
        #print(mapped_caption)
        return sample, torch.tensor(mapped_caption)




def start_training(train_loader, expPath, input_size_encoder, modelName, captionToIndex, indexToCaption, valLoader=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    LR = 1e-3
    WD = 0#1e-4
    PATIENCE = 5
    EPOCHS = 50#150#100#150

    # Inputs for the encoder, decoder & encoder-decoder combined model.
    #input_size_encoder = 4096
    hidden_size =  512 #1024
    num_layers = 3  
    dropout_encoder = 0.4 #0.4
    num_embeddings = 150 #45
    embedding_dim = 300
    input_size_decoder = 300 
    dropout_decoder = 0.4 #0.1
    vocab_size =150#45













    if modelName == "EncoderDecoder":
        encoder = EncDecModel.EncoderRNN(input_size_encoder, hidden_size, num_layers, dropout_encoder).to(device)
        decoder = EncDecModel.DecoderRNN(num_embeddings, embedding_dim, input_size_decoder, hidden_size, num_layers, dropout_decoder, vocab_size).to(device)
        encoder_decoder = EncoderDecoder(encoder, decoder, vocab_size, modelName).to(device)
        
        # Load pretrained weights if available
        pretrained_path = 'checkpoints/phoenix_best.pt'
        if os.path.exists(pretrained_path):
            print("Loading pretrained weights...")
            try:
                checkpoint = torch.load(pretrained_path)
                if 'model_state_dict' in checkpoint:
                    encoder_decoder.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    print("Successfully loaded pretrained weights")
                    

                    
                    # Freeze early encoder layers and use different learning rates
                    freeze_encoder_layers(encoder_decoder.encoder, 1)  # Freeze first layer
                    optimizer = get_optimizer_with_different_lrs(encoder_decoder, encoder_lr=1e-4, decoder_lr=1e-3)
                    print("Using transfer learning strategy: Frozen encoder layers and different learning rates")
                else:
                    print("Could not find model weights in checkpoint")
                    optimizer = optim.Adam(encoder_decoder.parameters(), lr=LR)
            except Exception as e:
                print(f"Error loading pretrained weights: {e}")
                optimizer = optim.Adam(encoder_decoder.parameters(), lr=LR)
        else:
            optimizer = optim.Adam(encoder_decoder.parameters(), lr=LR)
    
    elif modelName == "EncoderDecoderAttention":
        print('--------------------------- Attention ------------------------')
        encoder = encDecMode.EncoderRNN(input_size_encoder, hidden_size, num_layers, dropout_encoder).to(device)
        decoder = encDecMode.DecoderRNN(num_embeddings, embedding_dim, input_size_decoder, hidden_size, num_layers, dropout_decoder, vocab_size).to(device)
        encoder_decoder = EncoderDecoder(encoder, decoder, vocab_size, modelName).to(device)
        optimizer = optim.Adam(encoder_decoder.parameters(), lr=LR)
    else:
        raise Exception('No defined model !!')    
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.01, patience=PATIENCE, verbose=True)

    loss_fn = nn.CrossEntropyLoss(ignore_index=0).to(device) # We ignore the index of <PAD> which is 0.
    train_losses = []
    train_WER = []
    val_WER = []
    # Training phase.
    for epoch in range(1, EPOCHS+1):
        encoder_decoder.train()
        
        # Gradual unfreezing at specific epochs if using pretrained weights
        if os.path.exists('checkpoints/phoenix_best.pt'):
            if epoch == EPOCHS // 3:  # Unfreeze at 1/3rd of training
                unfreeze_encoder_layers(encoder_decoder.encoder)
                print("Unfroze all encoder layers")
                # Adjust learning rate after unfreezing
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1
                print("Reduced learning rates by 10x after unfreezing")
        
        running_loss = 0.0
        trainWER = 0.0
        prog_bar = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
        for batch_idx, (img_frames, captions) in prog_bar:   
            optimizer.zero_grad() 
            
            img_frames = img_frames.permute(1, 0, 2) # Reshape into (frames, batch_size, features) i.e. (80, batch_size, 4096)
            captions = captions.permute(1, 0) # Reshap into (seq_len, batch_size) i.e. (14, batch_size)
            
            # Move tensors to the same device
            img_frames = img_frames.to(device)
            captions = captions.long().to(device)
            
            predicted_captions = encoder_decoder(img_frames, captions)
            
            # We dont want the <SOS> token. Hence, we take from the first word/token.
            predicted_captions = predicted_captions[1:].reshape(-1, predicted_captions.shape[2])
            captions = captions[1:].reshape(-1)
            
            loss = loss_fn(predicted_captions, captions) 
            
            running_loss += loss.item()
        
            loss.backward()
            optimizer.step()
            
            prog_bar.set_description(f'Epoch {epoch}/{EPOCHS}')
        
        tr_loss = running_loss / len(train_loader)
        train_losses.append(tr_loss)
        print(f'\ttrain_loss = {tr_loss:.6f}')
        trainWER = 0
        train_WER.append(trainWER)

        # validation WER  
        if valLoader != None:
            valWER, _, _, val_loss = eval_model(valLoader, encoder_decoder, captionToIndex, indexToCaption, modelName, vocab_size)
            print(f'\t Val loss = {val_loss}, Val WER = {valWER:.3f}')
            val_WER.append([val_loss, valWER])
        else:
            val_WER.append([0, 0])  # Ensure consistent shape

    # Save metrics
    train_metrics = np.column_stack((train_losses, train_WER, np.array(val_WER)))
    saveNumpyToCSV(train_metrics, os.path.join(expPath, "Metrics.csv"))
    torch.save(encoder_decoder, os.path.join(expPath, "model.pt"))
    return encoder_decoder

def blue(candidates, reference):
    score_1 = 0.0
    score_2 = 0.0
    score_3 = 0.0
    score_4 = 0.0
    n = len(candidates)
    for cand, ref in zip(candidates, reference):
        cand = cand.split(' ')
        cand = ' '.join(cand).split()
        # c = len(cand)
        #print(cand)
        ref = ref.split(' ')
        ref = ' '.join(ref).split()
        #print(ref)
        # r = len(ref)
        score_1 += nltk.translate.bleu_score.sentence_bleu([ref], cand, weights=(1, 0, 0, 0))
        score_2 += nltk.translate.bleu_score.sentence_bleu([ref], cand, weights=(0, 1, 0, 0))
        score_3 += nltk.translate.bleu_score.sentence_bleu([ref], cand, weights=(0, 0, 1, 0))
        score_4 += nltk.translate.bleu_score.sentence_bleu([ref], cand, weights=(0, 0, 0, 1))
        # print(f"Bleu-1 score {score_1 :.2f}")
        # print(f"Bleu-2 score {score_2 :.2f}")
        # print(f"Bleu-3 score {score_3 :.2f}")
        # print(f"Bleu-4 score {score_4 :.2f}")

    return score_1/n, score_2/n, score_3/n,score_4/n

# In[76]:
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, vocab_size, modelName) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size
        self.modelName = modelName
        

    def forward(self, img_frames, captions, teacher_force_ratio=0.5):
        # Shape of img_frames: (frames, batch_size, input_size) i.e. (80, batch_size, 4096)
        # Shape of captions: (seq_len, batch_size) i.e. (14, batch_size)

        seq_len, batch_size = captions.size()
        
        if self.modelName =='EncoderDecoder':
            enc_output, hidden  = self.encoder(img_frames)
        else:
            enc_output, hidden = self.encoder(img_frames)

        outputs = torch.zeros(seq_len, batch_size, self.vocab_size).to(device)
        #print('captions main: ', captions.shape)
        x = captions[0] # Grab the start token in the batch i.e. the <SOS> token whose index is 1.
        #####hidden = hidden[:self.encoder.num_layers]
        #print('hidden main: ', hidden.shape)

        for t in range(1, seq_len):
            # Use previous hidden, cell as context from encoder at start i.e. use enc_hidden & enc_cell.
            if self.modelName =='EncoderDecoder':
                predictions, hidden  = self.decoder(x, hidden)
            else:
                predictions, hidden = self.decoder(x, hidden, enc_output)

            # Store the prediction.
            outputs[t] = predictions

            # Get the best word the decoder predicted (index in the vocabulary)
            try:
                best_guess = predictions.argmax(1)
            except:
                best_guess = predictions.argmax(0)

            
           
            # With probability of teacher_force_ratio we take the actual next word
            # otherwise we take the word that the Decoder predicted it to be.
            x = captions[t] if random.random() < teacher_force_ratio else best_guess

        return outputs






def remove_special_tokens(sentence):
    cleaned_sent = ''
    tokenized_sent = sentence.split()
    for tok in tokenized_sent:
        if tok not in ['<PAD>', '<SOS>', '<EOS>']:
            cleaned_sent += tok + ' '
    return cleaned_sent


def decode_sentence(sentence, indexToCaption):
    sent = ''
    for idx in sentence:
        if isinstance(idx, torch.Tensor):
            sent += indexToCaption[idx.item()] + ' '
        else:
            sent += indexToCaption[idx] + ' '
    
    sent = remove_special_tokens(sent)
    return sent

def get_caption_encoderDecoder(model, img_frame, true_caption, captionToIndex, indexToCaption, vocab_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Shape of img_frame: (frames, features) i.e. (80, 4090). Hence, we will reshape it into (80, 1, 4090)
    # Shape of true_caption: (seq_len,) i.e. (14,). Hence, we will reshape it into (14, 1)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0).to(device)
    img_frame = img_frame.unsqueeze(1) # Now shape becomes (80, 1, 4090)
    true_caption = true_caption.unsqueeze(1) # Now shape becomes (14, 1)
    all_predicted = []
    model.eval()
    with torch.no_grad():
        img_frame, true_caption = img_frame.to(device), true_caption.long().to(device)
        en_output, hidden = model.encoder(img_frame)

        seq_len, batch_size = true_caption.size()

        predicted_caption = [captionToIndex['<SOS>']]
        for t in range(1, seq_len):
            x = torch.LongTensor([predicted_caption[-1]]).to(device)

            prediction, hidden = model.decoder(x, hidden)
            # Ensure prediction is within vocabulary range
            prediction[:, vocab_size:] = float('-inf')  # Set probability of invalid indices to -inf
            best_guess = prediction.argmax(1).item()
            best_guess = min(best_guess, vocab_size - 1)  # Clamp to valid range
            all_predicted.append(prediction)
            predicted_caption.append(best_guess)

    caption = true_caption.to('cpu').numpy()
    caption = caption[1:].reshape(-1) 
    caption = torch.Tensor(caption).to(device)
    prediction = torch.cat(all_predicted, dim=0)
    loss = loss_fn(prediction, true_caption[1:].squeeze(1))   
    predicted_caption_decoded = decode_sentence(predicted_caption, indexToCaption)
    
    return predicted_caption_decoded, loss.item()

def get_caption(model, img_frame, true_caption, captionToIndex, indexToCaption, vocab_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = nn.CrossEntropyLoss(ignore_index=0).to(device)
    img_frame = img_frame.unsqueeze(1) # Now shape becomes (80, 1, 4090)
    true_caption = true_caption.unsqueeze(1) # Now shape becomes (14, 1)
    all_predicted = []
    model.eval()
    with torch.no_grad():
        img_frame, true_caption = img_frame.to(device), true_caption.long().to(device)
        hidden, cell = model.encoder(img_frame)

        seq_len, batch_size = true_caption.size()

        predicted_caption = [captionToIndex['<SOS>']]
        for t in range(1, seq_len):
            x = torch.LongTensor([predicted_caption[-1]]).to(device)

            prediction, hidden = model.decoder(x, hidden)
            
            # Ensure prediction is within vocabulary range
            if len(prediction.shape) > 1:
                prediction[:, vocab_size:] = float('-inf')
                best_guess = prediction.argmax(1).item()
            else:
                prediction[vocab_size:] = float('-inf')
                best_guess = prediction.argmax(0).item()
            
            # Clamp to valid range
            best_guess = min(best_guess, vocab_size - 1)
            
            all_predicted.append(prediction)
            predicted_caption.append(best_guess)
    
    caption = true_caption.to('cpu').numpy()
    caption = caption[1:].reshape(-1) 
    caption = torch.Tensor(caption).to(device)
    prediction = torch.cat(all_predicted, dim=0)
    loss = loss_fn(prediction, true_caption[1:].squeeze(1))  
    predicted_caption_decoded = decode_sentence(predicted_caption, indexToCaption)
    
    return predicted_caption_decoded, loss.item()






def eval_model(test_loader, model, captionToIndex, indexToCaption, modelName, vocab_size=150, printPredictions=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wer = WER()
    model.eval()

    all_predicted = []
    all_true = []
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch_idx, (img_frames, captions) in enumerate(test_loader):
            for img_f, cap in zip(img_frames, captions):
                if modelName =='EncoderDecoder':
                    predicted_caption, loss = get_caption_encoderDecoder(model, img_f, cap, captionToIndex, indexToCaption, vocab_size)
                else:
                    predicted_caption, loss = get_caption(model, img_f, cap, captionToIndex, indexToCaption, vocab_size)
   
                true_caption = decode_sentence(cap, indexToCaption)
                all_true.append(true_caption)
                all_predicted.append(predicted_caption)
                total_loss += loss
                count += 1
    
    if printPredictions:
        for true_c, pred_c in zip(all_true, all_predicted):
            print(f'ACTUAL:    {true_c}')
            print(f'PREDICTED: {pred_c}')
            print()
    
    error_rate = wer(all_predicted, all_true).item()
    total_loss = total_loss / count
    return error_rate, all_true, all_predicted, total_loss






##
def saveNumpyToCSV(data, filePath):
    pd.DataFrame(data).to_csv(filePath, sep=',')

def createFolder(folderPath):
    if os.path.exists(folderPath) ==  False:
        os.mkdir(folderPath)
def savePredictionsToTxt(data, filePath):
    with codecs.open(filePath, 'w', encoding='utf-8') as outfile:
        for x, y in data:
            outfile.write(x + "\t" + y +"\n")
    

def startExperiment(expPath, trainPath, testPath, groundTruth, nFramesNorm, diFeature, modelName):
    # Read caption/ground truth
    captions, captionToIndex, indexToCaption = readCaptions(groundTruth)

    # Load data
    trainDataset = CustomCSVDataset(trainPath, captionToIndex, captions, None,'train', 0.2, True)    
    trainLoader = DataLoader(trainDataset, batch_size=32, shuffle=True)
    print('Train data size:', next(iter(trainLoader))[0].shape)

    valDataset = CustomCSVDataset(trainPath, captionToIndex, captions, None, 'val', 0.2, True)    
    valLoader = DataLoader(valDataset, batch_size=32, shuffle=True)   
    print('val data size:', next(iter(valLoader))[0].shape)

    testDataset = CustomCSVDataset(testPath, captionToIndex, captions)
    testLoader = DataLoader(testDataset, batch_size=32, shuffle=True)

    ## create model and start training
    model = start_training(trainLoader, expPath, diFeature["tuOutputShape"][0], modelName, captionToIndex, indexToCaption, valLoader )

    # model test
    wordErrorRate, all_true, all_predicted, test_loss = eval_model(testLoader, model, captionToIndex, indexToCaption, modelName)
    pred_Truth = np.transpose([all_true, all_predicted])
    print(pred_Truth)
    savePredictionsToTxt(pred_Truth, os.path.join(expPath, "prediction.txt"))
    #######pred_Truth_frame = pd.DataFrame(pred_Truth, columns = ['True','Predicted'])
    #######pd.DataFrame(pred_Truth_frame).to_csv(os.path.join(expPath, "prediction.csv"), sep='\t', encoding='utf-8')
    #pred_Truth.savetxt(os.path.join(expPath, "prediction.txt"),  delimiter='\t')
    #saveNumpyToCSV(zip(all_true, all_predicted), os.path.join(expPath, "prediction.csv"),encoding='utf8')
    print(f'Test Loss = {test_loss}, Test WER = {wordErrorRate}') 
    f= open(os.path.join(expPath, 'testWER.txt'),"a")
    #f.write(' '.join(expInfo)+'\n')
    f.write('WER:\t' +str(wordErrorRate)  + '\n')

    score_1, score_2, score_3,score_4 = blue(all_predicted, all_true)
    print(f"Bleu-1 score {score_1 :.2f}")
    print(f"Bleu-2 score {score_2 :.2f}")
    print(f"Bleu-3 score {score_3 :.2f}")
    print(f"Bleu-4 score {score_4 :.2f}")

    f.write(f"Bleu-1 score {score_1 :.2f}"+ '\n')
    f.write(f"Bleu-2 score {score_2 :.2f}"+ '\n')
    f.write(f"Bleu-3 score {score_3 :.2f}"+ '\n')
    f.write(f"Bleu-4 score {score_4 :.2f}"+ '\n')



    f.close()
## 
diVideoSet = {"dataset" : "ArabSign",
    "modelName": "EncoderDecoder", #EncoderDecoderAttention EncoderDecoder
    "nClasses" : 50,   # number of classes
    "nFramesNorm" : 80,    # number of frames per video
    "nMinDim" : 299,   # smaller dimension of saved video-frames
    "tuShape" : (299, 299), # height, width
    "nFpsAvg" : 30,
    "nFramesAvg" : 50, 
    "fDurationAvg" : 2.0,# seconds 
    "reshape_input": False}  #True: if the raw input is different from the requested shape for the model

  
# feature extractor 
diFeature = {"sName" : "inception",
    "tuInputShape" : (299, 299, 3),
    "tuOutputShape" : (1024, )}  # Inception output features size

#diFeature = {"sName" : "vgg16",
#     "tuInputShape" : (224, 224, 3),
#    "tuOutputShape" : (4096, )} # was 1024

# diFeature = {"sName" : "mobilenet",
#     "tuInputShape" : (224, 224, 3),
#     "tuOutputShape" : (1280, )} # was 1024


#dataSetHomePath = '/home/eye/ArSL-Continuous/80/features/images/mobilenet/color'
dataSetHomePath = 'data'
dataSetHomePathIndependent = 'data/independent'

#dataSetHomePath = '/home/eye/ArSL-Continuous/80/features/images/inception/color'
groundTruth = './data/groundTruth.txt'
#trainPath_all = [dataSetHomePath+'/test_01.csv', dataSetHomePath+'/test_02.csv', dataSetHomePath+'/test_03.csv', dataSetHomePath+'/test_04.csv', dataSetHomePath+'/test_05.csv', dataSetHomePath+'/test_06.csv']
#testPath_all = [dataSetHomePath+'/01.csv', dataSetHomePath+'/02.csv', dataSetHomePath+'/03.csv', dataSetHomePath+'/04.csv', dataSetHomePath+'/05.csv', dataSetHomePath+'/06.csv']

#trainPath_all = [dataSetHomePath+'/01_train.csv', dataSetHomePath+'/02_train.csv', dataSetHomePath+'/03_train.csv', dataSetHomePath+'/04_train.csv', dataSetHomePath+'/05_train.csv', dataSetHomePath+'/06_train.csv', dataSetHomePath+'/all_train.csv']
#testPath_all = [dataSetHomePath+'/01_test.csv', dataSetHomePath+'/02_test.csv', dataSetHomePath+'/03_test.csv', dataSetHomePath+'/04_test.csv', dataSetHomePath+'/05_test.csv', dataSetHomePath+'/06_test.csv', dataSetHomePath+'/all_test.csv']

# trainPath_all = [dataSetHomePath+'/train_01.csv',dataSetHomePath+'/train_02.csv',dataSetHomePath+'/train_03.csv', dataSetHomePath+'/train_04.csv', dataSetHomePath+'/train_05.csv', dataSetHomePath+'/train_06.csv']
# testPath_all = [dataSetHomePath+'/01.csv', dataSetHomePath+'/02.csv', dataSetHomePath+'/03.csv', dataSetHomePath+'/04.csv', dataSetHomePath+'/05.csv', dataSetHomePath+'/06.csv']

#signer dependent
#trainPath_all = [dataSetHomePath+'/01_train.csv',dataSetHomePath+'/02_train.csv',dataSetHomePath+'/03_train.csv', dataSetHomePath+'/04_train.csv', dataSetHomePath+'/05_train.csv', dataSetHomePath+'/06_train.csv', dataSetHomePath+'/all_train.csv',
#dataSetHomePath+'/train_01.csv',dataSetHomePath+'/train_02.csv',dataSetHomePath+'/train_03.csv', dataSetHomePath+'/train_04.csv', dataSetHomePath+'/train_05.csv', dataSetHomePath+'/train_06.csv']

#testPath_all = [dataSetHomePath+'/01_test.csv', dataSetHomePath+'/02_test.csv', dataSetHomePath+'/03_test.csv', dataSetHomePath+'/04_test.csv', dataSetHomePath+'/05_test.csv', dataSetHomePath+'/06_test.csv', dataSetHomePath+'/all_test.csv',
#dataSetHomePath+'/01.csv', dataSetHomePath+'/02.csv', dataSetHomePath+'/03.csv', dataSetHomePath+'/04.csv', dataSetHomePath+'/05.csv', dataSetHomePath+'/06.csv']

#expPath_all = ['GRU_ArabSign_encDec_0021_v1', 'GRU_ArabSign_encDec_0022_v1', 'GRU_ArabSign_encDec_0023_v1', 'GRU_ArabSign_encDec_0024_v1', 'GRU_ArabSign_encDec_0025_v1', 'GRU_ArabSign_encDec_0026_v1', 'GRU_ArabSign_encDec_0027_v1',
#'GRU_ArabSign_encDec_0028_v1','GRU_ArabSign_encDec_0029_v1','GRU_ArabSign_encDec_0030_v1','GRU_ArabSign_encDec_0031_v1','GRU_ArabSign_encDec_0032_v1','GRU_ArabSign_encDec_0033_v1']

#signer independent
trainPath_all = [dataSetHomePathIndependent+'/independent_01_train.csv',dataSetHomePathIndependent+'/independent_02_train.csv',dataSetHomePathIndependent+'/independent_03_train.csv', dataSetHomePathIndependent+'/independent_04_train.csv', dataSetHomePathIndependent+'/independent_05_train.csv', dataSetHomePathIndependent+'/independent_06_train.csv']

testPath_all = [dataSetHomePathIndependent+'/independent_01_test.csv',dataSetHomePathIndependent+'/independent_02_test.csv',dataSetHomePathIndependent+'/independent_03_test.csv', dataSetHomePathIndependent+'/independent_04_test.csv', dataSetHomePathIndependent+'/independent_05_test.csv', dataSetHomePathIndependent+'/independent_06_test.csv']
expPath_all = ['GRU_ArabSign_encDec_0021_v1', 'GRU_ArabSign_encDec_0022_v1', 'GRU_ArabSign_encDec_0023_v1', 'GRU_ArabSign_encDec_0024_v1', 'GRU_ArabSign_encDec_0025_v1', 'GRU_ArabSign_encDec_0026_v1']

i = 0
while i < len(expPath_all):
    print(expPath_all[i])
    expPath = expPath_all[i]
    trainPath = trainPath_all[i]
    testPath = testPath_all[i]
    expPath = os.path.join(os.getcwd(),'results',expPath)
    createFolder(expPath)
    startExperiment(expPath, trainPath, testPath, groundTruth, diVideoSet['nFramesNorm'] , diFeature, diVideoSet['modelName'])
    i = i + 1

###########################