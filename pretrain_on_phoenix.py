#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from tqdm import tqdm
import gzip
import pickle
import logging
import encoderDecoderModel as EncDecModel
from config import device

# Define EncoderDecoder class
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, vocab_size, modelName) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size
        self.modelName = modelName

    def forward(self, features, feature_lengths, targets=None, teacher_forcing_ratio=0.5):
        batch_size = features.size(1)
        max_length = targets.size(1) if targets is not None else 50
        
        # Initialize outputs tensor
        outputs = torch.zeros(max_length, batch_size, self.vocab_size).to(features.device)
        
        # Initial input is SOS token (shape: batch_size)
        decoder_input = torch.ones(batch_size, dtype=torch.long).to(features.device)  # SOS token index = 1
          # Get initial hidden state from encoder
        encoder_outputs, hidden = self.encoder(features)        # Handle GRU/LSTM states and bidirectional encoder
        if isinstance(hidden, tuple):
            hidden = hidden[0]  # For GRU/LSTM, only take the hidden state, not cell state
        
        # For bidirectional encoder, combine forward and backward states
        if hidden.size(0) > self.encoder.num_layers:
            # Reshape and combine directions
            hidden = hidden.view(2, self.encoder.num_layers, batch_size, -1)
            hidden = torch.cat([hidden[0], hidden[1]], dim=2)  # Concatenate forward and backward states
        
        for t in range(max_length):
            # decoder_input shape should be (batch_size)
            output, hidden = self.decoder(decoder_input, hidden)
            outputs[t] = output  # Store the output
            
            # Teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            if targets is not None and teacher_force:
                decoder_input = targets[:, t]  # Use target from current timestep
            else:
                # Get most likely word from current prediction
                decoder_input = output.argmax(1)  # Shape: (batch_size)
                
        return outputs.transpose(0, 1)  # Return batch first format (batch_size, seq_len, vocab_size)

# Special tokens
PAD_TOKEN = '<PAD>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhoenixDataset(Dataset):
    def __init__(self, file_path, max_seq_length=300):
        """
        Args:
            file_path: Path to phoenix dataset file (.pami0.train, .pami0.dev, or .pami0.test)
            max_seq_length: Maximum sequence length for padding
        """
        self.file_path = file_path
        self.max_seq_length = max_seq_length
        
        # Load the gzipped pickle file
        logger.info(f"Loading data from {file_path}...")
        try:
            with gzip.open(file_path, 'rb') as f:
                self.data = pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            raise
        
        # Initialize vocabulary 
        self.vocab = {
            PAD_TOKEN: 0,
            SOS_TOKEN: 1,
            EOS_TOKEN: 2,
            UNK_TOKEN: 3
        }
        self.build_vocab()
        
        logger.info(f"Loaded {len(self.data)} sequences")
        logger.info(f"Vocabulary size: {len(self.vocab)}")

    def build_vocab(self):
        """Build vocabulary from the dataset"""
        idx = len(self.vocab)  # Start after special tokens
        for item in self.data:            
            text = item['text'].strip().split()
            for word in text:
                if word not in self.vocab:
                    self.vocab[word] = idx
                    idx += 1

    def encode_sequence(self, sequence):
        """Convert a sequence of words to indices"""
        return [self.vocab.get(word, self.vocab[UNK_TOKEN]) for word in sequence.strip().split()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get features
        features = item['sign']  # Should be torch tensor of shape [T, 1024]
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
            
        # Normalize features if needed
        if torch.any(features > 100) or torch.any(features < -100):  
            features = (features - features.mean()) / (features.std() + 1e-8)
        
        # Add small epsilon for numerical stability
        features = features + 1e-8
          # Get text sequence
        text = item['text'].strip()
        text_indices = self.encode_sequence(text)
        
        # Add SOS and EOS tokens
        text_indices = [self.vocab[SOS_TOKEN]] + text_indices + [self.vocab[EOS_TOKEN]]
        
        # Convert to tensor and pad if needed
        text_indices = torch.tensor(text_indices, dtype=torch.long)
        if len(text_indices) < self.max_seq_length:
            padding = torch.full((self.max_seq_length - len(text_indices),), 
                               self.vocab[PAD_TOKEN], 
                               dtype=torch.long)
            text_indices = torch.cat([text_indices, padding])
        else:
            text_indices = text_indices[:self.max_seq_length]
        
        return {
            'name': item['name'],
            'features': features,
            'text_indices': text_indices,
            'text': text
        }

def collate_fn(batch):
    """
    Custom collate function for variable length sequences
    """
    # Sort by feature sequence length for packing
    batch = sorted(batch, key=lambda x: len(x['features']), reverse=True)
    
    # Get sequence lengths
    feature_lengths = torch.tensor([len(x['features']) for x in batch])
    max_feature_len = max(feature_lengths)
    
    # Pad features
    features = []
    for item in batch:
        feat = item['features']
        if len(feat) < max_feature_len:
            padding = torch.zeros((max_feature_len - len(feat), feat.size(1)), dtype=feat.dtype)
            feat = torch.cat([feat, padding])
        features.append(feat)
    
    # Stack features and transpose to (seq_len, batch_size, features)
    features = torch.stack(features, dim=1)  # Now shape: (seq_len, batch, features)
    
    # Stack text indices
    text_indices = torch.stack([x['text_indices'] for x in batch])
    
    return {
        'names': [x['name'] for x in batch],
        'features': features,
        'feature_lengths': feature_lengths,
        'text_indices': text_indices,
        'texts': [x['text'] for x in batch]
    }

def pretrain_on_phoenix(train_file, dev_file, checkpoint_dir='checkpoints', num_epochs=50, batch_size=32):
    """
    Pretrain the encoder-decoder model on the Phoenix dataset with improved regularization
    """
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = PhoenixDataset(train_file)
    dev_dataset = PhoenixDataset(dev_file)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )    # Build encoder and decoder separately with increased dropout
    input_size = 1024  # Feature dimension
    hidden_size = 256  # Reduced complexity
    num_layers = 2
    dropout = 0.7  # Increased dropout
    vocab_size = len(train_dataset.vocab)
    embedding_dim = 300

    # Create encoder and decoder
    encoder = EncDecModel.EncoderRNNPre(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)    
    decoder = EncDecModel.DecoderRNNPre(
        num_embeddings=vocab_size,
        embedding_dim=embedding_dim,
        input_size=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        vocab_size=vocab_size
    ).to(device)

    # Combine them into the EncoderDecoder model
    model = EncoderDecoder(encoder, decoder, vocab_size, modelName="EncoderDecoder").to(device)
    



    
    # L2 regularization
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocab[PAD_TOKEN])
    optimizer = optim.AdamW( # Changed to AdamW
        model.parameters(), 
        lr=0.0003,  # Lower learning rate
        weight_decay=0.01  # L2 regularization
    )
    
    # More patient learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        patience=3,
        factor=0.5,
        min_lr=1e-6
    )
    
    best_loss = float('inf')
    not_improved = 0
    
    logger.info("Starting training...")
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in train_batches:
            optimizer.zero_grad()
            
            # Move everything to device
            features = batch['features'].to(device)
            feature_lengths = batch['feature_lengths'].to(device)
            targets = batch['text_indices'].to(device)
            
            # Forward pass with increased teacher forcing at start
            teacher_forcing_ratio = max(0.5, 1.0 - epoch * 0.1)  # Decay from 1.0 to 0.5
            outputs = model(
                features, 
                feature_lengths, 
                targets[:, :-1],
                teacher_forcing_ratio=teacher_forcing_ratio
            )
            
            # Calculate loss
            loss = criterion(
                outputs.reshape(-1, outputs.shape[-1]),
                targets[:, 1:].reshape(-1)
            )
            
            # Add L2 regularization manually for encoder
            l2_lambda = 0.001
            l2_reg = torch.tensor(0.).to(device)
            for param in model.encoder.parameters():
                l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_batches.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc='Validating'):
                features = batch['features'].to(device)
                feature_lengths = batch['feature_lengths'].to(device)
                targets = batch['text_indices'].to(device)
                
                outputs = model(features, feature_lengths, targets[:, :-1], teacher_forcing_ratio=0.0)
                loss = criterion(
                    outputs.reshape(-1, outputs.shape[-1]),
                    targets[:, 1:].reshape(-1)
                )
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(dev_loader)
        scheduler.step(avg_val_loss)
        
        logger.info(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')
        logger.info(f'Learning rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save checkpoint if validation loss improves
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            not_improved = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
                'vocab': train_dataset.vocab
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'phoenix_best.pt'))
            logger.info(f'Saved new best checkpoint with loss {best_loss:.4f}')
        else:
            not_improved += 1
            
        # Early stopping
        if not_improved >= 7:  # More patience
            logger.info('Early stopping triggered')
            break
            
    return os.path.join(checkpoint_dir, 'phoenix_best.pt')

if __name__ == "__main__":
    train_file = "phoenix14t.pami0.train"
    dev_file = "phoenix14t.pami0.dev"
    
    checkpoint_path = pretrain_on_phoenix(train_file, dev_file)
    logger.info(f"Training complete. Best checkpoint saved at: {checkpoint_path}")
