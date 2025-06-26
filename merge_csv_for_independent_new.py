#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import os
import glob
import argparse
from tqdm import tqdm

def merge_csv_files(target_signer, output_dir='data'):
    """
    Merge CSV files for signer-independent mode with proper Arabic text handling.
    
    Args:
        target_signer (str): The signer ID to use as test set (e.g., '02')
        output_dir (str): Directory to save the merged CSV files
        
    Returns:
        tuple: (train_file_path, test_file_path) - Paths to the created files
    """
    all_signers = ['01', '02', '03', '04', '05', '06']
    train_signers = [s for s in all_signers if s != target_signer]
    
    print(f"Merging files for signer-independent mode on signer {target_signer}")
    print(f"Using signers {', '.join(train_signers)} for training")
    
    # Path to CSV files
    data_dir = 'data'
    # Initialize empty DataFrames for train and test
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
      # Define list of encodings to try
    encodings = ['utf-8-sig', 'utf-8', 'cp1256', 'windows-1256', 'iso-8859-6', 'arabic']
    
    def detect_encoding(file_path):
        """Try to detect the file encoding"""
        import chardet
        with open(file_path, 'rb') as file:
            raw = file.read()
            result = chardet.detect(raw)
            print(f"Detected encoding for {os.path.basename(file_path)}: {result}")
            return result['encoding']
    
    # Read all training files for non-target signers
    for signer in tqdm(train_signers, desc=f"Processing training signers for {target_signer}"):
        # Add both train and test files from other signers to our training set
        train_file = os.path.join(data_dir, f"{signer}_train.csv")
        test_file = os.path.join(data_dir, f"{signer}_test.csv")
        
        # Process training file
        if os.path.exists(train_file):
            success = False
            for encoding in encodings:
                try:
                    df = pd.read_csv(train_file, header=None, encoding=encoding)
                    train_df = pd.concat([train_df, df])
                    print(f"  Added {len(df)} rows from {train_file} using {encoding}")
                    success = True
                    break
                except Exception:
                    continue
            if not success:
                print(f"  Error: Failed to read {train_file} with any encoding")
        else:
            print(f"  Warning: File {train_file} does not exist")
        
        # Process test file
        if os.path.exists(test_file):
            success = False
            for encoding in encodings:
                try:
                    df = pd.read_csv(test_file, header=None, encoding=encoding)
                    train_df = pd.concat([train_df, df])
                    print(f"  Added {len(df)} rows from {test_file} using {encoding}")
                    success = True
                    break
                except Exception:
                    continue
            if not success:
                print(f"  Error: Failed to read {test_file} with any encoding")
        else:
            print(f"  Warning: File {test_file} does not exist")
    
    # Read all files for target signer for testing
    print(f"\nProcessing target signer {target_signer} for testing")
    target_train_file = os.path.join(data_dir, f"{target_signer}_train.csv")
    target_test_file = os.path.join(data_dir, f"{target_signer}_test.csv")
    
    files_found = 0
    
    # Process target training file
    if os.path.exists(target_train_file):
        success = False
        for encoding in encodings:
            try:
                df = pd.read_csv(target_train_file, header=None, encoding=encoding)
                test_df = pd.concat([test_df, df])
                print(f"  Added {len(df)} rows from {target_train_file} using {encoding}")
                files_found += 1
                success = True
                break
            except Exception:
                continue
        if not success:
            print(f"  Error: Failed to read {target_train_file} with any encoding")
    else:
        print(f"  Warning: File {target_train_file} does not exist")
    
    # Process target test file
    if os.path.exists(target_test_file):
        success = False
        for encoding in encodings:
            try:
                df = pd.read_csv(target_test_file, header=None, encoding=encoding)
                test_df = pd.concat([test_df, df])
                print(f"  Added {len(df)} rows from {target_test_file} using {encoding}")
                files_found += 1
                success = True
                break
            except Exception:
                continue
        if not success:
            print(f"  Error: Failed to read {target_test_file} with any encoding")
    else:
        print(f"  Warning: File {target_test_file} does not exist")
    
    if files_found == 0:
        raise FileNotFoundError(f"No files found for target signer {target_signer}")
    
    if test_df.empty:
        raise ValueError(f"No data found for test set (signer {target_signer})")
    
    if train_df.empty:
        raise ValueError(f"No data found for training set (signers {', '.join(train_signers)})")
    
    # Save the merged files
    train_output = os.path.join(output_dir, f"independent_{target_signer}_train.csv")
    test_output = os.path.join(output_dir, f"independent_{target_signer}_test.csv")    # Save with UTF-8-BOM encoding to ensure proper Arabic text handling
    # Create output directories
    os.makedirs(os.path.dirname(train_output), exist_ok=True)
    os.makedirs(os.path.dirname(test_output), exist_ok=True)
    
    def safe_save_df(df, output_file):
        """Safely save a dataframe with Arabic text"""
        temp_file = output_file + '.tmp'
        df.to_csv(temp_file, index=False, header=False, encoding='utf-8-sig', quoting=1)
        if os.path.exists(output_file):
            os.remove(output_file)
        os.rename(temp_file, output_file)
        
        # Verify the content
        try:
            with open(output_file, 'r', encoding='utf-8-sig') as f:
                first_line = f.readline().strip().split(',')
                print(f"\nVerified {os.path.basename(output_file)} - First row Arabic text: {first_line[5]}")
        except Exception as e:
            print(f"Warning: Could not verify {output_file}: {e}")
    
    # Save files separately
    print("\nSaving output files...")
    safe_save_df(train_df, train_output)
    safe_save_df(test_df, test_output)
    
    # Verify the output encoding and content
    print("\nVerifying output files encoding and content:")
    train_sample = pd.read_csv(train_output, header=None, encoding='utf-8-sig', nrows=1)
    test_sample = pd.read_csv(test_output, header=None, encoding='utf-8-sig', nrows=1)
    print(f"Training file first row Arabic text: {train_sample.iloc[0, 5]}")
    print(f"Testing file first row Arabic text: {test_sample.iloc[0, 5]}")
    
    print(f"\nTraining set saved to {train_output} with {len(train_df)} rows")
    print(f"Testing set saved to {test_output} with {len(test_df)} rows")
    
    return train_output, test_output

def create_independent_files_for_all_signers(output_dir='data'):
    """Create independent train/test files for each signer"""
    all_signers = ['01', '02', '03', '04', '05', '06']
    
    train_paths = []
    test_paths = []
    exp_paths = []
    
    for signer in all_signers:
        print(f"\n{'='*50}")
        print(f"Processing signer {signer}")
        train_file, test_file = merge_csv_files(signer, output_dir)
        
        # Add to path lists
        train_paths.append(train_file)
        test_paths.append(test_file)
        exp_paths.append(f"SI_ArabSign_encDec_{signer}_v1")
        
        print(f"{'='*50}\n")
    
    # Print the path lists that can be used in the main script
    print("\nFor use in your main script (Encoder-Decoder.py):")
    print("trainPath_all = [" + ", ".join([f"'{path}'" for path in train_paths]) + "]")
    print("testPath_all = [" + ", ".join([f"'{path}'" for path in test_paths]) + "]")
    print("expPath_all = [" + ", ".join([f"'{path}'" for path in exp_paths]) + "]")
    
    return train_paths, test_paths, exp_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge CSV files for signer-independent mode testing.')
    parser.add_argument('--signer', type=str, help='Target signer for testing (e.g., 01, 02, etc.)')
    parser.add_argument('--output-dir', type=str, default='data/independent', help='Path to output directory')
    parser.add_argument('--all', action='store_true', help='Process all signers')
    
    args = parser.parse_args()
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")
    
    if args.all:
        create_independent_files_for_all_signers(args.output_dir)
    elif args.signer:
        merge_csv_files(args.signer, args.output_dir)
    else:
        # Default behavior if no arguments are provided
        print("No arguments provided. Creating files for all signers...")
        create_independent_files_for_all_signers(args.output_dir)
