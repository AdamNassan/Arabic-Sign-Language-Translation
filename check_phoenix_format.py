import gzip
import pickle
import json
import numpy as np
import torch

def read_phoenix_file(filename):
    # First try reading as gzipped pickle
    try:
        with gzip.open(filename, 'rb') as f:
            data = pickle.load(f)
            print("Successfully read as gzipped pickle file")
            print("Content type:", type(data))
            if isinstance(data, (list, tuple)):
                print("Length:", len(data))
                if len(data) > 0:
                    print("First item type:", type(data[0]))
                    if isinstance(data[0], dict):
                        print("Dict keys:", list(data[0].keys()))
            return "pickle", data
    except Exception as e:
        print(f"Not a pickle file: {e}")

    # Try reading as gzipped text 
    try:
        with gzip.open(filename, 'rt', encoding='utf-8') as f:
            lines = []
            for i, line in enumerate(f):
                if i < 5:
                    lines.append(line.strip())
                else:
                    break
            print("Successfully read as gzipped text file")
            print("First 5 lines:", lines)
            return "text", lines
    except Exception as e:
        print(f"Not a text file: {e}")

    # Try reading as numpy file
    try:
        data = np.load(filename, allow_pickle=True)
        print("Successfully read as numpy file")
        print("Shape:", data.shape)
        if len(data) > 0:
            print("First item type:", type(data[0]))
            print("First item:", data[0])
        return "numpy", data
    except Exception as e:
        print(f"Not a numpy file: {e}")

    print("Could not determine file format")
    return None, None

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python check_phoenix_format.py <filename>")
        sys.exit(1)
    
    filename = sys.argv[1]
    print(f"\nChecking format of {filename}\n")
    fmt, data = read_phoenix_file(filename)
    print(f"\nFile format: {fmt}")
