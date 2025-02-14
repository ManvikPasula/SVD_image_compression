import os
import rawpy
import pickle
import argparse
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

def compress(args):
    raw = rawpy.imread(args.file)
    og_image = raw.postprocess()
    image = np.array(og_image)
    compressed_image = np.zeros_like(image)
    
    final_encoding = []
    for x in range(image.shape[-1]):
        temp_image = image[:,:,x]
        U, S, Vt = la.svd(temp_image, full_matrices=False)
        
        total = S.sum()
        cutoff = args.percent * total
        curr_sum = 0
        for i in range(S.shape[0]):
            curr_sum += S[i]
            if curr_sum >= cutoff:
                break
        S[i:] = 0
        
        new_U = U[:, :i]
        new_S = S[:i]
        new_Vt = Vt[:i, :]
        temp_compressed_image = new_U @ np.diag(new_S) @ new_Vt
        compressed_image[:,:,x] = temp_compressed_image
        
        final_encoding.append([new_U, new_S, new_Vt])
    
    if args.show_compressed:
        compressed_image = np.clip(compressed_image, 0, 255).astype(np.uint8)
        
        plt.imshow(compressed_image)
        plt.title("Compressed Image")
        plt.show()
    
    if args.save_encoded:
        base_name, _ = os.path.splitext(args.file)
        new_name =  f"{base_name}.pkl"
        with open(new_name, "wb") as f:
            pickle.dump(final_encoding, f)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str)
    parser.add_argument("--percent", type=float, default=0.75)
    parser.add_argument("--save_encoded", type=bool, default=True)
    parser.add_argument("--show_compressed", type=bool, default=False)
    
    args = parser.parse_args()
    
    compress(args)
    