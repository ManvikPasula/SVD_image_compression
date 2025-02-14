import pickle
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def decompress(args):
    with open(args.file, "rb") as f:
        encoded = pickle.load(f)
    to_combine = []
    for (U, S, Vt) in encoded:
        to_combine.append(np.expand_dims(U @ np.diag(S) @ Vt, axis=-1))
    compressed_image = np.concatenate(to_combine, axis=2)
    compressed_image = np.clip(compressed_image, 0, 255).astype(np.uint8)

    plt.imshow(compressed_image)
    plt.title("Compressed Image")
    plt.show()
    
    if args.save_compressed:
        output_filename = "compressed_" + args.file.split('/')[-1].split('.')[0] + ".jpg"
        Image.fromarray(compressed_image).save(output_filename)
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str)
    parser.add_argument("--save_compressed", type=bool, default=False)
    
    args = parser.parse_args()
    
    decompress(args)
    