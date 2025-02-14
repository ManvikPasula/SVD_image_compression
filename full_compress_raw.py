import os
import rawpy
import pickle
import argparse
import numpy as np
from PIL import Image
from numpy import linalg as la
import matplotlib.pyplot as plt

def compress(args):
    # load in a raw file and store
    # as a numpy array, and also
    # create a dummy array for the
    # compressed image that we will
    # store later
    
    raw = rawpy.imread(args.file)
    og_image = raw.postprocess()
    image = np.array(og_image)
    compressed_image = np.zeros_like(image)
    
    final_encoding = []
    for x in range(image.shape[-1]): # loop through the RGB channels
        temp_image = image[:,:,x] # store the image for a single channel
        U, S, Vt = la.svd(temp_image, full_matrices=False) # find SVD
        
        # this loop basically finds the 
        # sum of all the singular values
        # and finds a "cutoff", which lets
        # us know how many singular values
        # to keep, and the rest are set to 0
        #
        # since singular values are in order
        # of greatest to least, and that also
        # reflects their importance, we can 
        # just go straight through and only save
        # the most important aspects (bigger values) and
        # leave out the finer details (smaller values)
        
        total = S.sum() # total sum of singular values
        cutoff = args.percent * total # "cutoff" to use
        curr_sum = 0
        for i in range(S.shape[0]):
            curr_sum += S[i] # find current sum as we go through singular
            if curr_sum >= cutoff: # if we reached cutoff, stop
                break
        S[i:] = 0 # set everything that is beyond the important to 0
        
        # this creates a new set of
        # SVD matrices, similar to how
        # a reduced SVD doesn't include
        # stuff beyond the rank, this just
        # doesn't include stuff beyond the
        # "important" info., and it also
        # decreases the size of the pkl
        # if you choose to save the encoded
        # and probably makes matmul faster
        
        new_U = U[:, :i]
        new_S = S[:i]
        new_Vt = Vt[:i, :]
        temp_compressed_image = new_U @ np.diag(new_S) @ new_Vt
        compressed_image[:,:,x] = temp_compressed_image
        final_encoding.append([new_U, new_S, new_Vt]) # storing for saving encoded
    
    # we also have to make sure
    # that the values in the image
    # don't go below 0 or exceed
    # 255, so we clip them    
    
    compressed_image = np.clip(compressed_image, 0, 255).astype(np.uint8)
    
    if args.show_compressed: # show the image with plt if selected
        plt.imshow(compressed_image)
        plt.title("Compressed Image")
        plt.show()
        
    # save as a jpg, since what
    # this code is basically doing
    # is allowing us to further
    # compress raw files
    
    base_name, _ = os.path.splitext(args.file)
    new_name =  f"{base_name}.jpg"
    Image.fromarray(compressed_image).save(new_name)
    
    # this also saves the "extra reduced"
    # SVD for all color channels as a pkl
    # if the user selects that option
    
    if args.save_encoded:
        pickle_name =  f"{base_name}.pkl"
        with open(pickle_name, "wb") as f:
            pickle.dump(final_encoding, f)
    
if __name__ == "__main__":

    # parsing arguments
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str)
    parser.add_argument("--percent", type=float, default=0.75)
    parser.add_argument("--save_encoded", type=bool, default=False)
    parser.add_argument("--show_compressed", type=bool, default=False)
    
    args = parser.parse_args()
    
    # running the compression
    
    compress(args)
    