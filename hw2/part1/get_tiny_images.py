from PIL import Image
import numpy as np
import cv2

def get_tiny_images(image_paths):
    #############################################################################
    # TODO:                                                                     #
    # To build a tiny image feature, simply resize the original image to a very #
    # small square resolution, e.g. 16x16. You can either resize the images to  #
    # square while ignoring their aspect ratio or you can crop the center       #
    # square portion out of each image. Making the tiny images zero mean and    #
    # unit length (normalizing them) will increase performance modestly.        #
    #############################################################################
    '''
    Input : 
        image_paths: a list(N) of string where each string is an image 
        path on the filesystem.
    Output :
        tiny image features : (N, d) matrix of resized and then vectorized tiny
        images. E.g. if the images are resized to 16x16, d would equal 256.
    '''
    tiny_images = np.zeros((len(image_paths), 256))
    for i, image_path in enumerate(image_paths):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (16, 16), interpolation=cv2.INTER_AREA).reshape(256).astype(np.float)
        norm = np.linalg.norm(img)
        if norm != 0:
            img /= norm
        tiny_images[i] = img

    ##############################################################################
    #                                END OF YOUR CODE                            #
    ##############################################################################

    return tiny_images
