from PIL import Image, ImageOps
import cv2
import numpy as np
import matplotlib.pyplot as plt


import lilia
from k3m import skeletize

def clahe(im):
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(10,10))
    return clahe.apply(im)


def bilateralFilter(im):
    return cv2.bilateralFilter(im, d=10, sigmaColor=400, sigmaSpace=400)


def overexpose_and_blur(im):
    def gammaCorrection(src, gamma=2.5):
        invGamma = 1 / gamma
    
        table = [((i / 255) ** invGamma) * 255 for i in range(256)]
        table = np.array(table, np.uint8)
        return cv2.LUT(src, table)
    im = gammaCorrection(im)
    kernel = np.ones((4,4),np.float32)/16
    return cv2.filter2D(im,-1,kernel)


def qtz(im):
    im = Image.fromarray(im)
    im = im.quantize(colors=4, method=1)
    im = np.array(im.convert("L"))
    return im


def canny_edges(im):
    im = cv2.Canny(im, 100, 200)
    return im


def plot_image(img, number):
    img = np.array(img)
    fig.add_subplot(1, 2, number)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)

if __name__ == "__main__":
    fig = plt.figure(figsize=(8,6))
    #image = Image.open( "images/Pankraz.png")
    image = Image.open( "images/Aleks.jpeg")

    # resize to be a multiple of 297x210 (size of dinA4 sheet in mm)
    image = image.resize((210*3,297*3))
    plot_image(image, 1)

    # first convert to grayscale
    image = image.convert("L")
    # convert to numpy array
    image = np.array(image)

    # get original canny
    canny_orig = canny_edges(image)

    image = clahe(image)

    image = bilateralFilter(image)
    
    # get bilateral canny
    canny_bilateral = canny_edges(image)

    image = overexpose_and_blur(image)  
    
    # get overexpose_and_blur canny
    canny_overexposed = canny_edges(image)

    image = qtz(image)

    image = canny_edges(image)

    # combine cannys
    image = canny_orig + canny_bilateral + canny_overexposed + image
    image[np.nonzero(image)] = 255

    plot_image(image, 2)

    plt.show()
