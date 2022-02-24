from PIL import Image, ImageOps
import cv2
import numpy as np
import matplotlib.pyplot as plt


import lilia

def clahe(im):
    im = np.array(im)
    clahe = cv2.createCLAHE(clipLimit=2)
    return clahe.apply(im)


def bilateralFilter(im):
    im = np.array(im)
    return cv2.bilateralFilter(im, d=10, sigmaColor=50, sigmaSpace=50)


def overexpose_and_blur(im):
    def gammaCorrection(src, gamma=2):
        invGamma = 1 / gamma
    
        table = [((i / 255) ** invGamma) * 255 for i in range(256)]
        table = np.array(table, np.uint8)
        return cv2.LUT(src, table)
    im = np.array(im)
    im = gammaCorrection(im)
    kernel = np.ones((5,5),np.float32)/25
    return cv2.filter2D(im,-1,kernel)


def quantize(im):
    im = np.array(im)
    im = Image.fromarray(im)
    return im.quantize(4)


def canny_edges(im):
    im = np.array(im)
    im = Image.fromarray(im)
    im = im.convert("L")
    im = np.array(im)
    im = cv2.Canny(im, 100, 200)
    return im


def plot_image(img, number):
    img = np.array(img)
    fig.add_subplot(1, 2, number)
    plt.imshow(im, cmap='gray', vmin=0, vmax=255)

if __name__ == "__main__":
    fig = plt.figure(figsize=(8,6))
    im = Image.open( "images/lel.jpg")

    # resize to be a multiple of 297x210 (size of dinA4 sheet in mm)
    im = im.resize((210*2,297*2))
    plot_image(im, 1)

    # first convert to grayscale
    im = im.convert("L")

    im = clahe(im)

    im = bilateralFilter(im)

    im = overexpose_and_blur(im)

    im = quantize(im)

    im = canny_edges(im)

    plot_image(im, 2)

    plt.show()
