import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve2d
from scipy import ndimage
import os


"""
The purpose of this script is to investigate some classical image analysis algporithms
ALGORITHM: CANNY eddge detector
Processing steps: Guassian smoothing, gradiant calculation, non-max suppression, dual threshold and hysterisis
OUTPUT: Image to overlay over another image, showing the edges that have been found in the original image

PIPELINE OVERVIEW: 

                            Input image

                            Gaussian smoothing

                            Gradient magnitude & direction

                            Non-maximum suppression
         
                            Double threshold
       
                            Edge hysteresis
                            ↓
                            Final edge map

We are basically finding filters that will provide a nice response when an edge's "step like respnse" is passed through

"""

def gaussian_kernel(size=5, sigma=1.0):
    """
    Create a 2D Gaussian kernel.
    size must be odd.
    """
    assert size % 2 == 1, "Kernel size must be odd"

    k = size // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]

    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()

    return kernel


def gaussian_filter(image, size=5, sigma=1.0):
    kernel = gaussian_kernel(size, sigma)

    # Apply convolution (preserve image size)
    smoothed = convolve2d(
        image,
        kernel,
        mode="same",
        boundary="symm"
    )

    return smoothed

image_path = "/mnt/d/SCHOOL_crap/ece_523/sandbox/dataSets/COCO/val2017_downsized/images/"        # Your downsized image


imageList = os.listdir(image_path)


for file in imageList:

    fullPath = image_path + file
    # where is the unsuspecting image located?
    # imageFile = "/mnt/d/SCHOOL_crap/ece_532/projectIdeas/cannyGearImage.png" # for intital testing
    image = Image.open(fullPath).convert("L") # it is black and white
    image = np.array(image) # for processing, convert to numpy array


    fig, axes = plt.subplots(1, 5, figsize=(16, 8))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input Image")
    # step 1 is to convolve the image with a guassian kernal of some sort to get rid of noise that may look like an edge to the detector
    blurredImage = gaussian_filter(image, 3, sigma=1.0) # make your own darned filter once this basic pipeline is working
    axes[1].imshow(blurredImage,cmap="gray")
    axes[1].set_title("After Guassian Conv.")

    # step 2 is to calculate the gradient (large changes in slope implies an edge )
    # sobel filter can be used to compute the x,y gradient of the image, to find magnitude and direction
    gradX = ndimage.sobel(blurredImage, axis=1)
    gradY = ndimage.sobel(blurredImage, axis=0)

    gradMag = np.hypot(gradX, gradY) # magnitude of jthe gradient
    angle = np.arctan2(gradY, gradX) # angle (direction)

    axes[2].imshow(gradMag, cmap="gray")
    axes[2].set_title("Gradient mag")
    axes[2].axis("off")

    # step 3 is to suppress nun-maximum values (narrow down the actual edge)
    #determine the threshold for "weak edges" and "strong edges"
    # first find the minimum and maximum pixel values in the gradient magnitude
    avgMag = np.mean(gradMag)
    threshold_weak = 1.0*avgMag
    threshold_strong = 2.0*avgMag

    strong_edges = gradMag >= threshold_strong
    weak_edges = (gradMag < threshold_strong) & (gradMag >= threshold_weak)

    edge_map = np.zeros((*gradMag.shape, 3), dtype=np.uint8)  # RGB image

    # Strong edges in red
    edge_map[strong_edges] = [255, 0, 0]

    # Weak edges in blue
    edge_map[weak_edges] = [0, 0, 255]


    axes[3].imshow(edge_map)
    axes[3].set_title("Strong (Red) and Weak (Blue) Edges")


    # final resulting image of edges only hopefully
    edge_map[strong_edges] = [0, 255, 0]
    edge_map[weak_edges] = [0, 255, 0]
    axes[4].imshow(edge_map)
    axes[4].set_title("Final result")

    # show the processing intermediate results 
    plt.show()


