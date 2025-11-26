import numpy as np
import cv2 as cv


def Read_Image(filename):
    image = cv.imread(filename)
    image = np.uint8(image)
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image = cv.resize(image, [512, 512])
    return image


def Image_Result():
    IMAGE = [4, 5, 11, 12, 27, 28]  # [4, 5, 11,12, 27, 28]
    Methods = ['FCN', 'UNET', 'RESUNET', 'DeepLabv3', 'TransDeeplabv3', 'Proposed']
    Images = np.load('Img_2.npy', allow_pickle=True)
    Prep = np.load('Preprocess_Images.npy', allow_pickle=True)
    Seg_Images = np.load('Segmented.npy', allow_pickle=True)  # Fusion
    Seg_Images1 = np.load('Fuzzy.npy', allow_pickle=True)
    Seg_Images2 = np.load('Segmented1.npy', allow_pickle=True)
    # Mask_Directory = './Data/Datsaet ' + str(n + 1) + '/Mask/'
    # Image_Dir = './Image_Results1/'
    for i in range(len(IMAGE)):  # Dataset-1-Image-2-1
        image = Images[IMAGE[i]]
        preprocess = Prep[IMAGE[i]]
        image1 = Seg_Images1[IMAGE[i]]
        image2 = Seg_Images2[IMAGE[i]]
        image3 = Seg_Images[IMAGE[i]]
        cv.imshow("Original Image " + str(i + 1), image)
        cv.waitKey(0)
        cv.imshow("Preprocessed Image " + str(i + 1), preprocess)
        cv.waitKey(0)
        cv.imshow("FCN Image " + str(i + 1), image1)
        cv.waitKey(0)
        cv.imshow("Region Growing Image " + str(i + 1), image2)
        cv.waitKey(0)
        cv.imshow("Proposed Image " + str(i + 1), image3)
        cv.waitKey(0)
        # cv.imwrite('./Results/Image_Results1/' + 'orig-' + str(i + 1) + '.png', image)
        # cv.imwrite('./Results/Image_Results1/' + 'Preprocess-' + str(i + 1) + '.png', preprocess)
        # cv.imwrite('./Results/Image_Results1/' + 'fcn-' + str(i + 1) + '.png', image1)
        # cv.imwrite('./Results/Image_Results1/' + 'Region-' + str(i + 1) + '.png',
        #            (image2).astype(np.uint8))
        # cv.imwrite('./Results/Image_Results1/' + 'Proposed-' + str(i + 1) + '.png', image3)


Image_Result()
