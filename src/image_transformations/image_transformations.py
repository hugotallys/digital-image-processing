"""
DIP - Digital Image Processing
Implementing image transformations with numpy and opencv.

Author: Hugo Tallys Martins Olveira
"""

import os
import sys

import cv2 as cv
import numpy as np


def crop_and_flip(img, x0, y0, x1, y1):
    """
    Crop and flip an image. The cropped section is the rectangle
    delimited by points (x0, y0) (upper left) and (x1, y1) (lower right).
    """
    if x0 > x1 or y0 > y1:
        raise ValueError(
            "x0 and y0 must be smaller than x1 and y1 respectively"
        )
    return img[x1:x0:-1, y1:y0:-1, :]


def affine_transform(img, tx, ty, theta=0.):
    '''
    Performs an affine transformation in the image. The transformation
    is a composition of a translation of (tx, ty) followed by a
    rotation of theta radians.
    '''
    T = np.array([
        np.cos(theta), np.sin(theta), tx,
        -np.sin(theta), np.cos(theta), ty,
    ]).reshape(2, 3)
    height, width = img.shape[:2]
    return cv.warpAffine(img, T, (width, height))


def resize(img, width, height, mode=cv.INTER_AREA):
    '''
    Resizes a image to dimensions (width, height, N_CHANNELS) using
    the interpolation given by `mode`.
    '''
    return cv.resize(img, (width, height), mode)


def bitwise_op(img1, img2, op="AND"):
    """
    Performs a bitwise operation in each pixel of the image.
    The dimensions of each image must match.
    """
    foo = {
        "AND": np.bitwise_and,
        "OR": np.bitwise_or,
        "XOR": np.bitwise_xor
    }

    assert img1.shape == img2.shape, (
        "Mismatch of dimensions: "
        f"{img1.shape} != {img2.shape}"
    )
    assert op in foo, "Operation must be `AND`, `OR` or `XOR`"

    img = np.zeros_like(img1)

    for c in range(img1.shape[2]):
        img[:, :, c] = foo[op](img1[:, :, c], img2[:, :, c])
    return img


def mask(img, bmask):
    '''
    Applies a binary mask (single channel) to an image and performs AND
    bitwise operation.
    '''
    assert img.shape[:2] == bmask.shape, (
        "Mismatch of dimensions: "
        f"img.shape={img.shape[:2]} != mask.shape={bmask.shape}"
    )

    for c in range(img.shape[2]):
        img[:, :, c] = np.bitwise_and(img[:, :, c], bmask)
    return img


if __name__ == "__main__":
    data = []

    bgr_img = cv.imread("imgs/imageTransformations/hummingbird-1024px.jpg")

    if bgr_img is None:
        sys.exit("Could not read the image.")

    cv.imshow("Original Iimage", bgr_img)

    cropped_img = crop_and_flip(bgr_img, 256, 256, 756, 756)
    cv.imshow("Cropped and flipped image", cropped_img)
    data.append(("cropped.jpeg", cropped_img))

    translated_img = affine_transform(bgr_img, 256, 256)
    cv.imshow("Translated image", translated_img)
    data.append(("translated.jpeg", translated_img))

    rotated_img = affine_transform(bgr_img, 0, 0, np.pi/6)
    cv.imshow("Rotated image", rotated_img)
    data.append(("rotated.jpeg", rotated_img))

    new_height, new_width = np.array(bgr_img.shape[:2]) // 3
    resized_img = resize(bgr_img, new_width, new_height)
    cv.imshow("Resized image", resized_img)
    data.append(("resized.jpeg", resized_img))

    dummy_mask = np.full_like(bgr_img, 255)
    dummy_mask = affine_transform(dummy_mask, new_width, new_height)
    cv.imshow("Dummy mask", dummy_mask)
    data.append(("dummy_mask.jpeg", dummy_mask))

    and_img = bitwise_op(bgr_img, dummy_mask, "AND")
    cv.imshow("Bitwise AND", and_img)
    data.append(("and.jpeg", and_img))

    or_img = bitwise_op(bgr_img, dummy_mask, "OR")
    cv.imshow("Bitwise OR", or_img)
    data.append(("or.jpeg", or_img))

    xor_img = bitwise_op(bgr_img, dummy_mask, "XOR")
    cv.imshow("Bitwise XOR", xor_img)
    data.append(("xor.jpeg", xor_img))

    threshold = 128
    bin_mask = bgr_img[:, :, 0]
    bin_mask[bin_mask > threshold] = 255
    bin_mask[bin_mask <= threshold] = 0

    masked_img = mask(bgr_img, bin_mask)
    cv.imshow("Masked image", masked_img)
    data.append(("masked.jpeg", masked_img))

    for fname, img in data:
        cv.imwrite(os.path.join(
            "imgs/imageTransformations", fname), img)

    cv.waitKey(0)
