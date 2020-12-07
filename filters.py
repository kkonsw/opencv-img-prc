# выполнил: Кузнецов Константин
# группа: 381503-3

import cv2
import numpy as np


def gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray


def contrast(img):
    contr = cv2.equalizeHist(gray(img))
    return contr


def edges(img):
    edge = cv2.Canny(contrast(img), 100, 200)
    return edge


def cor1(img):
    cor_img = edges(img)
    corners = cv2.goodFeaturesToTrack(cor_img, 100, 0.2, 2)
    color = [255, 255, 255]
    corners = np.int0(corners)
    for i in corners:
        x, y = i.ravel()
        cv2.circle(cor_img, (x, y), 10, color)
    return cor_img


def cor2(img):
    cor_img = contrast(img)
    corners = cv2.goodFeaturesToTrack(cor_img, 100, 0.2, 2)
    color = [255, 255, 100, 255]
    corners = np.int0(corners)
    for i in corners:
        x, y = i.ravel()
        cv2.circle(cor_img, (x, y), 10, color)
    return cor_img


def distance(img):
    dist = cor1(img)
    dist = cv2.bitwise_not(dist)
    dist = cv2.distanceTransform(dist, cv2.DIST_L2, 3)
    dist = cv2.normalize(dist, None, 0, 1, cv2.NORM_MINMAX)
    return dist


def clamp(n, left, right):
    return max(left, min(n, right))


def filter(img):
    dist = cor1(img)
    dist = cv2.bitwise_not(dist)
    dist = cv2.distanceTransform(dist, cv2.DIST_L2, 3)
    height = img.shape[0]
    width  = img.shape[1]

    k = 0.5
    int_img = cv2.integral(contrast(img))
    fil = np.zeros((height, width), np.uint8)

    for y in range(height):
        for x in range(width):
            r = min(int(k * dist[y, x]), 4)

            a = int_img[clamp(y - r, 0, height - 1), clamp(x - r, 0, width - 1)]
            b = int_img[clamp(y + 1 + r, 0, height - 1), clamp(x - r, 0, width - 1)]
            c = int_img[clamp(y - r, 0, height - 1), clamp(x + 1 + r, 0, width - 1)]
            d = int_img[clamp(y + 1 + r, 0, height - 1), clamp(x + 1 + r, 0, width - 1)]

            fil[y, x] = (a + d - c - b) / ((1 + 2*r)**2)

    return fil


if __name__ == '__main__':
    img = cv2.imread('pyramid.jpg', 1)

    cv2.imshow("original", img)
    cv2.imshow("gray", gray(img))
    cv2.imshow("contrast", contrast(img))
    cv2.imshow("edges", edges(img))
    cv2.imshow("points", cor2(img))
    cv2.imshow("distance", distance(img))
    cv2.imshow("filtered", filter(img))

    cv2.waitKey()
