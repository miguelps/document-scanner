"""
参考： http://fengjian0106.github.io/2017/05/08/Document-Scanning-With-TensorFlow-And-OpenCV/
"""

import argparse
import os

import cv2
import numpy as np

CANNY_THRESH_MIN = 10  # 小于这个阈值的像素不会被认为是边缘
CANNY_THRESH_MAX = 50  # 大于这个阈值的像素才会被认为是边缘

CONTOURS_LENGTH_THRESH = 200
CONTOURS_AREA_THRESH = 50

HOUGH_THRESH = 200
HOUGH_MIN_LINE_LENGTH = 20
HOUGH_MAX_LINE_GAP = 50
LINE_LENGTH_THRESH = 150  # 过滤霍夫直线检测的结果


def main(args):
    image = cv2.imread(args.img)
    watch(image, 'Origin')
    print("Origin size: %s" % str(image.shape))

    scale = 1600 / max(image.shape)
    image = cv2.resize(image, None, None, fx=scale, fy=scale)
    print("Resize size: %s" % str(image.shape))

    # image = ndimage.rotate(image, 90)
    # watch(image, 'Rotate')
    # print("Rotate size: %s" % str(image.shape))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    watch(gray, 'Gray')

    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    # blurred = cv2.medianBlur(gray, 5)
    watch(blurred, 'Blur')

    edged = cv2.Canny(blurred, CANNY_THRESH_MIN, CANNY_THRESH_MAX)
    watch(edged, 'Canny edged')

    _, contours, aa = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print("Contours num: %d" % len(contours))

    filtered_contours = []
    contours_img = np.zeros(edged.shape, dtype=np.uint8)
    for c in contours:
        arcLen = cv2.arcLength(c, True)
        area = cv2.contourArea(c)
        # 文档的边缘计算出来的 area 可能会很小
        if arcLen < CONTOURS_LENGTH_THRESH and area < CONTOURS_AREA_THRESH:
            continue

        filtered_contours.append(c)
        cv2.drawContours(contours_img, [c], -1, (255, 255, 255), 3)

    print("Filtered contours num: %d" % len(filtered_contours))
    watch(contours_img, 'Contours image')

    lines = cv2.HoughLinesP(contours_img, 1, np.pi / 180, HOUGH_THRESH, HOUGH_MIN_LINE_LENGTH, HOUGH_MAX_LINE_GAP)
    filterd_lines = []
    if lines is not None:
        print("Hough Lines num: %d" % len(lines))

        lines_img = image.copy()
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) < LINE_LENGTH_THRESH:
                continue
            filterd_lines.append(line)
            cv2.line(lines_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        print("Filtered Hough Lines num: %d" % len(filterd_lines))
        watch(lines_img, "Hough Lines")

    # 做投影变换，摆正视图，目标视图的比例应该和文档的比例相关
    # approx = rectify(target)
    # pts2 = np.float32([[0, 0], [800, 0], [800, 800], [0, 800]])
    #
    # M = cv2.getPerspectiveTransform(approx, pts2)
    # dst = cv2.warpPerspective(image, M, (800, 800))
    #
    # cv2.drawContours(image, [target], -1, (0, 255, 0), 2)
    # watch(image, 'Image with contours')
    #
    # dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    # watch(dst, 'Result')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', default='./demo.jpg')
    parser.add_argument('--mode', default='img', choices=['img', 'webcam'])

    args = parser.parse_args()

    if args.mode == 'img' and not os.path.exists(args.img):
        parser.error("Image not exist.")

    return args


def rectify(h):
    h = h.reshape((4, 2))
    hnew = np.zeros((4, 2), dtype=np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h, axis=1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew


def watch(img, name):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 1200, 800)
    cv2.imshow(name, img)
    cv2.waitKey()


if __name__ == '__main__':
    main(parse_args())
