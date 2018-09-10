"""
参考： http://fengjian0106.github.io/2017/05/08/Document-Scanning-With-TensorFlow-And-OpenCV/
"""

import argparse
import os

import cv2
import numpy as np

from line import Line, Point

MAX_LENGTH = 1200  # 图片的长边 resize 到 MAX_LENGTH

CANNY_THRESH_MIN = 10  # 小于这个阈值的像素不会被认为是边缘
CANNY_THRESH_MAX = 50  # 大于这个阈值的像素才会被认为是边缘

CONTOURS_LENGTH_THRESH = 200
CONTOURS_AREA_THRESH = 50

HOUGH_THRESH = 200
HOUGH_MIN_LINE_LENGTH = 20
HOUGH_MAX_LINE_GAP = 50
LINE_LENGTH_THRESH = 150  # 过滤霍夫直线检测的结果

INTERSECTION_ANGLE_MIN = 45
INTERSECTION_ANGLE_MAX = 135

# 合并靠的比较近的直线，用的阈值
MERGE_LINE_MAX_DISTANCE = 20


def main(args):
    image = cv2.imread(args.img)
    # watch(image, 'Origin')
    print("Origin size: %s" % str(image.shape))

    scale = 1200 / max(image.shape)
    image = cv2.resize(image, None, None, fx=scale, fy=scale)
    height = image.shape[0]
    width = image.shape[1]
    print("Resize size: %s" % str(image.shape))

    # image = ndimage.rotate(image, 90)
    # watch(image, 'Rotate')
    # print("Rotate size: %s" % str(image.shape))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # watch(gray, 'Gray')

    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    # blurred = cv2.medianBlur(gray, 5)
    # watch(blurred, 'Blur')

    edged = cv2.Canny(blurred, CANNY_THRESH_MIN, CANNY_THRESH_MAX)
    # watch(edged, 'Canny edged')

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
    # watch(contours_img, 'Contours image')

    # 对轮廓图进行霍夫变换检测直线
    lines = cv2.HoughLinesP(contours_img, 1, np.pi / 180, HOUGH_THRESH, HOUGH_MIN_LINE_LENGTH, HOUGH_MAX_LINE_GAP)
    filterd_lines = []
    if lines is not None:
        print("Hough Lines num: %d" % len(lines))

        lines_img = image.copy()
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 过滤掉长度过短的直线
            if np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) < LINE_LENGTH_THRESH:
                continue
            filterd_lines.append(line[0])
            cv2.line(lines_img, (x1, y1), (x2, y2), (255, 0, 0), 1)

        print("Filtered Hough Lines num: %d" % len(filterd_lines))
        watch(lines_img, "Hough Lines")

    # 获得直线段的延长线
    extended_lines = []
    extended_lines_img = image.copy()
    for line in filterd_lines:
        x1, y1, x2, y2 = line

        if x1 == x2:
            # 竖直的线
            extend_x1 = extend_x2 = x1
            extend_y1 = 0
            extend_y2 = height
        elif y2 == y1:
            # 水平的线
            extend_x1 = 0
            extend_y1 = extend_y2 = y1
            extend_x2 = width
        else:
            k = float(y2 - y1) / float(x2 - x1)

            extend_y1 = 0
            extend_x1 = x1 - y1 / k
            if extend_x1 < 0:
                extend_x1 = 0
                extend_y1 = y1 - k * x1
            elif extend_x1 >= width:
                extend_x1 = width
                extend_y1 = y1 + k * (width - x1)

            extend_y2 = height
            extend_x2 = x2 + (height - y2) / k
            if extend_x2 > width:
                extend_x2 = width
                extend_y2 = y2 + k * (width - x2)
            elif extend_x2 < 0:
                extend_x2 = 0
                extend_y2 = y2 - k * x2

            extend_x1, extend_y1, extend_x2, extend_y2 = int(extend_x1), int(extend_y1), int(extend_x2), int(extend_y2)

        cv2.line(extended_lines_img, (extend_x1, extend_y1), (extend_x2, extend_y2), (0, 255, 0), 1)
        extended_lines.append(Line(Point(extend_x1, extend_y1), Point(extend_x2, extend_y2)))
    watch(extended_lines_img, "Extended Lines")

    # 合并斜率相近，并且靠的很近的直线
    merged_extended_lines = []
    merged_lines = {}
    merged_lines_img = image.copy()
    for i in range(len(extended_lines)):
        if i in merged_lines:
            continue

        line = extended_lines[i]
        for j in range(i + 1, len(extended_lines)):
            if j in merged_lines:
                continue

            _line = extended_lines[j]
            if line.close_to(_line, MERGE_LINE_MAX_DISTANCE):
                merged_lines[j] = j
                line.merge(_line)

        merged_extended_lines.append(line)

    for line in merged_extended_lines:
        cv2.line(merged_lines_img, (line.p0.x, line.p0.y), (line.p1.x, line.p1.y), color=(0, 255, 0))

    print("Lines num after merge: %d" % len(merged_extended_lines))
    watch(merged_lines_img, "Merged Lines")

    # 获得线段的延长线交点
    for i in range(len(merged_extended_lines)):
        line = merged_extended_lines[i]
        for j in range(i + 1, len(merged_extended_lines)):
            _line = merged_extended_lines[j]
            point = line.cross(_line)
            if point:
                cv2.circle(merged_lines_img, (point.x, point.y), 5, color=(0, 0, 255), thickness=2)
    watch(merged_lines_img, "Extended Lines cross point")





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
