import argparse
import os

import cv2
import numpy as np


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


def main(args):
    image = cv2.imread(args.img)
    print("Origin size: %s" % str(image.shape))
    image = cv2.resize(image, (1500, 880))
    watch(image, 'Origin')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    watch(gray, 'Gray')

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # blurred = cv2.medianBlur(gray, 5)
    watch(blurred, 'Blur')

    edged = cv2.Canny(blurred, 0, 50)
    watch(edged, 'Canny edged')

    _, contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # 从大到小对轮廓排序
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # x,y,w,h = cv2.boundingRect(contours[0])
    # cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),0)

    # 从轮廓获得近似多边形
    for c in contours:
        p = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * p, True)

        if len(approx) == 4:
            target = approx
            break

    # 做投影变换，摆正视图，目标视图的比例应该和文档的比例相关
    approx = rectify(target)
    pts2 = np.float32([[0, 0], [800, 0], [800, 800], [0, 800]])

    M = cv2.getPerspectiveTransform(approx, pts2)
    dst = cv2.warpPerspective(image, M, (800, 800))

    cv2.drawContours(image, [target], -1, (0, 255, 0), 2)
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    watch(dst, 'Result')
    watch(image, 'Image with contours')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', default='./demo.jpg')
    parser.add_argument('--mode', default='img', choices=['img', 'webcam'])

    args = parser.parse_args()

    if args.mode == 'img' and not os.path.exists(args.img):
        parser.error("Image not exist.")

    return args


if __name__ == '__main__':
    main(parse_args())
