import math

import cv2
import numpy as np
from scipy.spatial import distance as dist


# https://www.pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
def clockwise_points(pnts):
    """
    sort clockwise
    :param pnts: numpy array [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    :return: numpy array [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    """
    # sort the points based on their x-coordinates
    xSorted = pnts[np.argsort(pnts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    # 使用下面这两句，在四边形是梯形时会输出错误的结果
    # D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    # (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # 根据 rightMost 的 y 坐标进行排序
    # 如果后面有问题，可以参考 https://github.com/jrosebr1/imutils/pull/13 进行修改
    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (tr, br) = rightMost

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")


class Line:
    def __init__(self, p0, p1):
        """
        :param p0/p1:  numpy array [x, y]
        """
        self.p0 = p0
        self.p1 = p1

        if p0[0] - p1[0] == 0:
            self.k = None
        else:
            self.k = (self.p0[1] - self.p1[1]) / (self.p0[0] - self.p1[0])

        # f = ax+by+c = 0
        self.a = self.p0[1] - self.p1[1]
        self.b = self.p1[0] - self.p0[0]
        self.c = self.p0[0] * self.p1[1] - self.p1[0] * self.p0[1]

    def cross(self, line):
        d = self.a * line.b - line.a * self.b
        if d == 0:
            return None

        x = (self.b * line.c - line.b * self.c) / d
        y = (line.a * self.c - self.a * line.c) / d

        return np.array([x, y])

    def merge(self, line):
        """
        合并另一条直线，p0 和 p1 取两条直线的中点
        """
        new_p0 = self.left_point + line.left_point
        new_p1 = self.right_point + line.right_point
        self.p0 = new_p0 / 2
        self.p1 = new_p1 / 2

    def close_to(self, line, max_dis) -> bool:
        # 这种判断方法是错误的，因为两条直线的左右点可能对应不起来
        # left_dis = distance(self.left_point, line.left_point)
        # if left_dis > max_dis:
        #     return False
        #
        # right_dis = distance(self.right_point, line.right_point)
        # if right_dis > max_dis:
        #     return False

        # TODO: 优化逻辑，减少计算
        left_close = False
        left_dis = distance(self.left_point, line.left_point)
        if left_dis < max_dis:
            left_close = True

        left_dis = distance(self.left_point, line.right_point)
        if left_dis < max_dis:
            left_close = True

        right_close = False
        right_dis = distance(self.right_point, line.left_point)
        if right_dis < max_dis:
            right_close = True

        right_dis = distance(self.right_point, line.right_point)
        if right_dis < max_dis:
            right_close = True

        return left_close and right_close

    def angle_to(self, line):
        return 180 - abs((self.angle - line.angle) % 180)

    @property
    def length(self):
        return distance(self.p0, self.p1)

    @property
    def angle(self):
        # 返回与 x 轴的夹角
        # http://opencv-users.1802565.n2.nabble.com/Angle-between-2-lines-td6803229.html
        # https://stackoverflow.com/questions/2339487/calculate-angle-of-2-points
        angle = math.atan2(self.right_point[1] - self.left_point[1],
                           self.right_point[0] - self.left_point[0])
        angle = angle * 180 / math.pi
        # angle = (int(angle) + 360) % 360
        if angle < 0:
            angle = 180 + angle
        return int(angle)

    @property
    def left_point(self):
        if self.p0[0] < self.p1[0]:
            return self.p0
        elif self.p0[0] > self.p1[0]:
            return self.p1
        else:
            if self.p0[1] > self.p1[1]:
                return self.p0
            else:
                return self.p1

    @property
    def right_point(self):
        if self.p0[0] > self.p1[0]:
            return self.p0
        elif self.p0[0] < self.p1[0]:
            return self.p1
        else:
            if self.p0[1] < self.p1[1]:
                return self.p0
            else:
                return self.p1

    @property
    def int_p0(self):
        return int(self.p0[0]), int(self.p0[1])

    @property
    def int_p1(self):
        return int(self.p1[0]), int(self.p1[1])


def distance(p0, p1):
    return np.linalg.norm(p0 - p1, ord=2)


def draw_four_vectors(img, line, color=(0, 255, 0), draw_text=True):
    """
    :param line: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        矩形四点坐标的顺序： left-top, right-top, right-bottom, left-bottom
    """
    img = cv2.line(img, (line[0][0], line[0][1]), (line[1][0], line[1][1]), color)
    img = cv2.line(img, (line[1][0], line[1][1]), (line[2][0], line[2][1]), color)
    img = cv2.line(img, (line[2][0], line[2][1]), (line[3][0], line[3][1]), color)
    img = cv2.line(img, (line[3][0], line[3][1]), (line[0][0], line[0][1]), color)

    if draw_text:
        # draw corner name
        cv2.putText(img, 'lt', (line[0][0], line[0][1]), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1,
                    color=(0, 0, 255))
        cv2.putText(img, 'rt', (line[1][0], line[1][1]), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1,
                    color=(0, 0, 255))
        cv2.putText(img, 'rb', (line[2][0], line[2][1]), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1,
                    color=(0, 0, 255))
        cv2.putText(img, 'lb', (line[3][0], line[3][1]), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1,
                    color=(0, 0, 255))

        # draw line name
        cv2.putText(img, 'top', (int((line[0][0] + line[1][0]) / 2), int((line[0][1] + line[1][1]) / 2)),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 0, 255))

        cv2.putText(img, 'right', (int((line[1][0] + line[2][0]) / 2), int((line[1][1] + line[2][1]) / 2)),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 0, 255))

        cv2.putText(img, 'bottom', (int((line[2][0] + line[3][0]) / 2), int((line[2][1] + line[3][1]) / 2)),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 0, 255))

        cv2.putText(img, 'left', (int((line[3][0] + line[0][0]) / 2), int((line[3][1] + line[0][1]) / 2)),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 0, 255))
    return img


def cos_angle(v1, v2):
    """
    使用 acos 求两个向量的夹角，注意向量的方向
    v1, v2 为 numpy 数组
    """
    l1 = np.sqrt(v1.dot(v1))
    l2 = np.sqrt(v2.dot(v2))

    cos = v1.dot(v2) / (l1 * l2)
    return np.rad2deg(np.arccos(cos))


def watch(img, name):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 1200 * 2, 800 * 2)
    cv2.imshow(name, img)
    cv2.waitKey()
