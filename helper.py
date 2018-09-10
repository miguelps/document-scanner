import math
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
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

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

    def close_to(self, line, max_dis) -> bool:
        if distance(self.left_point, line.left_point) > max_dis:
            return False
        if distance(self.right_point, line.right_point) > max_dis:
            return False
        return True


def distance(p0, p1):
    return np.linalg.norm(p0 - p1, ord=2)
