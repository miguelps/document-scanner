"""
Ref： http://fengjian0106.github.io/2017/05/08/Document-Scanning-With-TensorFlow-And-OpenCV/
"""

from itertools import combinations
import argparse
import os

from helper import *

DEBUG = False

MAX_LENGTH = 1200  # resize long side to MAX_LENGTH

# Pixels smaller than this threshold will not be considered as edges
CANNY_THRESH_MIN = 10
CANNY_THRESH_MAX = 50  # Pixels larger than this threshold will be considered edges

CONTOURS_LENGTH_THRESH = 200
CONTOURS_AREA_THRESH = 50

HOUGH_THRESH = 150
HOUGH_MIN_LINE_LENGTH = 20
HOUGH_MAX_LINE_GAP = 50
LINE_LENGTH_THRESH = 150  # Filter the results of Hough line detection

# Merging closer lines
MERGE_LINE_MAX_DISTANCE = 50

# If two intersections are very close, they merge into one
MERGE_CLOSE_CROSS_POINT_MIN_DISTANCE = 20

# Among the two opposite sides of the quadrilateral, the ratio of
# the smaller value/the larger value cannot be less than this value
DOC_SIDE_MIN_RATIO = 0.8
# The ratio of adjacent sides of the quadrilateral cannot exceed
# this value at most
DOC_SIDE_MAX_RATIO = 3

# The angle difference between the diagonal corners of the
# quadrilateral cannot exceed this value at most
DOC_ANGLE_MAX_DIFF = 30

# Quadrilateral angle constraint
INTERSECTION_ANGLE_MIN = 60
INTERSECTION_ANGLE_MAX = 110


def main(args):
    image = cv2.imread(args.img)
    print("Origin size: %s" % str(image.shape))

    # Scale the picture to a certain scale, otherwise the
    # adjusted parameters will be difficult to work
    scale = 1200 / max(image.shape)
    image = cv2.resize(image, None, None, fx=scale, fy=scale)
    height = image.shape[0]
    width = image.shape[1]
    watch(image, 'Resized Origin', True)
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
    watch(edged, 'Canny edged')

    contours = cv2.findContours(
        edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    print("Contours num: %d" % len(contours))

    filtered_contours = []
    contours_img = np.zeros(edged.shape, dtype=np.uint8)
    for c in contours:
        arcLen = cv2.arcLength(c, True)
        area = cv2.contourArea(c)
        # calculated document area may be small
        if arcLen < CONTOURS_LENGTH_THRESH and area < CONTOURS_AREA_THRESH:
            continue

        filtered_contours.append(c)
        cv2.drawContours(contours_img, [c], -1, (255, 255, 255), 3)

    print("Filtered contours num: %d" % len(filtered_contours))
    watch(contours_img, 'Contours image')

    # Hough transform to detect straight lines on contour map
    lines = cv2.HoughLinesP(contours_img, 1, np.pi / 180,
                            HOUGH_THRESH, HOUGH_MIN_LINE_LENGTH, HOUGH_MAX_LINE_GAP)
    filterd_lines = []
    if lines is not None:
        print("Hough Lines num: %d" % len(lines))

        lines_img = image.copy()
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Filter out lines that are too short
            if np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) < LINE_LENGTH_THRESH:
                continue
            filterd_lines.append(line[0])
            cv2.line(lines_img, (x1, y1), (x2, y2), (255, 0, 0), 1)

        print("Filtered Hough Lines num: %d" % len(filterd_lines))
        watch(lines_img, "Hough Lines")

    # Obtain the extension of the straight line
    extended_lines = []
    extended_lines_img = image.copy()
    for line in filterd_lines:
        extend_x1, extend_y1, extend_x2, extend_y2 = get_extend_line(
            line, height, width)
        cv2.line(extended_lines_img, (extend_x1, extend_y1),
                 (extend_x2, extend_y2), (0, 255, 0), 1)

        p0 = np.array([extend_x1, extend_y1])
        p1 = np.array([extend_x2, extend_y2])
        extended_lines.append(Line(p0, p1))
    watch(extended_lines_img, "Extended Lines")

    # Merge straight lines with similar slopes and close to each other
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
        cv2.line(merged_lines_img, line.int_p0, line.int_p1, color=(0, 255, 0))

    print("Lines num after merge: %d" % len(merged_extended_lines))
    watch(merged_lines_img, "Merged Lines")

    # Obtain the intersection of the extension line of the line segment
    cross_pnts = []
    cross_pnts_lines = []
    for i in range(len(merged_extended_lines)):
        line = merged_extended_lines[i]
        for j in range(i + 1, len(merged_extended_lines)):
            _line = merged_extended_lines[j]
            point = line.cross(_line)
            if point is not None and point_valid(point, height, width):
                cross_pnts.append(point)
                cross_pnts_lines.append((line, _line))
                cv2.circle(merged_lines_img, (int(point[0]), int(
                    point[1])), 5, color=(0, 0, 255), thickness=2)

    print("Cross points num: %d" % len(cross_pnts))
    watch(merged_lines_img, "Extended Lines cross point")

    # Merge adjacent intersections
    merged_cross_pnts = []
    merged_cross_pnts_lines = []
    merged_pnts = {}
    for i in range(len(cross_pnts)):
        if i in merged_pnts:
            continue

        pnt = cross_pnts[i]
        for j in range(i + 1, len(cross_pnts)):
            if j in merged_pnts:
                continue

            _pnt = cross_pnts[j]
            if distance(pnt, _pnt) < MERGE_CLOSE_CROSS_POINT_MIN_DISTANCE:
                merged_pnts[j] = j
                cross_pnts[i] = (pnt + _pnt) / 2

        cv2.circle(merged_lines_img, (int(pnt[0]), int(
            pnt[1])), 5, color=(0, 255, 255), thickness=2)
        merged_cross_pnts.append(cross_pnts[i])
        merged_cross_pnts_lines.append(cross_pnts_lines[i])

    print("Merged cross points num: %d" % len(merged_cross_pnts))
    watch(merged_lines_img, "Merged cross points")

    # Take four points at a time and arrange them clockwise to filter
    # out unreasonable quadrilaterals
    rect_pnts = []
    valid_pnts_img = image.copy()
    for data in combinations(zip(merged_cross_pnts, merged_cross_pnts_lines), 4):
        pnts = []
        lines = []
        for d in data:
            pnts.append(d[0])
            lines.append(d[1][0])
            lines.append(d[1][1])

        pnts = clockwise_points(np.asarray(list(pnts)))
        # TODO: Determine whether it is a convex polygon?
        top_line = Line(pnts[0], pnts[1])
        right_line = Line(pnts[1], pnts[2])
        bottom_line = Line(pnts[2], pnts[3])
        left_line = Line(pnts[3], pnts[0])

        # tmp = image.copy()
        # tmp = draw_four_vectors(tmp, pnts)
        # watch(tmp, "trbl rects")

        # Judge whether the four sides are close to the straight line forming the
        # corner point, as long as one side is not satisfied, continue
        if not line_valid(top_line, lines, height, width):
            continue
        if not line_valid(right_line, lines, height, width):
            continue
        if not line_valid(bottom_line, lines, height, width):
            continue
        if not line_valid(left_line, lines, height, width):
            continue
        print("===Pass line_valid===")

        # Opposite length difference constraint
        if min(top_line.length, bottom_line.length) / max(top_line.length, bottom_line.length) < DOC_SIDE_MIN_RATIO:
            continue

        if min(left_line.length, right_line.length) / max(left_line.length, right_line.length) < DOC_SIDE_MIN_RATIO:
            continue
        print("===Pass doc min_side/max_side ratio===")

        # Adjacent edge ratio constraint
        if max(top_line.length, right_line.length) / min(top_line.length, right_line.length) > DOC_SIDE_MAX_RATIO:
            continue
        if max(top_line.length, left_line.length) / min(top_line.length, left_line.length) > DOC_SIDE_MAX_RATIO:
            continue
        if max(bottom_line.length, right_line.length) / min(bottom_line.length, right_line.length) > DOC_SIDE_MAX_RATIO:
            continue
        if max(bottom_line.length, left_line.length) / min(bottom_line.length, left_line.length) > DOC_SIDE_MAX_RATIO:
            continue
        print("===Pass doc max/min ratio===")

        # Angle constraint
        top_left_angle = cos_angle(pnts[1] - pnts[0], pnts[3] - pnts[0])
        top_right_angle = cos_angle(pnts[0] - pnts[1], pnts[2] - pnts[1])
        bottom_right_angle = cos_angle(pnts[1] - pnts[2], pnts[3] - pnts[2])
        bottom_left_angle = cos_angle(pnts[0] - pnts[3], pnts[2] - pnts[3])

        print("*" * 20)
        print("Top left angle %f" % top_left_angle)
        print("Top right angle %f" % top_right_angle)
        print("Bottom right angle %f" % bottom_right_angle)
        print("Bottom left angle %f" % bottom_left_angle)

        if not angle_valid(top_left_angle):
            continue
        if not angle_valid(top_right_angle):
            continue
        if not angle_valid(bottom_right_angle):
            continue
        if not angle_valid(bottom_left_angle):
            continue
        print("===Pass angle_valid===")

        print("-" * 20)
        print("Top line angle %f" % top_line.angle)
        print("Right line angle %f" % right_line.angle)
        print("Bottom line angle %f" % bottom_line.angle)
        print("Left line angle %f" % left_line.angle)

        # Constrain the difference between the two diagonals
        angle_diff1 = abs(top_left_angle - bottom_right_angle)
        angle_diff2 = abs(top_right_angle - bottom_left_angle)

        print("-" * 20)
        print("Angle diff 1: %f" % angle_diff1)
        print("Angle diff 2: %f" % angle_diff2)

        if angle_diff1 > DOC_ANGLE_MAX_DIFF:
            continue
        if angle_diff2 > DOC_ANGLE_MAX_DIFF:
            continue
        print("===Pass angle diff===")

        rect_pnts.append(pnts)
        valid_pnts_img = draw_four_vectors(valid_pnts_img, pnts)

    print("Valid rect pnts group: %d" % len(rect_pnts))
    watch(valid_pnts_img, "Valid rects")

    if len(rect_pnts) == 0:
        print("Not found valid document")
        return

    # TODO： Further add constraint strategies, such as area,
    # whether it is in the center of the picture, the overall
    # rotation angle, etc.

    rect = rect_pnts[0]
    min_rect = cv2.minAreaRect(rect)
    min_rect = cv2.boxPoints(min_rect)
    min_rect = clockwise_points(min_rect)

    scale = abs(min_rect[1][1] - min_rect[2][1]) / \
        abs(min_rect[0][0] - min_rect[1][0])
    result_w = 800
    result_h = int(result_w * scale)

    # Perform projection transformation, straighten the view, the scale
    # of the target view should be related to the scale of the document
    pts2 = np.float32(
        [[0, 0], [result_w, 0], [result_w, result_h], [0, result_h]])

    M = cv2.getPerspectiveTransform(rect, pts2)
    dst = cv2.warpPerspective(image, M, (result_w, result_h))
    watch(dst, 'Result', True)


def angle_valid(angle):
    return INTERSECTION_ANGLE_MIN <= angle <= INTERSECTION_ANGLE_MAX


def point_valid(pnt, height, width):
    """
    Whether the point coordinates are inside the picture
    """
    if pnt[0] > width or pnt[0] < 0:
        return False
    if pnt[1] > height or pnt[1] < 0:
        return False
    return True


def line_valid(line, lines, height, width):
    """
    If line coincides with any one of lines, return True,
    otherwise return False
    This function will extend the line first, and the
    incoming lines should be considered to have been extended
    """
    line = [line.p0[0], line.p0[1], line.p1[0], line.p1[1]]
    extend_line = get_extend_line(line, height, width)
    p0 = np.array(extend_line[0:2])
    p1 = np.array(extend_line[2:4])
    extend_line = Line(p0, p1)
    for l in lines:
        if extend_line.close_to(l, MERGE_LINE_MAX_DISTANCE):
            return True
    return False


def get_extend_line(line, height, width):
    """
    Extend the line segment to the edge of the picture
    :param line: Straight segment to be extended [x1, y1, x2, y2]
    :param height: Image height
    :param width: Image width
    :return:
        (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = line

    if x1 == x2:
        # Vertical line
        extend_x1 = extend_x2 = x1
        extend_y1 = 0
        extend_y2 = height
    elif y2 == y1:
        # Horizontal line
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

        extend_x1, extend_y1, extend_x2, extend_y2 = int(
            extend_x1), int(extend_y1), int(extend_x2), int(extend_y2)

    return extend_x1, extend_y1, extend_x2, extend_y2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', default='./images/test.jpg')
    parser.add_argument('--mode', default='img', choices=['img', 'webcam'])

    args = parser.parse_args()

    if args.mode == 'img' and not os.path.exists(args.img):
        parser.error("Image not exist.")

    return args


if __name__ == '__main__':
    main(parse_args())
