import sys
import cv2
import numpy as np
import imutils


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


def process(image):
    # add image here.
    # We can also use laptop's webcam if the resolution is good enough to capture
    # readable document content

    # resize image so it can be processed
    # choose optimal dimensions such that important content is not lost
    # print(image.shape)
    image = imutils.resize(image, 600)
    # print(image.shape)

    # creating copy of original image
    orig = image.copy()

    # convert to grayscale and blur to smooth
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # blurred = cv2.medianBlur(gray, 5)

    # apply Canny Edge Detection
    edged = cv2.Canny(blurred, 0, 40)

    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    contours = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # x,y,w,h = cv2.boundingRect(contours[0])
    # cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),0)

    # get approximate contour
    for c in contours:
        p = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * p, True)

        if len(approx) == 4:
            target = approx
            break

    # mapping target points to 800x800 quadrilateral
    approx = rectify(target)
    pts2 = np.float32([[0, 0], [800, 0], [800, 800], [0, 800]])

    M = cv2.getPerspectiveTransform(approx, pts2)
    dst = cv2.warpPerspective(orig, M, (800, 800))

    cv2.drawContours(image, [target], -1, (0, 255, 0), 2)
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    # using thresholding on warped image to get scanned effect (If Required)
    th1 = cv2.threshold(dst, 127, 255, cv2.THRESH_BINARY)[1]
    th2 = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 11, 2)
    th3 = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)
    th4 = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    cv2.imshow("Original.jpg", orig)
    cv2.imshow("Gray.jpg", gray)
    cv2.imshow("Blurred.jpg", blurred)
    cv2.imshow("Edged.jpg", edged)
    cv2.imshow("Outline.jpg", image)
    cv2.imshow("Thresh Binary.jpg", th1)
    cv2.imshow("Thresh mean.jpg", th2)
    cv2.imshow("Thresh gauss.jpg", th3)
    cv2.imshow("Otsu's.jpg", th4)
    cv2.imshow("dst.jpg", dst)

    # other thresholding methods
    """
    thresh1 = cv2.threshold(dst,127,255,cv2.THRESH_BINARY)[1]
    thresh2 = cv2.threshold(dst,127,255,cv2.THRESH_BINARY_INV)[1]
    thresh3 = cv2.threshold(dst,127,255,cv2.THRESH_TRUNC)[1]
    thresh4 = cv2.threshold(dst,127,255,cv2.THRESH_TOZERO)[1]
    thresh5 = cv2.threshold(dst,127,255,cv2.THRESH_TOZERO_INV)[1]

    cv2.imshow("Thresh Binary", thresh1)
    cv2.imshow("Thresh Binary_INV", thresh2)
    cv2.imshow("Thresh Trunch", thresh3)
    cv2.imshow("Thresh TOZERO", thresh4)
    cv2.imshow("Thresh TOZERO_INV", thresh5)
    """

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return dst


def main():
    img_name = sys.argv[1]
    # img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
    img = cv2.imread(img_name)
    process(img)


if __name__ == '__main__':
    main()
