# import the necessary packages
import argparse
import pytesseract
import numpy as np
import cv2
import imutils
import os
import math


def order_points(points):
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = points.sum(axis=1)
    ordered[0] = points[np.argmin(s)]
    ordered[2] = points[np.argmax(s)]
    d = np.diff(points, axis=1)
    ordered[1] = points[np.argmin(d)]
    ordered[3] = points[np.argmax(d)]
    return ordered


def random_color():
    color = np.random.randint(0, 255, 3)
    return tuple([int(i) for i in color])


def transform(image, points):
    rect = order_points(points)
    tl, tr, br, bl = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1])**2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1])**2))
    max_width = int(max(widthA, widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1])**2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1])**2))
    max_height = int(max(heightA, heightB))
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped


def filter_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 0, 160], dtype="uint8")
    upper = np.array([170, 140, 255], dtype="uint8")
    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask


def draw_contours(edges, image):
    image = image.copy()
    contours = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, peri*0.02, True)
        cv2.drawContours(image, [approx], -1, random_color(), 2)
    cv2.imshow('ctrs', image)


def get_contour(image):
    contours = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    check = None
    best_area = 0
    for c in contours:
        perimeter = cv2.arcLength(c, False)
        approx = cv2.approxPolyDP(c, .01 * perimeter, False)
        min_rect = cv2.minAreaRect(c)
        sides = min_rect[1]
        area = sides[0] * sides[1]
        if area < best_area:
            continue
        best_area = area
        check = approx
    return check


def getOrientation(pts, img):
    # [pca]
    # Construct a buffer used by the pca analysis
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]

    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)

    angle = math.atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians
    return angle


def process_image(file):
    image = cv2.imread(file)

    cv2.imshow("Orignal Image", image)

    rot_data = pytesseract.image_to_osd(file, config='-c min_characters_to_try=10', output_type=pytesseract.Output.DICT)
    degrees = rot_data['orientation']
    conf = rot_data['orientation_conf']
    if conf > .3:
        image = imutils.rotate_bound(image, degrees)
    image = imutils.resize(image, width=1000, inter=cv2.INTER_LINEAR)

    mask = filter_image(image)
    # cv2.imshow("mks", mask)
    masked = cv2.bitwise_and(image, image, mask=mask)

    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    # gray = cv2.bitwise_and(gray, gray, mask=mask)
    # gray = cv2.bitwise_not(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edge = cv2.Canny(gray, 75, 200, 5)

    # draw_contours(edge, masked)

    check = get_contour(mask)
    angle = getOrientation(check, image) / math.pi * 180
    if abs(angle) > 45 and conf < .1:
        image = imutils.rotate_bound(image, degrees)
        edge = imutils.rotate_bound(edge, degrees)
        check = get_contour(edge)

    # color = (0, 0, 255)
    # cv2.drawContours(image, [check], -1, color, 2)
    # cv2.imshow("Outline", image)

    warped = transform(image, check.reshape(-1, 2))
    cv2.imshow('Transformed Image', warped)

    # k = 0
    # while k != 113:
    #     k = cv2.waitKey(0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Check prepartion project")
    parser.add_argument('--input_folder', type=str, default='samples', help='check images folder')

    args = parser.parse_args()
    input_folder = args.input_folder

    for check_img in os.listdir(input_folder):
        img_path = os.path.join(input_folder, check_img)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            process_image(img_path)


if __name__ == "__main__":
    main()
