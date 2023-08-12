from PIL import Image
import pytesseract
import numpy as np 
import cv2
import pandas


def getText(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text="";
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel)
    mser = cv2.MSER_create()
    vis = img.copy()
    regions, _ = mser.detectRegions(img)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv2.polylines(vis, hulls, 1, (0,255,0))
    cv2.imshow('hulls', vis)
    cv2.waitKey(0)
    mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
    for contour in hulls:
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
    mask = cv2.bitwise_not(mask)
    dimensions = img.shape
    height = img.shape[0]
    width = img.shape[1]
    contours,hierarchy = cv2.findContours(mask, 1, 2)
    for c in contours:
        rect = cv2.boundingRect(c)
        if rect[2] > (height/3) and rect[3] > (width/3):
            continue
        x,y,w,h = rect
    cv2.rectangle(mask,(x-(width//80),y-(height//80)),(x+w+(width//80),y+h+(height//80)),(0,255,0), cv2.FILLED) 
    cv2.imshow('mask', mask)
    cv2.waitKey(0)
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w> (height/2) and h > (width/2):
            continue

def getskewAngle(cvImage) -> float:

    newImage = cvImage.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur,0,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    contours,hierarchy = cv2.findContours(dilate, cv2.RETR_LIST , cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    for c in contours:
        rect = cv2.boundingRect(c)
        x,y,w,h = rect
        cv2.rectangle(newImage,(x,y),(x+w,y+h),(0,255,0),2)

    largestContour =contours[0]
    print(len(contours))
    minAreaRect = cv2.minAreaRect(largestContour)
    cv2.imwrite("img.jpg",newImage)
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * (90-angle)

def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h,w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return newImage
img = cv2.imread("D:\softwaRE\ocr\hello.jpeg")
cv2.imshow('img1', img)
cv2.waitKey(0)
img=rotateImage(img,getskewAngle(img))
cv2.imshow('rotateimage',img)
cv2.waitKey(0)