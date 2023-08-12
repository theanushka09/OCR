import cv2
from PIL import Image
import pytesseract
import numpy as np

def getskewAngle(cvImage) -> float:
    newImage = cvImage.copy()
    # gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(newImage, (9, 9), 0)
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
    mainAreaRect = cv2.minAreaRect(largestContour)
    angle = mainAreaRect[-1]
    # print(angle)
    if angle < -45:
        angle = 90 + angle
    if(abs(angle)<=5):
        return angle
    return -1.0 * (90-angle)

def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h,w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

def getText(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel)
    mser = cv2.MSER_create()
    vis = img.copy()
    regions, _ = mser.detectRegions(img)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv2.polylines(vis, hulls, 1, (0,255, 0))
    cv2.imshow('hulls', vis)
    cv2.imwrite('hulls.jpg',vis)
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
        cv2.rectangle(mask,(x-(width//50),y-(height//50)),(x+w+(width//50),y+h+(height//50)),(0,255,0), cv2.FILLED) 
    cv2.imshow('mask', mask)
    cv2.imwrite('boxes.jpg',mask)
    cv2.waitKey(0)
    contours,hierarchy = cv2.findContours(mask, 1, 2)
    i=0;
    for c in contours:
        i+=1
        x,y,w,h = cv2.boundingRect(c)
        if w> (height/2) and h > (width/2):
            continue
        subImg = img[y:y+h, x:x+w]
        cv2.imshow("subImg"+str(i),subImg);
        cv2.waitKey(0);
        rotatedImg=rotateImage(subImg,getskewAngle(subImg))
        cv2.imshow("rotated"+str(i),rotatedImg);
        cv2.waitKey(0)
        newText=pytesseract.image_to_data(rotatedImg, output_type='data.frame');
        newText = newText[newText.conf >80]
        lines = newText.groupby('block_num')['text'].apply(list)
        block = newText.groupby('page_num')['block_num'].apply(list)
        try:
            block[1] =list(dict.fromkeys(block[1]))
            for x in block[1]:
                print(*lines[x])
        except:
            pass

img = cv2.imread(r'D:\projects\ocr\ins.jpeg')
getText(img)
