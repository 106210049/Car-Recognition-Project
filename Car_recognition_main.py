# -*- coding: utf-8 -*-
# pip install easyocr
#
# pip install opencv-python==4.5.4.60
#
# pip install opencv-contrib-python==4.5.4.60
from PIL import ImageFont, ImageDraw, Image
import numpy as np
from easyocr import Reader
import cv2
import pandas as pd
from Canny import main_canny
# from Gauss import gaussian_blur
# from Gauss import apply_gaussian_blur
# loading images and resizing
def Get_infomation(text,infomation):
    for i in range (infomation.STT.nunique()):
        if np.float64(text[13:19])==(infomation.loc[i,'Mã biển số']):
            print("Thông tin: \n"+f"{infomation.loc[i]}")
            print("Matching Completed")
            break
    return 0

def Matching(text):
    data_path=r'C:\Users\Hi\Desktop\Car.xlsx'
    data_read=pd.read_excel(data_path,sheet_name='Tỉnh')
    imformation=data_read.loc[0,'Thành phố/Tỉnh']
    for i in range (data_read.STT.nunique()):
        if np.int64(text[9:11])==data_read.loc[i,'Mã số biển đầu']:
            print("Thành phố/Tỉnh:"+f"{data_read.loc[i,'Thành phố/Tỉnh']}")
            information=pd.read_excel(data_path,str(data_read.loc[i,'Thành phố/Tỉnh']))
            Get_infomation(text,information)
            continue
    return 0

def Detection(number_plate,img,grayscale):
    reader = Reader(['en'])
    detection = reader.readtext(number_plate)
    if len(detection) == 0:
        text = "No Detection"
        img_pil = Image.fromarray(img) #image biến lấy khung hình từ webcam
        draw = ImageDraw.Draw(img_pil)
        draw.text((150, 500), text, font = font, fill = (b, g, r, a))
        img = np.array(img_pil) #hiển thị ra window
        #cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 3)
        cv2.waitKey(0)
    else:
        cv2.drawContours(img, [number_plate_shape], -1, (255, 0, 0), 3)
        text ="Biển số: " + f"{detection[0][1]}"
        img_pil = Image.fromarray(img) #image biến lấy khung hình từ webcam
        draw = ImageDraw.Draw(img_pil)
        draw.text((200, 500), text, font = font, fill = (b, g, r, a))
        img = np.array(img_pil) #hiển thị ra window
        #cv2.putText(img, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        # cv2.imshow('Plate Detection', img)
        # cv2.waitKey(0)
        # cv2.drawContours(img, [number_plate_shape], -1, (0, 255, 0), 3) 
        mask = np.zeros(grayscale.shape, np.uint8) 
        new_image = cv2.drawContours(mask, [number_plate_shape], 0, 255, -1, ) 
        new_image = cv2.bitwise_and(img, img, mask=mask) 
        (x, y) = np.where(mask == 255) 
        (topx, topy) = (np.min(x), np.min(y)) 
        (bottomx, bottomy) = (np.max(x), np.max(y)) 
        Cropped = grayscale[topx:bottomx + 1, topy:bottomy + 1] 
        cv2.imshow('Input image', img) 
        cv2.imshow('License plate', Cropped) 
        Matching(text)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
        return 0
    
img = cv2.imread('image12.jpg')
img = cv2.resize(img, (800, 600))
# load font
fontpath = "./arial.ttf"
font = ImageFont.truetype(fontpath, 32)
b,g,r,a = 0,255,0,0
# making the image grayscale
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
edged = cv2.Canny(blurred, 10, 200)
# edged=main_canny(blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()

contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]

for c in contours:
    perimeter = cv2.arcLength(c, True)
    approximation = cv2.approxPolyDP(c, 0.02 * perimeter, True)
    print(approximation)
    if len(approximation) == 4: # rectangle
        number_plate_shape = approximation
        break

(x, y, w, h) = cv2.boundingRect(number_plate_shape)
number_plate = grayscale[y:y + h, x:x + w]

Detection(number_plate,img,grayscale)


