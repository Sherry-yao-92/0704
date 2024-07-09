import cv2
import numpy as np
import math
import time

start_time = time.time() #測量時間(起點)

img = cv2.imread('Test_images/Slight under focus/0066.tiff', cv2.IMREAD_GRAYSCALE) #以灰階模式讀取圖片
background = cv2.imread('Test_images/Slight under focus/background.tiff', cv2.IMREAD_GRAYSCALE) #以灰階模式讀取背景

blur_img = cv2.GaussianBlur(img, (3, 3), 0) #圖片模糊化
blur_background = cv2.GaussianBlur(background, (3, 3), 0) #背景模糊化

substract = cv2.subtract(blur_background, blur_img) #減去背景

ret, binary = cv2.threshold(substract, 10, 255, cv2.THRESH_BINARY) #進行二值化

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)) #進行膨脹侵蝕
erode1 = cv2.erode(binary, kernel)
dilate1 = cv2.dilate(erode1, kernel)
dilate2 = cv2.dilate(dilate1, kernel)
erode2 = cv2.erode(dilate2, kernel)

edge = cv2.Canny(erode2, 50, 150) #Canny邊緣檢測

contours,hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #找輪廓

if contours:
    cnt = contours[0]
    
    area = cv2.contourArea(cnt) #計算輪廓面積
    perimeter = cv2.arcLength(cnt, True) #計算輪廓周長
    circularity = float(2 * math.sqrt((math.pi) * area)) / perimeter #計算circularity
    deformation = 1 - circularity #計算Deformability
    
    print('Deformability: ', deformation)

draw_image = np.zeros_like(img) #將輪廓繪製於此塗上

contour_image = cv2.drawContours(draw_image, contours, -1, (255,255,255), 1) #描繪輪廓

cv2.imshow('Processed image',contour_image)

end_time = time.time() #結束時間
dif_time = end_time - start_time #總執行時長
print('Duration: ', dif_time)

cv2.waitKey(0)
cv2.destroyAllWindows()
