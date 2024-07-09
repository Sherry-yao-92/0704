import cv2
import numpy as np
import math
import time
import os

start_time = time.time()
total_duration = 0
count=0
avg=0
for filename in os.listdir('Test_images/Slight under focus'):
    if filename.endswith('.tiff'):
        # 讀取圖片
        img = cv2.imread(os.path.join('Test_images/Slight under focus', filename), cv2.IMREAD_GRAYSCALE)
        background = cv2.imread(os.path.join('Test_images/Slight under focus', 'background.tiff'), cv2.IMREAD_GRAYSCALE)

        single_start_time = time.time()

        blur_img = cv2.GaussianBlur(img, (3, 3), 0)
        blur_background = cv2.GaussianBlur(background, (3, 3), 0)

        substract = cv2.subtract(blur_background, blur_img)

        ret, binary = cv2.threshold(substract, 10, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        erode1 = cv2.erode(binary, kernel)
        dilate1 = cv2.dilate(erode1, kernel)
        dilate2 = cv2.dilate(dilate1, kernel)
        erode2 = cv2.erode(dilate2, kernel)

        edge = cv2.Canny(erode2, 50, 150)

        contours,hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            cnt = contours[0]
            
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            circularity = float(2 * math.sqrt((math.pi) * area)) / perimeter
            deformation = 1 - circularity
            
            print(f'Deformability for {filename}: {deformation}')

        draw_image = np.zeros_like(img)

        contour_image = cv2.drawContours(draw_image, contours, -1, (255,255,255), 1)
        count+=1
        cv2.imshow('Processed image', contour_image)
        cv2.waitKey(0)

        single_end_time = time.time()
        single_duration = single_end_time - single_start_time
        total_duration += single_duration
        print(f'Duration for {filename}: {single_duration} seconds')
#avg = total_duration / count
end_time = time.time()
dif_time = end_time - start_time
avg = dif_time / count
print('count',count)
print('Total Duration: ', dif_time)
print('Average Duration:', avg)
cv2.destroyAllWindows()