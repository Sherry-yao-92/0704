import cv2
import numpy as np
import math
import time
import os

# 主程式
directory = 'Test_images/Slight under focus'
background_path = os.path.join(directory, 'background.tiff')
files = [f for f in os.listdir(directory) if f.endswith('.tiff') and f != 'background.tiff']
background = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)

start_time = time.time()

results = {}
deformabilities = {}

for filename in files:
    # 讀取圖片
    img_path = os.path.join(directory, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # 處理圖像
    blur_img = cv2.GaussianBlur(img, (3, 3), 0)
    blur_background = cv2.GaussianBlur(background, (3, 3), 0)
    substract = cv2.subtract(blur_background, blur_img)
    ret, binary = cv2.threshold(substract, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    erode1 = cv2.erode(binary, kernel)
    dilate1 = cv2.dilate(erode1, kernel)
    dilate2 = cv2.dilate(dilate1, kernel)
    erode2 = cv2.erode(dilate2, kernel)

    # 尋找輪廓
    edge = cv2.Canny(erode2, 50, 150)
    contours, _ = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = np.zeros_like(img)
    cv2.drawContours(contour_image, contours, -1, (255,255,255), 1)

    # 計算變形度
    if contours:
        cnt = contours[0]
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circularity = float(2 * math.sqrt((math.pi) * area)) / perimeter
        deformation = 1 - circularity
        deformabilities[filename] = deformation

    # 儲存結果
    results[filename] = contour_image

end_time = time.time()
total_time = end_time - start_time
print(f"Total execution time: {total_time:.2f} seconds")
print(f"Average time per image: {total_time / len(files):.6f} seconds")

# 顯示結果
for image_name, contour_image in results.items():
    cv2.imshow(f'Processed Image: {image_name}', contour_image)
    if image_name in deformabilities:
        print(f"Deformability for {image_name}: {deformabilities[image_name]:.6f}")
    print(f"{image_name}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()