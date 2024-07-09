import cv2
import numpy as np
import math
import time

def process_image(img, background):
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
    contours, _ = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    deformation = None
    if contours:
        cnt = contours[0]
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circularity = float(2 * math.sqrt((math.pi) * area)) / perimeter
        deformation = 1 - circularity

    draw_image = np.zeros_like(img)
    contour_image = cv2.drawContours(draw_image, contours, -1, (255,255,255), 1)

    return contour_image, deformation

# 讀取圖片
img = cv2.imread('Test_images/Slight under focus/0066.tiff', cv2.IMREAD_GRAYSCALE)
background = cv2.imread('Test_images/Slight under focus/background.tiff', cv2.IMREAD_GRAYSCALE)

# 重複處理圖像多次以獲得更可靠的時間測量
num_iterations = 100
start_time = time.perf_counter()

for _ in range(num_iterations):
    contour_image, deformation = process_image(img, background)

end_time = time.perf_counter()

# 計算總執行時間和平均時間
total_time = end_time - start_time
avg_time = total_time / num_iterations

# 輸出時間統計
print(f"Total execution time for {num_iterations} iterations: {total_time:.6f} seconds")
print(f"Average time per iteration: {avg_time:.6f} seconds")

# 輸出變形度（如果有計算的話）
if deformation is not None:
    print(f"Deformability: {deformation:.6f}")

# 顯示結果（這部分不計入執行時間）
cv2.imshow('Processed image', contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()