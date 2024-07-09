import cv2
import numpy as np
import time
import os
from concurrent.futures import ThreadPoolExecutor #多線程並行處理圖片
import math #用來計算Deformability

directory = 'Test_images/Slight under focus'
background_path = os.path.join(directory, 'background.tiff') #獲取background圖片的路徑
files = [f for f in os.listdir(directory) if f.endswith('.tiff') and f != 'background.tiff'] #獲取除了background以外圖片的路徑
background = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE) #以灰度模式讀取background
blur_background = cv2.GaussianBlur(background, (3, 3), 0) #將background進行高斯模糊

def process_image(filename):
    img_path = os.path.join(directory, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) #以灰度模式讀取圖片

    blur_img = cv2.GaussianBlur(img, (3, 3), 0) #將圖片進行高斯模糊
    subtract = cv2.subtract(blur_background, blur_img) #減去背景
    _, binary = cv2.threshold(subtract, 10, 255, cv2.THRESH_BINARY) #進行二值化處理

    kernel = np.ones((3, 3), np.uint8) #3*3卷積核
    erode1 = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel) #進行closing(先侵蝕再膨脹)
    erode2 = cv2.morphologyEx(erode1, cv2.MORPH_OPEN, kernel) #進行opening(先膨脹再侵蝕)

    contours, _ = cv2.findContours(erode2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #尋找輪廓
    #cv2.RETR_EXTERNAL: 僅提取最外層的輪廓, cv2.CHAIN_APPROX_SIMPLE: 輪廓近似、減少內存
    contour_image = np.zeros_like(img) #建立全黑的圖
    cv2.drawContours(contour_image, contours, -1, 255, 1) #在此圖上描繪輪廓

    deformation = None
    if contours:
        cnt = max(contours, key=cv2.contourArea) 
        area = cv2.contourArea(cnt) #獲取輪廓面積
        perimeter = cv2.arcLength(cnt, True) #獲取輪廓周長
        circularity = float(2 * math.sqrt((math.pi) * area)) / perimeter #計算輪廓圓形度
        deformation = 1 - circularity #計算輪廓變形度

    return filename, contour_image, deformation


def main():
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor: #使用ThreadPoolExecutor並行處理files列表中的所有圖像，並將结果儲存在results中
        results = list(executor.map(process_image, files))

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Average time per image: {total_time / len(files):.6f} seconds") #計算時間

    for filename, contour_image, deformation in results:
        cv2.imshow(f'Processed Image: {filename}', contour_image) #顯示輪廓
        if deformation is not None:
            print(f"Deformability for {filename}: {deformation:.6f}") #計算變形度
        print(f"processed image {filename}")
        cv2.waitKey(0) #等待按鍵
        cv2.destroyAllWindows() #關掉所有螢幕

if __name__ == "__main__":
    main() 