# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 16:55:27 2021

@author: ghdtm
"""
import numpy as np
import cv2
import argparse
import os
from datetime import datetime

dir_del = None
clicked_points = []
clone = None
def draw_grid(img, grid_shape, color=(128, 128, 128), thickness=1):
    h, w, _ = img.shape
    rows, cols = grid_shape
    dy, dx = h / rows, w / cols

    # draw vertical lines
    for x in np.linspace(start=dx, stop=w-dx, num=cols-1):
        x = int(round(x))
        cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)

    # draw horizontal lines
    for y in np.linspace(start=dy, stop=h-dy, num=rows-1):
        y = int(round(y))
        cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)

    return img

def flatten(points,img,ratio,file_name):
    temp = []
    for y,x in points:
        temp.append([int(abs(y*ratio)),int(abs(x*ratio))])
    temp = sorted(temp,key=lambda x: x[0])
    
    tmp_a = sorted(temp[:2],key=lambda x:x[1])
    
    tmp_b = sorted(temp[2:4],key=lambda x:x[1])
    
    try:
        mean_x = (abs(temp[0][1]-temp[1][1]) + abs(temp[2][1]-temp[3][1])) // 2 + 3
        
        mean_y = (abs(temp[2][0]-temp[0][0]) + abs(temp[3][0]-temp[1][0])) // 2 + 3
        
        a = [[tmp_a[0][1]-3,tmp_a[0][0]+3],[tmp_a[1][1]-3,tmp_a[1][0]+3],[tmp_b[0][1]-3,tmp_b[0][0]+3],[tmp_b[1][1]-3,tmp_b[1][0]+3]]
        b = [[0,0],[mean_x,0],[0,mean_y],[mean_x,mean_y]]
    except:
        print("ERROR")
    pts1 = np.float32(a)
    pts2 = np.float32(b)
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(mean_x,mean_y))
    cv2.imwrite('./'+file_name, dst)
    
def MouseLeftClick(event, x, y, flags, param):
	# 왼쪽 마우스가 클릭되면 (x, y) 좌표를 저장한다.
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((y, x))

		# 원본 파일을 가져 와서 clicked_points에 있는 점들을 그린다.
        image = clone.copy()
        for point in clicked_points:
            cv2.circle(image, (point[1], point[0]), 2, (0, 0, 255), thickness = 2)
        cv2.imshow("image", image)
def opencv_resize(image, ratio):
    width = int(image.shape[1] * ratio)
    height = int(image.shape[0] * ratio)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
def GetArgument():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Enter the image files path")
    ap.add_argument("--sampling", default=1, help="Enter the sampling number.(default = 1)")
    args = vars(ap.parse_args())
    path = args['path']
    sampling = int(args['sampling'])
    return path, sampling

def main():
    global clone, clicked_points
    
    print("\n")
    print("1. 입력한 파라미터인 이미지 경로(--path)에서 이미지들을 차례대로 읽어옵니다.")
    print("2. 키보드에서 'n'을 누르면(next 약자) 다음 이미지로 넘어갑니다. 이 때, 작업한 점의 좌표가 저장 됩니다.")
    print("3. 키보드에서 'b'를 누르면(back 약자) 직전에 입력한 좌표를 취소한다.")
    print("4. 이미지 경로에 존재하는 모든 이미지에 작업을 마친 경우 또는 'q'를 누르면(quit 약자) 프로그램이 종료됩니다.")
    print("\n")
    print("출력 포맷 : 이미지명,y1,x1,y2,x2,...")
    print("\n")

    # 이미지 디렉토리 경로를 입력 받는다.
    path, sampling = GetArgument()
    # path의 이미지명을 받는다.
    image_names = os.listdir(path)

    # path를 구분하는 delimiter를 구한다.
    if len(path.split('\\')) > 1:
        dir_del = '\\'
    else :
        dir_del = '/'

    # path에 입력된 마지막 폴더 명을 구한다.    
    folder_name = path.split(dir_del)[-1]

    # 결과 파일을 저장하기 위하여 현재 시각을 입력 받는다.
    now = datetime.now()
    now_str = "%s%02d%02d_%02d%02d%02d" % (now.year - 2000, now.month, now.day, now.hour, now.minute, now.second)   

    # 새 윈도우 창을 만들고 그 윈도우 창에 click_and_crop 함수를 세팅해 줍니다.
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", MouseLeftClick)

    for idx, image_name in enumerate(image_names):
        
        if (idx % sampling != 0):
            continue

        image_path = path + dir_del + image_name
        o_image = cv2.imread(image_path)
        resize_ratio = 1000 / o_image.shape[0]
        ratio = pow(resize_ratio,-1)
        print(ratio)
        image = opencv_resize(o_image, resize_ratio)
        image = draw_grid(image,(20,10))
        clone = image.copy()

        flag = False

        while True:
            cv2.imshow("image", image)
            key = cv2.waitKey(0)

            if key == ord('n'):
                if len(clicked_points) != 4:
                    print("꼭지점 4점을 클릭해주세요")
                    image = clone.copy()
                    for point in clicked_points:
                        cv2.circle(image, (point[1],point[0]), 2, (0,0,255),thickness=2)
                    cv2.imshow("image",image)
                    continue
                # 텍스트 파일을 출력 하기 위한 stream을 open 합니다.
                # 중간에 프로그램이 꺼졌을 경우 작업한 것을 저장하기 위해 쓸 때 마다 파일을 연다.
                file_write = open('./' + now_str + '_' + folder_name + '.txt', 'a+')

                text_output = image_name
                #text_output += "," + str(len(clicked_points))
                for points in clicked_points:
                    text_output += "," + str(points[0]) + "," + str(points[1])
                text_output += '\n'
                file_write.write(text_output)
                flatten(clicked_points,o_image,ratio,image_name)
                # 클릭한 점 초기화
                clicked_points = []

                # 파일 쓰기를 종료한다.
                file_write.close()

                break

            if key == ord('b'):
                if len(clicked_points) > 0:
                    clicked_points.pop()
                    image = clone.copy()
                    for point in clicked_points:
                        cv2.circle(image, (point[1], point[0]), 2, (0, 0, 255), thickness = 2)
                    cv2.imshow("image", image)

            if key == ord('q'):
                # 프로그램 종료
                flag = True
                break

        if flag:
            break

    # 모든 window를 종료합니다.
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()