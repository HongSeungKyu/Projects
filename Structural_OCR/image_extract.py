import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt
from PIL import Image
import mrcnn
import mrcnn.config
import mrcnn.model as MD
import mrcnn.visualize
import cv2
import os
import numpy
import numpy as np
from PIL import Image, ImageDraw

CLASS_NAMES = ['BG', 'figure','formula']

class SimpleConfig(mrcnn.config.Config):
    NAME = "coco_inference"
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    NUM_CLASSES = len(CLASS_NAMES)

def merge_boxes(results_rois,results_masks):
    line = len(results_rois)
    boxes = list()
    for box in results_rois:
        ymin = box[0]
        xmin = box[1]
        ymax = box[2]
        xmax = box[3]

        coors = [[xmin, ymin],[xmax,ymin], [xmax, ymax],[xmin,ymax]]

        boxes.append(coors)

    size = list(results_masks.shape[:2])
    size.append(3)

    stencil1 = numpy.zeros(size).astype(np.dtype("uint8"))
    stencil2= numpy.zeros(size).astype(np.dtype("uint8"))

    color = [255, 255, 255]

    for i in range(len(boxes)):
        stencil1 = numpy.zeros(size).astype(np.dtype("uint8"))

        contours = [numpy.array(boxes[i])]
        cv2.fillPoly(stencil1, contours, color)


        for j in range(i+1,len(boxes)):
            stencil2= numpy.zeros(size).astype(np.dtype("uint8"))
            contours = [numpy.array(boxes[j])]
            cv2.fillPoly(stencil2, contours, color)


            intersection = np.sum(numpy.logical_and(stencil1, stencil2))
        
            if intersection > 0:
                xmin = min(boxes[i][0][0],boxes[j][0][0])
                ymin = min(boxes[i][0][1],boxes[j][0][1])
                xmax = max(boxes[i][2][0],boxes[j][2][0])
                ymax = max(boxes[i][2][1],boxes[j][2][1])

                '''
                coors = [[xmin, ymin],[xmax,ymin], [xmax, ymax],[xmin,ymax]]
                '''
                print(" {},{} INTERSECTION : {}".format(i,j,np.sum(intersection)))

                results_rois[i] = [ymin,xmin,ymax,xmax]
                arr = np.delete(results_rois,j,0)
                
                return merge_boxes(arr,results_masks)

    return results_rois

def extract_Figures(model,pil_image):
    
    image=np.array(pil_image)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    r = model.detect([image], verbose=0)
    r = r[0]
    merged = merge_boxes(r['rois'],r['masks'])
    extract_imgs = list()
    for i in merged:

        
        #  cropped_img = img[y: y + h, x: x + w]
        cropped_img = image[i[0]:i[2], i[1]: i[3]]
        extract_imgs.append(Image.fromarray(cropped_img))
    image = Image.fromarray(image)
    for i in merged:
        print(i)
        shape = [(i[1], i[0]), (i[3],i[2])]
        
        img1 = ImageDraw.Draw(image)
        img1.rectangle(shape, fill ="#FFFFFF")
    
    return extract_imgs , image




def mse(A, B):
    err = np.sum((A.astype("float") - B.astype("float")) ** 2)
    err /= float(A.shape[0] * A.shape[1])


def extract_from_video(model,video_dir):
    
    video_dir = '/Users/jeong-wonlyeol/Desktop/캡스톤/youtube/youtube_26.mp4'



    cnt = 1

    cap = cv2.VideoCapture(video_dir)

    FPS = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / FPS

    second = 1
    cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
    success, frame = cap.read()

    plt.imshow(frame)
    plt.show()
    pil_Image = Image.fromarray(frame)


    image_deleted_list = list()
    image_cropped_list = list()
    print(cnt)
    image_cropped , image_deleted = extract_Figures(model,pil_Image)
    image_deleted_list.append(image_deleted)
    image_cropped_list.append(image_cropped)

    num = 0
    increase_width = 3

    while success and second <= duration:
        num += 1
        second += increase_width
        x1 = frame
        cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
        success, frame = cap.read()
        x2 = frame

        x1 = cv2.cvtColor(x1, cv2.COLOR_BGR2GRAY)
        try:

            x2 = cv2.cvtColor(x2, cv2.COLOR_BGR2GRAY)
        except:
            continue

        diff = cv2.subtract(x1, x2)
        result = not np.any(diff)
        s = ssim(x1, x2)

        if s < 0.9:
            # 바뀐경우 flame 을 바꿔야 함
            pil_Image = Image.fromarray(frame)
            plt.imshow(frame)
            plt.show()
            image_cropped , image_deleted  = extract_Figures(model,pil_Image)
            image_deleted_list.append(image_deleted)
            image_cropped_list.append(image_cropped)

            cnt += 1

    return image_deleted_list , image_cropped_list
    
    
if __name__ == "__main__":
# 예시다 승규야!! 잘 보라!! 
    model = mrcnn.model.MaskRCNN(mode="inference",
                                    config=SimpleConfig(),
                                    model_dir=os.getcwd())
    model.load_weights(filepath= "./capstone_200_ppt.h5",
                        by_name=True)
    video_dir = '/Users/jeong-wonlyeol/Desktop/캡스톤/youtube/youtube_26.mp4'
    extract_from_video(model,video_dir)
