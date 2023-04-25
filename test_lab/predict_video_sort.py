import argparse
import cv2
import math
# import deep_sort.nn_matching as nn
# from deep_sort.tracker import Tracker
from ultralytics import YOLO
import numpy as np
from sort import * #TODO
# from deep_sort import *

class_names = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']


pred_no_mask = [] 
pred_no_vest= [] 
pred_no_hardhat = [] 
def process_video(video_path: str, model: YOLO,title = 'default'):
    tracker = Sort(max_age=2000,min_hits= 3,iou_threshold=0.01)
    # metric = nn.NearestNeighborDistanceMetric("cosine", 0.4,None)
    # tracker = Tracker(metric=metric)
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # create a VideoWriter object to write the processed frames to a video file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'./output/{title}_output.mp4', fourcc, fps, (width, height))

    

    while True:
        # read a frame from the video
        ret, img = cap.read()
        if not ret:
            break

        # process the frame with YOLO
        results = model(img, stream=True)

        detections = np.empty((0,5))
        classes = []
        for r in results:
            boxes = r.boxes
            i = 0
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w,h = x2-x1 , y2-y1


                # getting confidence level
                conf = math.ceil((box.conf[0] * 100))
                # class names
                cls = class_names[int(box.cls[0])]

                text = f"{cls}: {conf}%"
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)
                text_offset_x = x1 + (x2 - x1) // 2 - text_width // 2
                text_offset_y = y1 - text_height

                if cls in ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest'] and conf > 30:
                    # cv2.rectangle(img,(x1,y1),(x2,y2),(255,000,255),1)
                    # cv2.putText(img, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(000,000,255), thickness=2)
                    currentArray = np.array([x1,y1,x2,y2,conf])  #check conf[0]
                    detections = np.vstack((detections,currentArray))
                    classes.append(cls)

        resultsTracker = tracker.update(detections)
        i = 0
        for result in resultsTracker:
                x1, y1, x2, y2, id = result
                x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
                # (text_width, text_height), _ = cv2.getTextSize(f'{id } {classes[i]}', cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)
                # text_offset_x = x1 + (x2 - x1) // 2 - text_width // 2
                # text_offset_y = y1 - text_height

                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, thickness=1)
                text_offset_x = x1
                text_offset_y = y1 - text_height
                # if classes[i]  == 'No-Mask' and id not in pred_no_mask: pred_no_mask.append(id)
                # elif classes[i]  == 'NO-Safety Vest' and id not in pred_no_vest: pred_no_vest.append(id)
                # elif classes[i]  == 'NO-hardhat' and id not in pred_no_hardhat: pred_no_hardhat.append(id)
                # cv2.rectangle(img,(x1,y1),(x2,y2),(000,000,255),1)
                cv2.rectangle(img,(x1,y1),(x2,y2),(128, 0, 128),1)
                cv2.rectangle(img,(x1,y1),(x1 + text_width+1, y1 - text_height),(128, 0, 128),-1)
                # cv2.putText(img, f'{id} {classes[i]}', (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(000,000,255), thickness=2)
                # cv2.putText(img, f'{classes[i]}', (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(000,000,255), thickness=2)
                cv2.putText(img, f'{classes[i]}', (text_offset_x, text_offset_y+6), cv2.FONT_HERSHEY_PLAIN, fontScale=0.5, color=(255, 255, 255), thickness=1)
                i+=1

        out.write(img)
        cv2.imshow('video',img)
        print(f'No Masks {len(pred_no_mask)} \n No Safety Vest {len(pred_no_vest)} \n No HardHat {len(pred_no_hardhat)}')
        if cv2.waitKey(1) & 0xFF == ord('q'): # press 'q' to quit
            break
        # elif cv2.waitKey(0):
        #     pass
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-video', type=str, default='./construction_site_example_2.mp4', help='video path')
    parser.add_argument('-model', type=str, default="./models/best_10Class_35Epochs.pt", help='model path')
    args = parser.parse_args()
    return args

def main():
    import logging
    logging.getLogger('yolo').setLevel(logging.WARNING)
    args = parse_args()
    title = args.video.split('/')[-1].split('.')[0]
    model = YOLO(args.model)
    process_video(args.video,model,title)

if __name__ == '__main__':
    main()
