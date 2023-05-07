import argparse
import cv2
import math
from ultralytics import YOLO
import numpy as np
from sort import * 


class_names = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

def process_video(video_path: str, model: YOLO):
    tracker = Sort(max_age=2000,min_hits= 3,iou_threshold=0.01)
    cap = cv2.VideoCapture(video_path)
    
    while True:
        # read a frame from the video
        ret, img = cap.read()
        if not ret:
            break

        # process the frame with YOLO
        results = model(img, stream=True)

        detections = np.empty((0,5))
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
                if cls in ['Person','NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']:
                    currentArray = np.array([x1,y1,x2,y2,conf])  #check conf[0]
                    detections = np.vstack((detections,currentArray))
        resultsTracker = tracker.update(detections)

        area = {}
        for result in resultsTracker:
                x1, y1, x2, y2, id = result
                x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)

                area[id] = [x1, y1, x2, y2]

                (text_width, text_height), _ = cv2.getTextSize(f'{id}', cv2.FONT_HERSHEY_PLAIN, fontScale=0.5, thickness=1)
                text_offset_x = x1
                text_offset_y = y1 - text_height
                cv2.rectangle(img,(x1,y1),(x2,y2),(000,000,255),1)
                cv2.putText(img, f'{id} ', (text_offset_x, text_offset_y+6), cv2.FONT_HERSHEY_PLAIN, fontScale=0.5, color=(255, 255, 255), thickness=1)
  
        print(area)
        cv2.imshow('video',img)
        if cv2.waitKey(1) & 0xFF == ord('q'): # press 'q' to quit
            break
        elif cv2.waitKey(0):
            pass
    cap.release()
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-video', type=str, default='./samples/construction_site_example_2.mp4', help='video path')
    parser.add_argument('-model', type=str, default="./models/best_10Class_100Epochs.pt", help='model path')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    title = args.video.split('/')[-1].split('.')[0]
    model = YOLO(args.model)
    process_video(args.video,model)

if __name__ == '__main__':
    main()
