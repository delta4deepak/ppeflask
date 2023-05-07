# sending using model.track from outside

import argparse
import cv2
import math
from flask import Flask, render_template, request, Response
from ultralytics import YOLO
import supervision as sv
import ultralytics
import numpy as np
from nomask import *
no_mask = []

app = Flask(__name__)
box_annotator = sv.BoxAnnotator(
    thickness=1,
    text_thickness=1,
    text_scale=0.5
)
def detect_objects(result,model,frame):

    global no_mask
    detections = sv.Detections.from_yolov8(result)      

    if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

    detections = detections[(detections.class_id != 1) & (detections.class_id != 0) 
                                    & (detections.class_id != 6) & (detections.class_id != 7)
                                & (detections.class_id != 8) & (detections.class_id != 9)
                               & (detections.class_id != 4)  
                            ]
    no_mask = no_mask_person(no_mask,result,detections)
    detections = detections[(detections.class_id != 3)]  #to only show person related boxes
    
    labels = []
    for _, confidence, class_id, tracker_id in detections:
                if tracker_id in no_mask:
                    labels.append(f" {model.model.names[class_id]} {tracker_id} not wearing mask")
                else:
                    labels.append(f" {model.model.names[class_id]} {tracker_id} ")

    frame = box_annotator.annotate(
            scene=frame, 
            detections=detections,
            labels=labels
        )
    return frame

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames(model, video_path, title, save):

    for result in model.track(source = video_path, stream=True):
        frame = result.orig_img
        frame = detect_objects(result,model,frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    args = parse_args()
    model = YOLO(args.model)
    title = args.video.split('/')[-1].split('.')[0]
    video_path = args.video
    return Response(gen_frames(model, video_path,title,args.save), mimetype='multipart/x-mixed-replace; boundary=frame')

def parse_args():
    parser = argparse.ArgumentParser()
    print('parsing args ................')
    parser.add_argument('-video', type=str, default='./sample.mp4', help='video path')
    parser.add_argument('-model', type=str, default="../models/best_10Class_100Epochs.pt", help='path to YOLO model')
    parser.add_argument('-save',type=int, default=0 , help='bool to save or not')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=3000, debug=True)

