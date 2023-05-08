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
from novest import *
from nohat import *

notified = []

no_mask = []
no_vest = []
no_hat = []

app = Flask(__name__)
box_annotator = sv.BoxAnnotator(
    thickness=1,
    text_thickness=1,
    text_scale=0.3,
    text_padding=3
)
def detect_objects(result,model,frame):

    global no_mask,no_vest,no_hat,notified
    detections = sv.Detections.from_yolov8(result)      

    if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

    detections = detections[(detections.class_id != 1) & (detections.class_id != 0) 
                                    & (detections.class_id != 6) & (detections.class_id != 7)
                                & (detections.class_id != 8) & (detections.class_id != 9)    
                            ]
    no_mask = no_mask_person(no_mask,result,detections)
    no_vest = no_vest_person(no_vest,result,detections)
    no_hat = no_hat_person(no_hat,result,detections)
    
    detections = detections[(detections.class_id != 3) & (detections.class_id != 4) & (detections.class_id != 2)]  #to only show person related boxes
    
    labels = []
    

    for _, confidence, class_id, tracker_id in detections:
                if tracker_id in no_mask and tracker_id in no_vest and tracker_id in no_hat:
                    message = f" {model.model.names[class_id]} {tracker_id} not wearing mask, safety vest and hardhat"
                    labels.append(message)
                    if tracker_id not in notified: notified.append(tracker_id)
                         
                elif tracker_id in no_mask and tracker_id not in no_vest and tracker_id not in no_hat:
                    labels.append(f" {model.model.names[class_id]} {tracker_id} not wearing mask")
                    if tracker_id not in notified: notified.append(tracker_id)
                elif tracker_id not in no_mask and tracker_id in no_vest and tracker_id not in no_hat:
                    labels.append(f" {model.model.names[class_id]} {tracker_id} not wearing safety vest")
                    if tracker_id not in notified: notified.append(tracker_id)
                elif tracker_id not in no_mask and tracker_id not in no_vest and tracker_id in no_hat:
                    labels.append(f" {model.model.names[class_id]} {tracker_id} not wearing hardhat")
                    if tracker_id not in notified: notified.append(tracker_id)
                elif tracker_id in no_mask and tracker_id in no_vest and tracker_id not in no_hat:
                    labels.append(f" {model.model.names[class_id]} {tracker_id} not wearing mask and safety vest")
                    if tracker_id not in notified: notified.append(tracker_id)
                elif tracker_id in no_mask and tracker_id not in no_vest and tracker_id in no_hat:
                    labels.append(f" {model.model.names[class_id]} {tracker_id} not wearing mask and hardhat")
                    if tracker_id not in notified: notified.append(tracker_id)
                elif tracker_id not in no_mask and tracker_id in no_vest and tracker_id in no_hat:
                    labels.append(f" {model.model.names[class_id]} {tracker_id} not wearing hardhat and safetyvest")
                    if tracker_id not in notified: notified.append(tracker_id)
                else:
                    labels.append(f" {model.model.names[class_id]} {tracker_id} ")

    frame = box_annotator.annotate(
            scene=frame, 
            detections=detections,
            labels=labels
        )
    box_x = 50
    box_y = frame.shape[0] - 100
    box_width = 200
    box_height = 50
    text = f'{len(notified)} non-compliances found'
    # cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 255, 0), -1)
    cv2.putText(frame, text, (box_x + 10, box_y + 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2)

    return frame

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames(model, video_path):

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
    video_path = args.video
    return Response(gen_frames(model, video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

def parse_args():
    parser = argparse.ArgumentParser()
    print('parsing args ................')
    parser.add_argument('-video', type=str, default='https://www.youtube.com/watch?v=q-OVSJbhpNY', help='video path')
    parser.add_argument('-model', type=str, default="../models/best_10Class_100Epochs.pt", help='path to YOLO model')
    # parser.add_argument('-save',type=int, default=0 , help='bool to save or not')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=3000, debug=True)

