from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
import threading
import torch 
import argparse
import supervision as sv
from nomask import *

CAMERAS = []

app = Flask(__name__)


box_annotator = sv.BoxAnnotator(
    thickness=1,
    text_thickness=1,
    text_scale=0.5
)

no_mask = []
def find_camera(list_id):
    return CAMERAS[int(list_id)]

def generate_frames(model, src,id):
    global no_mask
    for result in model.track(source = src, stream=True):
        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)      

        if result.boxes.id is not None:
                detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

        detections = detections[(detections.class_id != 1) & (detections.class_id != 0) 
                                        & (detections.class_id != 6) & (detections.class_id != 7)
                                    & (detections.class_id != 8) & (detections.class_id != 9)
                                & (detections.class_id != 4)  
                                ]
        no_mask[id] = no_mask_person(no_mask[id],result,detections)
        detections = detections[(detections.class_id != 3)]  #to only show person related boxes
       
        labels = []
        for _, confidence, class_id, tracker_id in detections:
                    if tracker_id in no_mask[id]:
                        # labels.append(f" {model.model.names[class_id]} {tracker_id} not wearing mask")
                        labels.append(f"Not wearing mask")
                    # else:
                    #     labels.append(f" {model.model.names[class_id]} {tracker_id} ")
        detections = detections[(detections.class_id != 5)]  #to only show person related boxes

        frame = box_annotator.annotate(
                scene=frame, 
                detections=detections,
                labels=labels
            )

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/', methods=["GET"])
def index():
    global no_mask
    args = parse_args()
    global CAMERAS
    print('In index')
    # urls = ["http://statenisland.dnsalias.net/mjpg/video.mjpg",
    # "http://pendelcam.kip.uni-heidelberg.de/mjpg/video.mjpg",
    # "http://129.125.136.20/axis-cgi/mjpg/video.cgi?camera=1",
    # "http://cam-stadthaus.dacor.de/cgi-bin/faststream.jpg?stream=full&fps=0"
    # "http://sunds.tobit.net/cgi-bin/faststream.jpg?stream=full&fps=0"]
    # urls = [url for url in args.urls]
    urls = ['./samples/sample.mp4','./samples/construction_site_example.mp4','samples/construction_site_example_2.mp4']
    CAMERAS = urls
    no_mask = [ [] for _ in range(len(CAMERAS)) ]
    return render_template('index_ip.html', camera_list=len(CAMERAS), camera=CAMERAS) #send list to html

import time
i = 0
@app.route('/video_feed/<string:list_id>/', methods=["GET"]) #receives index from html
def video_feed(list_id):
    print('IN video feed')
    global i
    id = int(list_id)
    # caps = [cv2.VideoCapture(url) for url in CAMERAS]
    print(CAMERAS)
    
    models = [YOLO("./models/best_10Class_100Epochs.pt") for _ in range(len(CAMERAS))]
    print(f'$$$$$$$$$$$$$$$$$$$$$$$$$$      {i}       $$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    i+=1
    # time.sleep(5) 
    return Response(generate_frames(models[id], CAMERAS[id],id), mimetype='multipart/x-mixed-replace; boundary=frame')

def parse_args():
    print('parsing args')
    parser = argparse.ArgumentParser(description='Multi-Camera Object Detection')
    # parser.add_argument('-urls', nargs='+', help='List of URLs for the cameras', required=True)
    parser.add_argument('-urls', nargs='+', help='List of URLs for the cameras') 
    args = parser.parse_args()
    print(args.urls)
    return args

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)

    print('Main method')
