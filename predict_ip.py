from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
import threading
import torch 
import argparse

CAMERAS = []

app = Flask(__name__)

class_names = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

def find_camera(list_id):
    return CAMERAS[int(list_id)]

def generate_frames(model, cap):
    while True:
        # read frames from cameras
        # frames = []
        # for cap in caps:
        success, frame = cap.read()
            # if success:
            #     frames.append(img)
        
        # predict objects in frames from cameras
        # results = [model(frames[i], stream=True) for i, model in enumerate(models)]
        result = model(frame, stream=True)
        for i, result in enumerate(result):
            for r in result:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # getting confidence level
                    conf = torch.round(box.conf[0] * 100)
                    # class names
                    cls = class_names[int(box.cls[0])]

                    text = f"{cls}: {conf}%"
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)
                    text_offset_x = x1 + (x2 - x1) // 2 - text_width // 2
                    text_offset_y = y1 - text_height

                    if cls in ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest'] and conf > 30:
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),1)
                        cv2.putText(frame, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,255), thickness=2)
                    else:
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),1)
                        cv2.putText(frame, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,0,0), thickness=2)


        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/', methods=["GET"])
def index():
    args = parse_args()
    global CAMERAS
    print('In index')
    # urls = ["http://statenisland.dnsalias.net/mjpg/video.mjpg","http://pendelcam.kip.uni-heidelberg.de/mjpg/video.mjpg","http://129.125.136.20/axis-cgi/mjpg/video.cgi?camera=1","http://amuse.miemasu.net/nphMotionJpeg?Resolution=640x480&Quality=Clarity","http://sunds.tobit.net/cgi-bin/faststream.jpg?stream=full&fps=0"]
    urls = [url for url in args.urls]
    CAMERAS = urls
    return render_template('index_ip.html', camera_list=len(CAMERAS), camera=CAMERAS) #send list to html

import time
@app.route('/video_feed/<string:list_id>/', methods=["GET"]) #receuves index from html
def video_feed(list_id):
    print('IN video feed')
 
    id = int(list_id)
    caps = [cv2.VideoCapture(url) for url in CAMERAS]
    print(CAMERAS)
    models = [YOLO("./models/best_10Class_20Epochs.pt") for i in range(len(CAMERAS))]
    # time.sleep(5) 
    return Response(generate_frames(models[id], caps[id]), mimetype='multipart/x-mixed-replace; boundary=frame')

def parse_args():
    print('parsing args')
    parser = argparse.ArgumentParser(description='Multi-Camera Object Detection')
    parser.add_argument('-urls', nargs='+', help='List of URLs for the cameras', required=True)
    args = parser.parse_args()
    print(args.urls)
    return args

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)

    print('Main method')
