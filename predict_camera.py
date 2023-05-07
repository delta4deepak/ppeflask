import argparse
import math

import cv2
from flask import Flask, Response, render_template, request
from ultralytics import YOLO


class_names = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']
app = Flask(__name__)

def detect_objects(model,frame):
    results = model(frame, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # getting confidence level
            conf = math.ceil((box.conf[0] * 100))
            # class names
            cls = class_names[int(box.cls[0])]

            text = f"{cls}: {conf}%"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)
            text_offset_x = x1 + (x2 - x1) // 2 - text_width // 2
            text_offset_y = y1 - text_height

            if cls in ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest'] and conf > 40:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(000,000,255),1)
                cv2.putText(frame, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(000,000,255), thickness=2)
            else:
                pass
    return frame


def gen_frames(model,device):
    cap = cv2.VideoCapture(device)
    while True:
        # read a frame from the video
        ret, img = cap.read()
        if not ret:
            break
            
        frame = detect_objects(model,img)

        ret, buffer = cv2.imencode('.jpg',frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    args = parse_args()
    model = YOLO(args.model)
    device = args.device
    return Response(gen_frames(model, device), mimetype='multipart/x-mixed-replace; boundary=frame')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', type=int, default=0, help='camera id')
    parser.add_argument('-model', type=str, default="./models/best_10Class_100Epochs.pt", help='model path')  
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=3000, debug=True)