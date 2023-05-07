import argparse
import cv2
import math
from flask import Flask, render_template, request, Response
from ultralytics import YOLO
from sort import * # TODO

class_names = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

app = Flask(__name__)

def detect_objects(model, frame):
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
                # cv2.rectangle(frame,(x1,y1),(x2,y2),(255,000,000),1)
                # cv2.putText(frame, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,000,000), thickness=2)

    return frame

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames(model, video_path, title, save):
    cap = cv2.VideoCapture(video_path)
    fps = height = width = fourcc = out = None
    if save: 
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # create a VideoWriter object to write the processed frames to a video file
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f'./output/{title}_output.mp4', fourcc, fps, (width, height))
        print(f'Saving the video as ./output/{title}_output.mp4')

    while True:
        # read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # process the frame with YOLO
        frame = detect_objects(model, frame)
        # write the processed frame to the output video file
        if save : out.write(frame)

        # convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # yield the frame in a Flask response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    if save : out.release()

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
    parser.add_argument('-model', type=str, default="./models/best_10Class_100Epochs.pt", help='path to YOLO model')
    parser.add_argument('-save',type=int, default=0 , help='bool to save or not')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=3000, debug=True)

