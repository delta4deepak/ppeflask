import argparse
import cv2
import math
from ultralytics import YOLO
# from sort import * #TODO

class_names = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

def process_camera(device: int, model: YOLO):
    cap = cv2.VideoCapture(device)

    while True:
        # read a frame from the video
        ret, img = cap.read()
        if not ret:
            break

        # process the frame with YOLO
        results = model(img, stream=True)

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
                    cv2.rectangle(img,(x1,y1),(x2,y2),(000,000,255),1)
                    cv2.putText(img, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(000,000,255), thickness=2)
                else:
                    cv2.rectangle(img,(x1,y1),(x2,y2),(255,000,000),1)
                    cv2.putText(img, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,000,000), thickness=2)

        cv2.imshow('video',img)
        if cv2.waitKey(1) & 0xFF == ord('q'): # press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', type=int, default=0, help='camera id')
    parser.add_argument('-model', type=str, default="./models/best_10Class_20Epochs.pt", help='model path')
    
    args = parser.parse_args()
    return args

def main():
    import logging
    logging.getLogger('yolo').setLevel(logging.WARNING)
    args = parse_args()
    model = YOLO(args.model)
    process_camera(args.device,model)

if __name__ == '__main__':
    main()
