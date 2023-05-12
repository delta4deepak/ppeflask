import argparse
import cv2
import math
from ultralytics import YOLO

class_names = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

def process_image(image_path: str, model: YOLO):
    img = cv2.imread(image_path)
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
                cv2.putText(img, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(000,000,255), thickness=1)
            else:
                cv2.rectangle(img,(x1,y1),(x2,y2),(000,255,000),1)
                cv2.putText(img, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(000,255,000), thickness=1)

    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-img', type=str, default='./sample.jpg', help='image path')
    parser.add_argument('-model', type=str, default="./models/best_10Class_20Epochs.pt", help='model path')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    model = YOLO(args.model)
    process_image(args.img,model)

if __name__ == '__main__':
    main()
