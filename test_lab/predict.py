import argparse
import sys 
sys.path.extend('./')
from predict_image import process_image
from predict_video import process_video
from predict_camera import process_camera
from predict_ip import process_ip
from ultralytics import YOLO
import threading
import logging
import cv2
logging.getLogger('yolo').setLevel(logging.WARNING)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-img', type=str, default=None, help='image path')
    parser.add_argument('-video', type=str, default=None, help='video path')
    parser.add_argument('-device', type=int, default=None, help='camera id')
    parser.add_argument('-urls', nargs='+',default=None, help='list of IP camera URLs')
    parser.add_argument('-model', type=str, default="./models/best_10Class_20Epochs.pt", help='model path')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # print('Deciphering model using key')
    # cipher = Fernet(b'skqauwzjSe24yga3tfE-CLfz87OLVMk1LN3afy0vCKc=')
    # print('cipher created')
    # # Read the encrypted data from a file
    # with open(args.model, 'rb') as f:
    #     encrypted_data = f.read()
    # print('done loading encrypted model')
    # # Decrypt the data using the same key
    # decrypted_data = cipher.decrypt(encrypted_data)
    # print('deciphering model')
    
    # model = YOLO(decrypted_data)
    model = YOLO(args.model)


    if args.img is not None:
        model = YOLO(args.model)
        process_image(args.img, model)

    elif args.video is not None: 
        model = YOLO(args.model)
        process_video(args.video, model)

    elif args.urls is not None:
        models = [YOLO("./models/best_10Class_20Epochs.pt") for i in range(len(args.urls))]
        process_ip(args.urls,models)

    elif args.device is not None:
        model = YOLO(args.model)
        process_camera(args.device, model)

    else:
        print("No argument was provided.")
if __name__ == '__main__':
    main()
