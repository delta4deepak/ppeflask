from ultralytics import YOLO
import cv2
import argparse
import threading
import torch 
class_names = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']



def process_ip(urls,models):
    # initialize capture objects for all cameras
    caps = [cv2.VideoCapture(url) for url in urls] #if no need of urls the send caps as arguments

    # set camera properties (if needed)
    for cap in caps:
        cap.set(3, 640)
        cap.set(4, 480)

    class CameraThread(threading.Thread):
        def __init__(self, model, cap):
            threading.Thread.__init__(self)
            self.model = model
            self.cap = cap

        def run(self):
            while True:
                # read frames from camera
                success, img = self.cap.read()

                # predict objects in frames from camera
                if success:
                    results = self.model(img, stream=True)
                    for r in results:
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
                                cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),1)
                                cv2.putText(img, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,255), thickness=2)
                            else:
                                cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),1)
                                cv2.putText(img, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,0,0), thickness=2)

                    # display frame (if needed)
                    img = cv2.resize(img, (640, 480))
                    cv2.imshow(self.getName(), img)

                    # exit on key press (if needed)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            # release capture object and close window
            self.cap.release()
            cv2.destroyAllWindows()

    # create camera threads
    threads = []
    for i in range(len(models)):
        threads.append(CameraThread(models[i], caps[i]))

    # start camera threads
    for thread in threads:
        thread.start()

    # wait for camera threads to finish
    for thread in threads:
        thread.join()
    # close all OpenCV windows
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-Camera Object Detection')
    parser.add_argument('--urls', nargs='+', help='List of URLs for the cameras', required=True)
    args = parser.parse_args()
    return args

def main():
    import logging
    logging.getLogger('yolo').setLevel(logging.WARNING)
    args = parse_args()
    models = [YOLO("./models/best_10Class_20Epochs.pt") for i in range(len(args.urls))]
    process_ip(args.urls,models)

if __name__ == '__main__':
    main()

# # initialize YOLO models for both cameras
# model1 = YOLO("./models/best_10Class_20Epochs.pt")
# model2 = YOLO("./models/best_10Class_20Epochs.pt")
# model3 = YOLO("./models/best_10Class_20Epochs.pt")

# # initialize capture objects for both cameras
# cap1 = cv2.VideoCapture("http://statenisland.dnsalias.net/mjpg/video.mjpg")
# cap2 = cv2.VideoCapture("http://88.117.170.10/axis-cgi/mjpg/video.cgi")
# cap3 = cv2.VideoCapture("http://cam-stadthaus.dacor.de/cgi-bin/faststream.jpg?stream=full&fps=0")
                    

# # set camera properties (if needed)
# cap1.set(3, 640)
# cap1.set(4, 480)

# cap2.set(3, 640)
# cap2.set(4, 480)

# cap3.set(3, 640)
# cap3.set(4, 480)

# class CameraThread(threading.Thread):
#     def __init__(self, model, cap):
#         threading.Thread.__init__(self)
#         self.model = model
#         self.cap = cap

#     def run(self):
#         while True:
#             # read frames from camera
#             success, img = self.cap.read()

#             # predict objects in frames from camera
#             if success:
#                 results = self.model(img, stream=True)
#                 for r in results:
#                     boxes = r.boxes
#                     for box in boxes:
#                         x1, y1, x2, y2 = box.xyxy[0]
#                         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

#                         # getting confidence level
#                         conf = torch.round(box.conf[0] * 100)
#                         # class names
#                         cls = class_names[int(box.cls[0])]

#                         text = f"{cls}: {conf}%"
#                         (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)
#                         text_offset_x = x1 + (x2 - x1) // 2 - text_width // 2
#                         text_offset_y = y1 - text_height

#                         if cls in ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest'] and conf > 30:
#                             cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),1)
#                             cv2.putText(img, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,255), thickness=2)
#                         else:
#                             cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),1)
#                             cv2.putText(img, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,0,0), thickness=2)

#                 # display frame (if needed)
#                 img = cv2.resize(img, (640, 480))
#                 cv2.imshow(self.getName(), img)

#                 # exit on key press (if needed)
#                 if cv2.waitKey(1) & 0xFF == ord("q"):
#                     break

#         # release capture object and close window
#         self.cap.release()
#         cv2.destroyAllWindows()

# # create camera threads
# thread1 = CameraThread(model1, cap1)
# thread2 = CameraThread(model2, cap2)
# thread3 = CameraThread(model3, cap3)

# # start camera threads
# thread1.start()
# thread2.start()
# thread3.start()

# # wait for camera threads to finish
# thread1.join()
# thread2.join()
# thread3.join()


# # # delay in video
# import argparse
# from concurrent.futures import ThreadPoolExecutor
# from functools import partial
# import cv2
# from ultralytics import YOLO

# parser = argparse.ArgumentParser()
# parser.add_argument('-urls', nargs='+', help='stream URLs')
# args = parser.parse_args()

# # initialize YOLO model
# model = YOLO("./models/best_10Class_20Epochs.pt")

# # initialize capture objects and windows for each stream
# caps = []
# windows = []
# for i, stream_url in enumerate(args.urls):
#     cap = cv2.VideoCapture(stream_url)
#     cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#     caps.append(cap)
#     window_name = f"Stream {i}"
#     cv2.namedWindow(window_name)
#     windows.append(window_name)

# # set camera properties (if needed)
# for cap in caps:
#     cap.set(3, 640)
#     cap.set(4, 480)

# # read and process frames from all streams
# while True:
#     for i, cap in enumerate(caps):
#         success, img = cap.read()
#         if not success:
#             break
#         results = model(img, stream=True)

#         for result in results:
#             if result is not None:
#                 for box in result.boxes:
#                     x1, y1, x2, y2 = box.xyxy[0]
#                     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

#                     # getting confidence level
#                     conf = torch.round(box.conf[0] * 100)
#                     # class names
#                     cls = class_names[int(box.cls[0])]

#                     text = f"{cls}: {conf}%"
#                     (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)
#                     text_offset_x = x1 + (x2 - x1) // 2 - text_width // 2
#                     text_offset_y = y1 - text_height

#                     if cls in ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest'] and conf > 30:
#                         cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),1)
#                         cv2.putText(img, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,255), thickness=2)
#                     else:
#                         cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),1)
#                         cv2.putText(img, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,0,0), thickness=2)

#         # display frames
#         window_name = windows[i]
#         img = cv2.resize(img, (640, 480))
#         cv2.imshow(window_name, img)

#     # exit on key press
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # release capture objects and close windows
# for cap in caps:
#     cap.release()
# cv2.destroyAllWindows()

# error
# from ultralytics import YOLO
# import cv2
# import threading
# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('-urls', nargs='+', help='stream URLs')
# args = parser.parse_args()

# # initialize YOLO models
# model = YOLO("./models/best_10Class_20Epochs.pt")
# caps = []
# for i, stream_url in enumerate(args.urls):
#     cap = cv2.VideoCapture(stream_url)
#     caps.append(cap)
# # set camera properties (if needed)
# for cap in caps:
#     cap.set(3, 640)
#     cap.set(4, 480)


# class CameraThread(threading.Thread):
#     def __init__(self, model, cap):
#         threading.Thread.__init__(self)
#         self.model = model
#         self.cap = cap

#     def run(self):
#         while True:
#             # read frames from camera
#             success, img = self.cap.read()

#             # predict objects in frames from camera
#             if success:
#                 results = self.model(img, stream=True)
#                 for r in results:
#                     boxes = r.boxes
#                     for box in boxes:
#                         x1, y1, x2, y2 = box.xyxy[0]
#                         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

#                         # getting confidence level
#                         conf = torch.round(box.conf[0] * 100)
#                         # class names
#                         cls = class_names[int(box.cls[0])]

#                         text = f"{cls}: {conf}%"
#                         (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)
#                         text_offset_x = x1 + (x2 - x1) // 2 - text_width // 2
#                         text_offset_y = y1 - text_height

#                         if cls in ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest'] and conf > 30:
#                             cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),1)
#                             cv2.putText(img, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,255), thickness=2)
#                         else:
#                             cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),1)
#                             cv2.putText(img, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,0,0), thickness=2)

#                 # display frame (if needed)
#                 img = cv2.resize(img, (640, 480))
#                 cv2.imshow(self.getName(), img)

#                 # exit on key press (if needed)
#                 if cv2.waitKey(1) & 0xFF == ord("q"):
#                     break

#         # release capture object and close window
#         self.cap.release()
#         cv2.destroyAllWindows()

# threads = []
# # create camera threads
# for cap in caps:
#     t = CameraThread(model, cap)
#     t.start()
#     threads.append(t)
    
# for t in threads:
#     t.join()




# # multithreading method
# def process_ip(model, cap, cam_id):
#     while True:
#         # read a frame from the video
#         ret, img = cap.read()
#         if not ret:
#             break

#         # process the frame with YOLO
#         results = model(img, stream=True)

#         for r in results:
#             boxes = r.boxes
#             for box in boxes:
#                 x1, y1, x2, y2 = box.xyxy[0]
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

#                 # getting confidence level
#                 conf = torch.round(box.conf[0] * 100)
#                 # class names
#                 cls = class_names[int(box.cls[0])]

#                 text = f"{cls}: {conf}%"
#                 (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)
#                 text_offset_x = x1 + (x2 - x1) // 2 - text_width // 2
#                 text_offset_y = y1 - text_height

#                 if cls in ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest'] and conf > 30:
#                     cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),1)
#                     cv2.putText(img, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,255), thickness=2)
#                 else:
#                     cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),1)
#                     cv2.putText(img, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,0,0), thickness=2)

#         cv2.imshow(f"Camera {cam_id}", img)
#         if cv2.waitKey(1) & 0xFF == ord('q'): # press 'q' to quit
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-urls', nargs='+', help='list of IP camera URLs')
#     parser.add_argument('-model', type=str, default="./models/best_10Class_20Epochs.pt", help='model path')

#     args = parser.parse_args()
#     return args

# def main():
#     args = parse_args()
#     model = YOLO(args.model)

#     # create capture objects for each camera
#     caps = []
#     for url in args.urls:
#         cap = cv2.VideoCapture(url)
#         cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # set buffer size to 1
#         caps.append(cap)

#     # start threads for each camera
#     threads = []
#     for i, cap in enumerate(caps):
#         t = threading.Thread(target=process_ip, args=(model, cap, i+1))
#         t.start()
#         threads.append(t)

#     # wait for threads to finish
#     for t in threads:
#         t.join()

# if __name__ == '__main__':
#     main()





# simple way
# from ultralytics import YOLO
# import cv2

# # initialize YOLO models for both cameras
# model1 = YOLO("./models/best_10Class_20Epochs.pt")
# model2 = YOLO("./models/best_10Class_20Epochs.pt")

# # initialize capture objects for both cameras
# cap1 = cv2.VideoCapture("http://statenisland.dnsalias.net/mjpg/video.mjpg")
# cap2 = cv2.VideoCapture("http://88.117.170.10/axis-cgi/mjpg/video.cgi")

# # set camera properties (if needed)
# cap1.set(3, 640)
# cap1.set(4, 480)
# cap2.set(3, 640)
# cap2.set(4, 480)

# # read and process frames from both cameras
# while True:
#     # read frames from both cameras
#     success1, img1 = cap1.read()
#     success2, img2 = cap2.read()

#     # predict objects in frames from both cameras
#     if success1:
#         results1 = model1(img1, stream=True)
#         for r in results1:
#             boxes = r.boxes
#             for box in boxes:
#                 x1, y1, x2, y2 = box.xyxy[0]
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

#                 # getting confidence level
#                 conf = torch.round(box.conf[0] * 100)
#                 # class names
#                 cls = class_names[int(box.cls[0])]

#                 text = f"{cls}: {conf}%"
#                 (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)
#                 text_offset_x = x1 + (x2 - x1) // 2 - text_width // 2
#                 text_offset_y = y1 - text_height

#                 if cls in ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest'] and conf > 30:
#                     cv2.rectangle(img1,(x1,y1),(x2,y2),(0,0,255),1)
#                     cv2.putText(img1, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,255), thickness=2)
#                 else:
#                     cv2.rectangle(img1,(x1,y1),(x2,y2),(255,0,0),1)
#                     cv2.putText(img1, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,0,0), thickness=2)

#         # cv2.imshow(f"Camera 0", img1)


#     if success2:
#         results2 = model2(img2, stream=True)
#         for r in results2:
#             boxes = r.boxes
#             for box in boxes:
#                 x1, y1, x2, y2 = box.xyxy[0]
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

#                 # getting confidence level
#                 conf = torch.round(box.conf[0] * 100)
#                 # class names
#                 cls = class_names[int(box.cls[0])]

#                 text = f"{cls}: {conf}%"
#                 (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)
#                 text_offset_x = x1 + (x2 - x1) // 2 - text_width // 2
#                 text_offset_y = y1 - text_height

#                 if cls in ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest'] and conf > 30:
#                     cv2.rectangle(img2,(x1,y1),(x2,y2),(0,0,255),1)
#                     cv2.putText(img2, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,255), thickness=2)
#                 else:
#                     cv2.rectangle(img2,(x1,y1),(x2,y2),(255,0,0),1)
#                     cv2.putText(img2, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,0,0), thickness=2)

#         # cv2.imshow(f"Camera 1", img2)

#     # display frames (if needed)
#     cv2.imshow("Camera 1", img1)
#     cv2.imshow("Camera 2", img2)

#     # exit on key press (if needed)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # release capture objects and close windows
# cap1.release()
# cap2.release()
# cv2.destroyAllWindows()


