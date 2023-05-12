import cv2

from ultralytics import YOLO
import supervision as sv
import numpy as np




def main():

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5
    )


    # model = YOLO('D:/docs/deltafour/Tata Power Solar/Data/AjaxOperatorWithoutPPE/bestANP.pt')
    model = YOLO('./models/best_40epochs_8kdata.pt')
    # for result in model.track(source=0, show=True, stream=True, agnostic_nms=True):
    for result in model.track(source='./sample.mp4', stream=True):
        
        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)
        # returns
        # xyxy=yolov8_results.boxes.xyxy.cpu().numpy(),
        # confidence=yolov8_results.boxes.conf.cpu().numpy(),
        # class_id=yolov8_results.boxes.cls.cpu().numpy().astype(int),

        #tracking ids are stored in boxes.id for object detection models.
        #the id tensor is stored on the CPU(RAM),which is then converted to a NumPy array of integers.

        if result.boxes.id is not None:    #to prevent crashing when there are no detections since then it wont get any tracker id and will return None
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)



        # detections = detections[(detections.class_id != 1) & (detections.class_id != 0) 
        #                         & (detections.class_id != 6) & (detections.class_id != 7)
        #                      & (detections.class_id != 8) & (detections.class_id != 9)]
        # print(f'detections : {detections}')
        
        
        labels = [
            f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
            for _,_,confidence, class_id, tracker_id
            in detections
        ]
        # print(f'labels : {labels}')
        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections,
            labels=labels
        )
        print(detections)
        # {0: 'HH', 1: 'M', 2: 'NHH', 3: 'NM', 4: 'NSV', 5: 'P', 6: 'SB', 7: 'SV'}
#         Detections(xyxy=array([[      141.7,      231.93,      180.44,      347.99]], dtype=float32), mask=None, confidence=array([ 
#    0.10196], dtype=float32), class_id=array([5]), tracker_id=None)

        # Detections(xyxy=array([[     29.487,      262.04,      72.823,      342.44],
        #        [       39.1,      231.48,      71.638,      251.95],
        #        [     141.11,      214.35,       175.7,      337.15],
        #        [     145.68,      235.42,      169.57,      274.33],
        #        [      28.52,      229.95,      84.859,      427.29],
        #        [     152.07,      215.64,      165.71,      225.87],
        #        [     56.388,      252.88,      66.596,      266.25]], dtype=float32), class_id=array([7, 0, 5, 4, 5, 0, 3]), confidence=array([    0.92667,     0.88788,     0.88945,     0.86797,    
        #  0.88087,     0.65338,     0.83958], dtype=float32), tracker_id=array([1, 2, 3, 4, 5, 6, 7]))  
        # video 1/1 (40/610)

        cv2.imshow("yolov8", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): # press 'q' to quit
            break

        if (cv2.waitKey(30) == 27):
            break


if __name__ == "__main__":
    main()