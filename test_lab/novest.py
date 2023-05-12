def get_person_boxes(detections):
        person_list = detections.class_id
        indices = [i for i in range(len(person_list)) if person_list[i] == 5]
        if not indices:
            return [], []
        return [ detections.xyxy[j] for j in indices ], [ detections.tracker_id[j] for j in indices ]

def get_vest_boxes(detections):
        mask_list = detections.class_id
        indices = [i for i in range(len(mask_list)) if mask_list[i] == 4]
        if not indices:
            return [], []
        return [ detections.xyxy[j] for j in indices ]

def center(box):
     return int((box[0] + box[2]) // 2) , int((box[1] + box[3]) // 2)

def inBoundingBox(x1, y1, x2, y2, x, y):
    isXInRange = x >= x1 and x <= x2
    isYInRange = y >= y1 and y <= y2
    return (isXInRange and isYInRange)

def no_vest_person(no_vest,result,detections):
        if result.boxes.id is not None:    
            if 5 in detections.class_id and 4 in detections.class_id and detections.xyxy.any():
                pb,tids = get_person_boxes(detections)
                mb = get_vest_boxes(detections)
                for pbox,tid in zip(pb,tids):
                    # print(tid)
                    for mbox in mb:
                        if inBoundingBox(pbox[0],pbox[1],pbox[2],pbox[3],center(mbox)[0],center(mbox)[1]):
                            if tid not in no_vest: no_vest.append(tid)
        return no_vest

# 0: 'Hardhat', 1: 'Mask', 2: 'NO-Hardhat', 
# 3: 'NO-Mask', 4: 'NO-Safety Vest', 
# 5: 'Person', 6: 'Safety Cone', 7: 'Safety Vest', 
# 8: 'machinery', 9: 'vehicle'