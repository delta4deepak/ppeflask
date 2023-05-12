import argparse
import cv2
import math
# from flask import Flask, render_template, request, Response
from ultralytics import YOLO
# import supervision as sv
# import ultralytics



model = YOLO('./models/best_15epochs_8kdata.pt')
print(model.model.names)
# for result in model.track(source = video_path, stream=True):
#     frame = result.orig_img
#     frame = detect_objects(result,model,frame)

