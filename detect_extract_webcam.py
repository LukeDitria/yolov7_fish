import argparse
import time
from pathlib import Path
from tqdm import trange, tqdm

import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from collections import defaultdict
import pandas as pd
import copy
from numpy import random
import os
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def detect():
    weights, imgsz, root_save_dir, record_mins, count_secs = opt.weights, opt.img_size, opt.root_save_dir, opt.record_mins, opt.count_secs

    # Initialize
    set_logging()
    device = select_device(opt.device)
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model

    # Run the data extractor, it will return a pandas dataframe of the timestamped counts per class
    summary_dict = extract_frames(model, imgsz, device, record_mins, count_secs)
    summary_dict.to_csv("video_summary.csv")


def extract_frames(model, imgsz, device, record_mins, count_secs):
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Setup webcam capture
    cap = cv2.VideoCapture(0)

    # Time when recording started
    start_time = time.time()

    # Start record recording
    record_time = time.time()

    # Setup global record
    total_df = None
    # Record the total max counts and largest area for example frame saving
    max_count = defaultdict(lambda: {"count": 0, "total_area": 0, "sum_conf": 0})

    # Create current frame record
    new_row = pd.DataFrame(dict.fromkeys(names, 0), index=[0])

    # Record for the set period of time
    while (time.time() - start_time) < (60 * record_mins):
        # cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        _, im0 = cap.read()

        # Resize and convert the numpy frame to a torch tensor
        img = letterbox(im0, imgsz, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB and transpose from WxHxC to CxHxW
        img = np.ascontiguousarray(img)
        img = torch.tensor(img).unsqueeze(0).to(device)/255

        # Inference
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            with torch.cuda.amp.autocast():
                pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Data logger for the current frame
        data_logger = defaultdict(lambda: {"count": 0, "total_area": 0, "sum_conf": 0})
        num_objects = 0

        # Make a memory copy of the current frame to draw boxes on
        frame_to_label = copy.deepcopy(im0)

        # Process detections
        for i, det in enumerate(pred):  # detections per image

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # Convert the output of the model (top left xy, bottom right xy) to top left XY and HW
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    # line = (names[int(cls.item())], *xywh)  # label format
                    # print("Height %.4f, Width %.4f, Area %.4f" % (xywh[-1], xywh[-2], xywh[-1]*xywh[-2]))

                    # Calculate the relative area of the current detection
                    area = xywh[-1]*xywh[-2]

                    # Only log the detection as an object if the area is large enough
                    if area >= 0.005:
                        num_objects += 1
                        class_name = names[int(cls)]
                        label = f'{class_name} {conf:.2f}'
                        plot_one_box(xyxy, frame_to_label, label=label, color=colors[int(cls)], line_thickness=3)

                        # Keep track of the count of each class in the frame and the total area each object takes up
                        data_logger[class_name]["count"] += 1
                        data_logger[class_name]["total_area"] += area
                        data_logger[class_name]["sum_conf"] += conf

        # If any objects were logged
        if num_objects > 0:
            # For each class (of the ones that appeared in the frame)
            for key, value in data_logger.items():
                # If the count of this class in this frame is greater than the max count for the current time peroid
                # update the max count
                if value["count"] > new_row[key][0]:
                    new_row[key] = value["count"]

                # Save an example frame if the count of this class in this frame is greater than the max count overall
                # OR if it is the same AND the total area is greater than current total area
                # This prevents us from only logging as soon as the object enters the frame (it might be cut off)
                count_gt = value["count"] > max_count[key]["count"]
                count_eq = value["count"] == max_count[key]["count"]
                area_gt = value["total_area"] > max_count[key]["total_area"]
                conf_gt = value["sum_conf"] > max_count[key]["sum_conf"]

                curr_conf_area = value["sum_conf"] * value["total_area"]
                max_conf_area = max_count[key]["sum_conf"] * max_count[key]["total_area"]
                conf_area_gt = curr_conf_area > max_conf_area
                # print("Class %s, Old Area %.4f, Curr Area %.4f" % (key, max_count[key]["total_area"], value["total_area"]))
                # print("Class %s, Old Conf %.4f, Curr Conf %.4f" % (key, max_count[key]["mean_conf"], (value["sum_conf"]/value["count"])))
                # print("Class %s, Old Conf %.4f, Curr Conf %.4f" % (key, max_conf_area, curr_conf_area))

                if count_gt or (count_eq and conf_area_gt):
                    max_count[key]["count"] = value["count"]
                    max_count[key]["total_area"] = value["total_area"]
                    max_count[key]["sum_conf"] = value["sum_conf"]

                    # Save frame with and without labels
                    cv2.imwrite("class_frames/" + key + " labeled " + str(value["count"]) + ".jpg", frame_to_label)
                    cv2.imwrite("class_frames/" + key + " no label " + str(value["count"]) + ".jpg", im0)

        # If the current logging interval has elapsed then append the record for the current time period to the
        # global record
        if (time.time() - record_time) > count_secs:
            new_row["time"] = pd.Timestamp.now()
            new_row.set_index("time", inplace=True)

            # If total_df does not exist initialise it with the current record
            # If it does exist then simply append the current record to it
            if total_df is None:
                total_df = copy.deepcopy(new_row)
            else:
                total_df = pd.concat([total_df, copy.deepcopy(new_row)])

            # Reset the current recording period
            record_time = time.time()
            new_row = pd.DataFrame(dict.fromkeys(names, 0), index=[pd.Timestamp.now()])

        # Display the labeled frame
        cv2.imshow('Counter Intelligence', frame_to_label)
        cv2.waitKey(1)
        time.sleep(0.1)

    return total_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--root-save-dir', '-svd', type=str, default='.', help='root save dir')

    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--record-mins', type=int, default=5, help='number of minutes to record for')
    parser.add_argument('--count-secs', type=int, default=5, help='number of seconds to count over')

    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
