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
import shutil
from datetime import datetime, timedelta

from numpy import random
import os
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def detect():
    # Initialize
    set_logging()
    device = select_device(opt.device)
    # Load model
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model

    save_dir = os.path.join(opt.root_save_dir, opt.root_source_dir.split("/")[-1])
    valid_file_ext = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']
    print('##################################################')
    print('##################EXTRACTING!#####################')

    # Get the total number of files and directories
    total_files_and_directories = sum([len(files) for root, dirs, files in os.walk(opt.root_source_dir)])

    # Initialize a counter for the number of files and directories processed
    files_and_directories_processed = 0

    # Use os.walk() to loop through the nested directories
    for root, dirs, files in os.walk(opt.root_source_dir):
        if len(files) > 0:
            sub_dir = save_dir + root.replace(opt.root_source_dir, '')
            if not os.path.isdir(sub_dir):
                os.mkdir(sub_dir)

            # Loop through the list of files
            for file in files:
                file_path = os.path.join(root, file)
                file_name = file.split(".")[0]
                file_type = file.split(".")[-1]

                if file_type.lower() in valid_file_ext:
                    csv_filename = file_name + ".csv"
                    summary_filepath = os.path.join(sub_dir,  csv_filename)

                    if not os.path.isfile(summary_filepath) or opt.restart_job:
                        summary_dict = extract_frames(file_path, file_name, model, opt.img_size, opt.sample_fps, device)
                        if summary_dict is not None:
                            summary_dict.to_csv(summary_filepath)
                    else:
                        print("Already processed %s" % file)

                    files_and_directories_processed += 1
                    completion_percentage = (files_and_directories_processed / total_files_and_directories) * 100
                    print(f"Completion percentage: {completion_percentage:.2f}%")

    print('##################################################')
    print('##################COMPLETE!#####################')


def extract_frames(file_path, file_name, model, imgsz, sample_fps, device):
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    cap = cv2.VideoCapture(file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    sample_rate = video_fps//sample_fps

    ti_m = os.path.getmtime(file_path)
    video_start_time = datetime.utcfromtimestamp(ti_m)
    video_dataframe = None

    class_names = model.module.names if hasattr(model, 'module') else model.names

    if video_fps == 0 or not cap.isOpened():
        print("Video File %s is Corrupt" % file_name)
        return None

    if sample_fps > video_fps:
        raise ValueError("Sample FPS (%d) cannot be greater than Video FPS (%d)" % (sample_fps, video_fps))

    for fno in trange(0, total_frames, sample_rate):
        frame_delta = timedelta(seconds=fno / video_fps)
        timestamp = video_start_time + frame_delta

        frame_dict = dict(zip(class_names, np.zeros(len(class_names))))
        frame_dict["timestamp"] = timestamp
        frame_dict["frame_number"] = fno
        frame_dict["file_name"] = file_name
        frame_dict["total_detections"] = 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        _, im0 = cap.read()

        img = letterbox(im0, imgsz, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.tensor(img).unsqueeze(0).to(device)/255

        # Inference
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            with torch.cuda.amp.autocast():
                pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    class_name = class_names[int(cls)]

                    frame_dict[class_name] += 1
                    frame_dict["total_detections"] += 1

        if video_dataframe is None:
            video_dataframe = pd.DataFrame(frame_dict, index=[0])
        else:
            video_dataframe = pd.concat([video_dataframe, pd.DataFrame(frame_dict, index=[0])])

    return video_dataframe


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--root-source-dir', '-srd', type=str, default='.', help='root source dir')
    parser.add_argument('--root-save-dir', '-svd', type=str, default='.', help='root save dir')

    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--sample-fps', '-fps', type=int, default=5, help='Frames per second to sample Video at')

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
    parser.add_argument("--restart_job", '-rj', action='store_true', help="overwrite all previous extractions")

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
