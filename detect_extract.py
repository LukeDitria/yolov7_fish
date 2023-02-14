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

    # Use os.walk() to loop through the nested directories
    for root, dirs, files in os.walk(opt.root_source_dir):
        if len(files) > 0:
            sub_dir = save_dir + root.replace(opt.root_source_dir, '')
            # Loop through the list of files
            for file in files:
                file_path = os.path.join(root, file)
                file_name = file.split(".")[0]
                file_type = file.split(".")[-1]

                if file_type.lower() in valid_file_ext:
                    new_file_dir = os.path.join(sub_dir, file_name)
                    if not os.path.isdir(new_file_dir):
                        os.makedirs(new_file_dir)

                    summary_filepath = os.path.join(new_file_dir,  " video_summary.csv")
                    if not os.path.isfile(summary_filepath) or opt.restart_job:
                        summary_dict = extract_frames(file_path, file_name, model, opt.img_size, opt.sample_fps, new_file_dir, device)
                        summary_dict.to_csv(summary_filepath)


def extract_frames(file_path, file_name, model, imgsz, sample_fps, save_dir, device):
    # Set Dataloader
    data_logger = defaultdict(lambda: 0)

    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    # dataset = LoadImages(file_path, img_size=imgsz, stride=stride)

    temp_filepath = os.path.join("/export/home/s2997103/lscratch", file_name)
    shutil.copyfile(file_path, temp_filepath)

    cap = cv2.VideoCapture(temp_filepath)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))

    if sample_fps > video_fps:
        raise ValueError("Sample FPS (%d) cannot be greater than Video FPS (%d)" % (sample_fps, video_fps))

    sample_rate = video_fps//sample_fps

    frame_num = 0
    for fno in trange(0, total_frames, sample_rate):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        _, im0 = cap.read()

        img = letterbox(im0, imgsz, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        frame_num += 1
        img = torch.tensor(img).unsqueeze(0).to(device)/255

        # Inference
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            with torch.cuda.amp.autocast():
                pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Process detections
        for i, det in enumerate(pred):  # detections per image

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                if "person" in names:
                    person_detected = det[:, -1] == names.index("person")

                    if person_detected.sum() > 0:
                        max_person_conf = (det[:, -2][person_detected]).max().item()
                        if max_person_conf > 0.7:
                            break

                frame_name = file_name + "_frame_" + str(frame_num).zfill(8)
                frame_save_path = os.path.join(save_dir, frame_name)

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    # line = (names[int(cls.item())], *xywh)  # label format
                    class_name = names[int(cls)]
                    label = f'{class_name} {conf:.2f}'
                    # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                    data_logger[class_name] += 1

                    # txt_path = os.path.join(save_dir, frame_name)
                    # with open(txt_path + '.txt', 'a') as f:
                    #     f.write('%s, %.2f ' % (label, conf) + '\n')

                cv2.imwrite(frame_save_path + ".jpg", im0)

    if os.path.isfile(temp_filepath):
        os.remove(temp_filepath)

    return pd.Series(dict(data_logger), name='Count')


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
