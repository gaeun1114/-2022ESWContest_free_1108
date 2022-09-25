import argparse
import os
import platform
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams
from utils.general import (LOGGER, check_img_size, check_imshow, check_requirements, cv2,
                           non_max_suppression, print_args, scale_coords, strip_optimizer)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode
from utils.track import tracking_update, delete_list, count_person,classify,choose_tracking
import cv2

# import firebase package

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import storage
import datetime
from PIL import ImageGrab

from uuid import uuid4



@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        update=False,  # update all models
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    source = str(source)

    # firebase upload
    cred = credentials.Certificate("accountkey.json")
    firebase_admin.initialize_app(cred,{'storageBucket' : 'real-push-b562d.appspot.com'})
    db = firestore.client()
    doc_ref = db.collection(u'crosswalk').document(u'detecting')
    bucket = storage.bucket()

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    view_img = check_imshow()
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
    bs = len(dataset)  # batch_size

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows = 0, []
    tracking = []
    dt = []
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
 
        # Inference
        pred = model(im, augment=augment)

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
      
        # Process predictions
        transp = []
        n = 0
    
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path[i], im0s[i].copy(), dataset.count

            p = Path(p)  # to Path
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            ROI = [1,200,635,470] # x_min, y_min, x_max, y_max
            color = (0,255,0)
            thickness = 5


            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                person_x =[]
                person_y =[]

                # count person
                n = count_person(det)

                # get person/transp coordinates
                classify(det,person_x,person_y,ROI,transp)

                # calculate length and start tracking
                choose_tracking(person_x, person_y,transp,tracking,dt)
      
                # data update
                doc_ref.set({u'person' : f'{n}'})
                  
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))


            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                
               
                
                
                
                
                
                x = ROI[0]
                y = ROI[1]
                w = ROI[2] - ROI[0]
                h = ROI[3] - ROI[1]
                roi = im0[y:y+h, x:x+w]
                cv2.rectangle(img = roi, pt1 = (0,0),pt2 = (w,h), color = color, thickness = thickness)
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

        # tracking update
        if(len(tracking)>0):
            tracking_update(tracking,transp,dt)

            # accident
            if 10 in dt:
                LOGGER.info(f'accident')
                
                ac = [i for i in range(len(dt)) if dt[i] >= 10]
                dt = delete_list(ac,dt)
                tracking = delete_list(ac,tracking)
               
                global time
                time = (datetime.datetime.now().strftime('%Y-%m-%d %Hh%Mm%S.%fms'))
                img=ImageGrab.grab()
                saveas="{}{}".format(f'{time}', '.jpg')
                print('capture successfully')
                img.save(saveas)

                fileUpload(bucket,saveas)
 
        # Print time (inference-only)
        LOGGER.info(f'{n}')

    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

def fileUpload(bucket,saveas):
    blob = bucket.blob('accident/'+f'{saveas}')
    new_token = uuid4()
    metadata = {"firebaseStorageDownloadTokens": new_token} 
    blob.metadata = metadata

    #upload file
    blob.upload_from_filename(filename= f'{saveas}', content_type='image/jpeg')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
