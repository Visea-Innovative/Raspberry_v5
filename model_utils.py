import math
import time

import cv2
import numpy as np
import torch
import torch.nn as nn

from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from libs.yolov7models.common import Conv
from libs.yolov7utils.torch_utils import TracedModel
import torchvision
from models.common import DetectMultiBackend
import onnx
import onnxruntime as ort
ort.set_default_logger_severity(3)
import numpy as np


real_imgw , real_imgh = 640 , 360

class TrackArgs:
    track_thresh = 0.5
    track_buffer = 30
    match_thresh = 0.7
    aspect_ratio_thresh = 10.0
    min_box_area = 1.0
    mot20 = False


class Tracker():

    def __init__(self):
        args = TrackArgs()
        self.tracker = BYTETracker(args, frame_rate=10)
    
    def track(self, trackInput, Pred_d_bboxes, Frame_shape):
        trackInput = np.float32(trackInput)
        imh, imw, _ = Frame_shape 
        Pred_t_bboxes = []
        try:
            trackInput = torch.from_numpy(trackInput)
            online_targets = self.tracker.update(trackInput, [imh,imw], (imh,imw) )
        except:
            Pred_d_bboxes_torch =torch.empty(1,5)
            online_targets = self.tracker.update(Pred_d_bboxes_torch , [imh, imw] , (imh,imw))
        min_box_area = 10  

        for i ,t in enumerate(online_targets):
                tlwh = t.tlwh
                tid = t.track_id
                tlbr = t.tlbr
                if tlwh[2] * tlwh[3] > min_box_area:
                    Pred_t_bboxes.append([tlbr,tid]) 
        out_boxes = match_boxes(Pred_d_bboxes, Pred_t_bboxes)
        return out_boxes


def match_boxes(boxlist1, Pred_t_bboxes):
    matches = []
    for i in range(len(boxlist1)):
        max_iou = 0
        match = -1
        for l, tracklet in enumerate(Pred_t_bboxes):
            iou_value = iou(boxlist1[i][0], tracklet[0])
            if iou_value > max_iou:
                max_iou = iou_value
                match = l
        matches.append(match)
    out_boxes = []
    for k, box in enumerate(boxlist1):
        bbox = box[0]
        c = box[-1]
        index = matches[k]
        if index == -1:
            continue
        else:
            out_boxes.append([bbox, Pred_t_bboxes[index][-1], c])

    return out_boxes


def iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2.
    xi1 = max(x1_min, x2_min)
    yi1 = max(y1_min, y2_min)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)
    inter_area = max(yi2 - yi1, 0) * max(xi2 - xi1, 0)

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (y1_max - y1_min) * (x1_max - x1_min)
    box2_area = (y2_max - y2_min) * (x2_max - x2_min)
    union_area = box1_area + box2_area - inter_area

    # Compute the IoU
    iou = inter_area / union_area

    return iou

class DetModel:
    def __init__(self,weights, device, image_size= [416, 416]):
        self.device = torch.device(device)
        self.model = DetectMultiBackend(weights, device=self.device, dnn=False, data="libs/coco128.yaml", fp16=False)
        stride, pt, jit, engine = self.model.stride, self.model.pt, self.model.jit, self.model.engine
        imgsz = image_size[0]
        imgsz = check_img_size(imgsz, s=stride)
        self.model.warmup(imgsz=(1 if pt else 1, 3, imgsz, imgsz))
        self.imgsz = image_size

    def detect(self, image, conf_thres, iou_thres):
        Pred_d_bboxes = []
        trackInput = []

        input_w , input_h  = self.imgsz

        imgh, imgw, _ = image.shape
        rat = imgh/imgw
        Frame = image.copy()
        im = cv2.resize(image,(input_w, int(rat*input_w)))
        h,w,_ = im.shape

        
        im = letterbox(im, self.imgsz, stride=64,auto = False)[0]
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.device)

        im = im.float()
        im /= 255
        im = torch.unsqueeze(im, 0)
        pred = self.model(im, augment=False, visualize=False)
        
        pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=50)

        for i , det in enumerate(pred):
            if len(det):
                    predn = det.clone()
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], Frame.shape).round()                  
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)
                        x1 , y1 , x2 , y2 = xyxy
                        x1 , y1 , x2 , y2 = int(x1), int(y1), int(x2), int(y2)
                        conf = float(conf)
                        trackInput.append([x1,y1,x2,y2,conf])
                        Pred_d_bboxes.append([[x1,y1,x2,y2],None,cls])
        return trackInput, Pred_d_bboxes, Frame.shape

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2

def Count_persons(pred_bboxes, imgsz , Persons, CusIn, CusOut, StaffIn, StaffOut, Paspartu_Range, im_shape, mode_flag):
    real_imgh , real_imgw = im_shape
    rat = real_imgh / real_imgw
    imgh, imgw = imgsz
    # pred_img_w, pred_img_h  = imgw , int(rat*imgh)
    pred_img_w, pred_img_h  = imgw , imgh
    coff_w , coff_h =  real_imgw/pred_img_w , real_imgh/pred_img_h
    TempPersons = []
    FinalPersons = []

    threshLine = real_imgh//2
    high_line = real_imgh *(1-Paspartu_Range)
    low_line = real_imgh * Paspartu_Range


    for box in pred_bboxes:
        checkflag = False
        for person in Persons:
            (x1,y1,x2,y2) , id , c = box
            if not mode_flag:
                x1 , y1, x2, y2 = int(x1*coff_w) ,int(y1*coff_h), int(x2*coff_w), int(y2*coff_h)
            if id == person.id:
                checkflag = True
                if (high_line > (y1 + y2)/2 > low_line):
                    person.update(((x1+x2)/2, (y1+y2)/2),c)
                    TempPersons.append(person)
        if not checkflag:
            (x1,y1,x2,y2) , id , c = box
            if not mode_flag:
                x1 , y1, x2, y2 = int(x1*coff_w) ,int(y1*coff_h), int(x2*coff_w), int(y2*coff_h)
            if (high_line > (y1 + y2)/2 > low_line):
                TempPersons.append(Person(id,c,((x1+x2)/2, (y1+y2)/2)))
                TempPersons[-1].updated = True
        checkflag = False

    for person in Persons:
        if not person.updated and person.Age <10:
            TempPersons.append(person)

    for person in TempPersons:
        if not person.updated:
            person.Age += 1
            if person.Age  < 10:
                FinalPersons.append(person)
            else:
                if person.FirstLoc[1] > threshLine and person.LastLoc[1] < threshLine:
                    if person.Class < 0:
                        CusOut += 1
                    else:
                        StaffOut += 1
                if person.FirstLoc[1] < threshLine and person.LastLoc[1] > threshLine:
                    if person.Class < 0:
                        CusIn += 1
                    else:
                        StaffIn += 1
        else:
            FinalPersons.append(person)
        person.updated = False
    
    return FinalPersons, CusIn, CusOut, StaffIn, StaffOut


def attempt_load(weights, map_location=None):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        
        ckpt = torch.load(w, map_location=map_location)  # load
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model
    
    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
    
    if len(model) == 1:
        return model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble

class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None

def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor

def check_img_size(img_size, s=32):
    # Verify img_size is a multiple of stride s
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size

def letterbox(im, new_shape=(416, 416), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=False, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def scale_coords(ori_shape, boxes, target_shape):
    '''Rescale the output to the original image shape'''
    ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
    padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (ori_shape[0] - target_shape[0] * ratio) / 2

    boxes[:, [0, 2]] -= padding[0]
    boxes[:, [1, 3]] -= padding[1]
    boxes[:, :4] /= ratio

    boxes[:, 0].clamp_(0, target_shape[1])  # x1
    boxes[:, 1].clamp_(0, target_shape[0])  # y1
    boxes[:, 2].clamp_(0, target_shape[1])  # x2
    boxes[:, 3].clamp_(0, target_shape[0])  # y2

    return boxes

def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)


    return output

def xywh2xyxy(x):
    '''Convert boxes with shape [n, 4] from [x, y, w, h] to [x1, y1, x2, y2] where x1y1 is top-left, x2y2=bottom-right.'''
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


class Person:
    def __init__(self, id, PersonClass, FirstLoc, GTid = None):
        self.id = id
        self.Class = PersonClass
        self.FirstLoc = FirstLoc
        self.LastLoc = FirstLoc
        self.Age = 0
        self.updated = False
        self.GTid = GTid

    def update(self, newCoor, c):
        self.LastLoc = newCoor
        self.updated = True
        if c == 0:
            self.Class -= 1
        else:
            self.Class += 1
    
    def refresh(self, newid , c, newCoor):
        self.id = newid
        self.Class = c
        self.updated = True
        self.LastLoc = newCoor
    
    def close(self):
        self.updated = False

    def getOld(self):
        self.Age += 1
        self.updated = True


def DrawandShow(bboxes, image, imgsz, Pr, mode_flag):
    real_imgh , real_imgw = image.shape[:2]
    imgh, imgw, = imgsz
    rat = real_imgh / real_imgw
    pred_img_w, pred_img_h  = imgw , int(rat*imgh)
    # pred_img_w, pred_img_h  = imgw , imgh
    coff_w , coff_h =  real_imgw/pred_img_w , real_imgh/pred_img_h

    

    for box, id, classid in bboxes:
        x1 , y1, x2, y2 = box
        if not mode_flag:
            x1 , y1, x2, y2 = int(x1*coff_w) ,int(y1*coff_h), int(x2*coff_w), int(y2*coff_h)
        xc , yc  = (x1+x2)//2 , (y1+y2)//2
        if not (real_imgw*Pr < xc < real_imgw*(1-Pr) and real_imgh*Pr < yc < real_imgh*(1-Pr)):
            cv2.rectangle(image,(x1,y1),(x2,y2),(255,255,255),2)
            cv2.circle(image,((x1+x2)//2,(y1+y2)//2),5,(255,255,255),-1)
            cv2.putText(image,"ID: {}".format(id),(int(x1+x2)//2+10,int(y1+y2)//2),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),6,cv2.LINE_AA)
            cv2.putText(image,"ID: {}".format(id),(int(x1+x2)//2+10,int(y1+y2)//2),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)
        else:
            if classid == 0:
                cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.circle(image,((x1+x2)//2,(y1+y2)//2),5,(0,255,0),-1)
                cv2.putText(image,"ID: {}".format(id),(int(x1+x2)//2+10,int(y1+y2)//2),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2,cv2.LINE_AA)
            else:
                cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,255),2)
                cv2.circle(image,((x1+x2)//2,(y1+y2)//2),5,(0,0,255),-1)
                cv2.putText(image,"ID: {}".format(id),(int(x1+x2)//2+10,int(y1+y2)//2),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2,cv2.LINE_AA)
    cv2.imshow("", image)
    cv2.waitKey(1)