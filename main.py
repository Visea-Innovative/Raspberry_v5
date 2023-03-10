import glob, time
import cv2, sys
sys.path.append("ByteTrack/")
from model_utils import DetModel, Tracker, Count_persons, DrawandShow
import argparse
import warnings
warnings.filterwarnings("ignore")

def main(args):
    print("Starting ....")

    weights = args.weight
    device = "cpu"

    mode_flag = False
    if "onnx" in weights:
        mode_flag = True
    

    im_size = args.img_size
    detector = DetModel(weights, device, [im_size,im_size]) 
    tracker = Tracker()

    images = glob.glob(args.img_paths + "/*")

    CusIn, CusOut, StaffIn, StaffOut = 0 , 0 , 0 , 0 
    Persons = []

    Paspartu_Range = 0.15
    conf_thresh, iou_thresh = args.conf, args.iou 
    total_ms = 0
    for image_path in images:
        image = cv2.imread(image_path)
        t1 = time.time()
        trackInput, Pred_d_bboxes, Frame_shape = detector.detect(image, conf_thresh, iou_thresh)
        Pred_t_bboxes = tracker.track(trackInput, Pred_d_bboxes, Frame_shape)
        if args.show:
            DrawandShow(Pred_t_bboxes, image, [im_size, im_size], Paspartu_Range, mode_flag)
        Persons, CusIn, CusOut, StaffIn, StaffOut = Count_persons(Pred_t_bboxes, [im_size, im_size] , Persons, CusIn, CusOut, StaffIn, StaffOut, Paspartu_Range, image.shape[:2], mode_flag)
        timer = (time.time() - t1)*1000
        total_ms += timer
        print("\rCusIn: {} CusOut: {} StaffIn: {} StaffOut: {} Process Time: {:.2f} ms".format(CusIn, CusOut, StaffIn, StaffOut, timer),end="")
    print("\n" + "*" * 70)
    print("Avg process time: {:.2f} ms".format(total_ms/len(images)))

if __name__ == '__main__':

    parser = argparse.ArgumentParser('RaspberryPi-Visea')
    parser.add_argument('--weight', type=str, default = "D:/Huseyin/yolov5-master/runs/train\exp\weights/best.onnx")
    parser.add_argument('--img_paths', type=str, default = "C:/Users\Huseyin\Desktop\Raspberry\images")
    parser.add_argument('--conf', type=float, default = 0.1)
    parser.add_argument('--iou', type=float, default = 0.5)
    parser.add_argument('--img_size', type=int, default = 320)
    parser.add_argument('--show', type=bool, default = True)
    args = parser.parse_args()
    main(args)