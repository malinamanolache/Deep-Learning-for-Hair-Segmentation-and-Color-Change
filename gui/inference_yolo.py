import torch
from networks.models import Darknet
from utils.general import non_max_suppression
import cv2
import numpy as np

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, auto_size=32):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, auto_size), np.mod(dh, auto_size)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def crop_images(image, bb_list):
    sorted_boxes = sorted(bb_list, key=lambda bb: bb[1])

    top_box = sorted_boxes[-1]

    x, y, width, height = top_box

    image_height, image_width, _ = image.shape
    left = int((x - width / 2) * image_width)
    top = int((y - height / 2) * image_height)
    right = int((x + width / 2) * image_width)
    bottom = int((y + height / 2) * image_height)

    # Crop the image
    cropped_image = image[top:bottom, left:right]
    return cropped_image, top_box


def infer_yolo(original_image):
   
    weights = "/home/intern2/Deep-Learning-for-Hair-Segmentation-and-Color-Change/final_models_licenta/best.pt"
    cfg = "/home/intern2/Deep-Learning-for-Hair-Segmentation-and-Color-Change/train_yolov4/cfg/yolov4.cfg"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model = Darknet(cfg, 512).cuda()

    model.load_state_dict(torch.load(weights, map_location=device)['model'])
    model.to(device).eval()

    # img = original_image.copy()
    img = letterbox(original_image, new_shape=512, auto_size=64)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    with torch.no_grad():
        pred = model(img, augment=False)[0]

        pred = non_max_suppression(pred,0.001, 0.7, classes=[0], agnostic=False)
        # print(pred)
        bb_list = []
        for i, det in enumerate(pred):
            gn = torch.tensor(original_image.shape)[[1, 0, 1, 0]]

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size

                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], original_image.shape).round()
                s=''
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, "hair")  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    bb_list.append(xywh)

                    label = 'hair %.2f' % (conf)
                    orig = original_image.copy()
                    plot_one_box(xyxy, orig, label=None, color=[255,0,0], line_thickness=3)

    cropped_img = crop_images(original_image, bb_list)

    return cropped_img
            
    # cv2.imwrite('C:\\Users\\Maly\\licenta\\postprocessing\\test.png', cropped_img)

    # print('%sDone.' % (s))

# image_path = "C:\\Users\\Maly\\licenta\\postprocessing\\original\\Frame00010-org.jpg"
# image = cv2.imread(image_path)
# infer_yolo(image)