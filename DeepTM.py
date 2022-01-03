import numpy as np
import torch
import cv2
from network import Network
from collections import OrderedDict

checkpoint = 'model_best.pth'
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def input_transform(self, image):
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std
    return image


def eval(img, tmp):
    h, w = img.shape

    W = 480
    H = 640

    scale_x = W / w
    scale_y = H / h

    img = cv2.resize(img, dsize=(W,H), interpolation=cv2.INTER_AREA)
    tmp = cv2.resize(tmp, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_AREA)

    img = input_transform(img)
    tmp = input_transform(tmp)

    # 
    tmp_x = (int)(img.cols/2)-(int)(tmp.cols/2) 
    tmp_y = (int)(img.rows/2)-(int)(tmp.rows/2)
    tmp_w = tmp.cols
    tmp_h = tmp.rows

    tmp_in = np.zeros_like(img)
    tmp_in[tmp_y:tmp_y+tmp_h, tmp_x:tmp_x+tmp_w] = tmp[0:tmp_h, 0:tmp_w]

    with torch.no_grad():
        model.eval()
        (img, tmp_in) = (img.cuda(), tmp_in.cuda())
        y = model.forward(img, tmp_in)

    output = y.permute(0, 2, 3, 1).detach().cpu().numpy()*255
    output = output.astype(np.uint8)

    return output


def copy_state_dict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True) 

    category = ['positive', 'negative']

    model = Network()
    model = model.cuda()

    if checkpoint:
        print('Loading checkpoint')
        checkpoint = torch.load(checkpoint)
        DTOID_state_dict = copy_state_dict(checkpoint['DTOID'])
        model.load_state_dict(DTOID_state_dict)
