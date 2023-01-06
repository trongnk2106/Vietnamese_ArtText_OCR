from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import cv2
import os
from PIL import Image
import argparse
from tqdm import tqdm
import numpy as np
from ESRGAN import RRDBNet_arch as arch
# import ESRGAN.RRDBNet_arch as arch
import glob
# import os.path as osp
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--cfg_name', type=str, default = 'vgg_transformer',  help='Config file name')
    parser.add_argument('--weights', default = './weights/vietocr_artext.pth' ,type=str, help='Checkpoint file path')
    parser.add_argument('--path_detect', type=str, default="../res", help='Path of results detector: txt file ')
    parser.add_argument('--path_img', type=str, 
                                        default = "../test_data/uaic2022_public_valid/images",
                                        help='input image path of folder')
    parser.add_argument('--device', default="cuda:0") 
    parser.add_argument('--submission', type=int, default=0)                            
    parser.add_argument('--path_output', type=str, default="../Results/vietocr" , help='paht out of results')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    config = Cfg.load_config_from_name(args.cfg_name)
    config['vocab'] = 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~° ' + '̉'+ '̀' + '̃'+ '́'+ '̣' + '´' + '’' +  '‘' +  'Ð'
    config['device'] = args.device
    config['weights'] = args.weights
    
    
    model_path = './ESRGAN/models/RRDB_ESRGAN_x4.pth'
    esr_device = torch.device(args.device)
    model = arch.RRDBNet(3, 3, 64, 23, gc = 32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(esr_device)

    print(f'LOAD CONFIG: {args.cfg_name}')

    detector = Predictor(config)

    if not os.path.exists(args.path_output):
        os.makedirs(args.path_output)

    for img_name in tqdm(sorted(os.listdir(args.path_img))):
        filename = img_name.split('.')[0]
        
        img = cv2.imread(os.path.join(args.path_img, img_name), 0)
        out_txt = open(os.path.join(args.path_output, f'{filename}.txt'), 'w', encoding="utf-8")
        try:
            file_txt = open(os.path.join(args.path_detect, f'{filename}.txt'), 'r', encoding='utf-8')
        except:
            out_txt.close()
            file_txt.close()
            continue
            
        lines = file_txt.readlines()
        for line in lines:
            prob_detect = line.split('\t')[-1][:-1]
            prob_detect = 'TIU'
            bbox = line.split('\t')[0]
            #add by trongnt
            # bbox = line
            bbox = bbox.split(',')
            pts = bbox.copy()
            pts = [int(p) for p in pts]
            pts = np.array(pts).astype(np.int32).reshape((-1, 2))
            rect = cv2.boundingRect(pts)
            x, y, w, h = rect
            croppeded = img[y:y+h, x:x+w].copy()
            pts = pts - pts.min(axis=0)
            mask = np.zeros(croppeded.shape[:2], np.uint8)  
            # print(img_name)
            cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
            dst = cv2.bitwise_and(croppeded, croppeded, mask= mask)
            bg = np.ones_like(croppeded, np.uint8) * 255
            cv2.bitwise_not(bg, bg, mask=mask)
            crop_img =  bg + dst
            # print(crop_img.shape)
            # prob_detect = line.split('\t')[-1][:-1]
            # prob_detect = 'TIU'
            # bbox = line.split('\t')[0]
            # # bbox = line
            # bbox = bbox.split(',')
            # bbox = [int(ele) for ele in bbox]

            # top = min(bbox[1:8:2])
            # bottom = max(bbox[1:8:2])
            # left = min(bbox[0:7:2])
            # right = max(bbox[0:7:2])

            # if top == 0: top = 1
            # if left == 0: left = 1
            # x = int((left + right)/2)
            # y = int((top+ bottom)/2)
            # height = int((bottom - y) *)
            # width = int((right - x) * 1.05)
            # crop_img = img[y - height: y + height, x - width: x + width]
            try:
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            except:
                continue
            if crop_img.shape[0] < 20 or crop_img.shape[1] < 20:
                # print('super resolution')
                crop_img = crop_img * 1.0 / 255
                crop_img = torch.from_numpy(np.transpose(crop_img[:, :, [2, 1, 0]], (2, 0, 1))).float()
                img_LR = crop_img.unsqueeze(0)
                img_LR = img_LR.to(esr_device)

                with torch.no_grad():
                    output_super = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
                output_super = np.transpose(output_super[[2, 1, 0], :, :], (1, 2, 0))
                output_super = (output_super * 255.0).round()
                img_pil =  Image.fromarray((output_super * 1).astype(np.uint8)).convert('RGB')
                # print(output_super.shape)
            else:
                # output_super = crop_img
            # img_pil =  Image.fromarray((output_super * 1).astype(np.uint8)).convert('RGB')
                img_pil = Image.fromarray(crop_img)
            pred, prob = detector.predict(img_pil , return_prob = True)
            pred = pred.strip()
            
            if args.submission==True:
                content = ','.join([str(p).strip() for p in bbox]) + ',' + pred + '\n'
            else:
                content = str(prob_detect[:-1]) + ' '
            out_txt.write(content)
            
        out_txt.close()
        file_txt.close()
if __name__ == '__main__':
    main()
