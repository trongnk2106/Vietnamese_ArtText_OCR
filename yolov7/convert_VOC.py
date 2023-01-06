import os
from PIL import Image, ImageDraw, ImageFont
import cv2
import argparse



def parse_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--path_detect', type=str, default="./runs/detect/exp/labels", help='Path of results detector: txt file ')
    parser.add_argument('--path_img', type=str, 
                                        default = "../dataset/private_test/uaic2022_private_test/images",
                                        help='input image path of folder')                       
    parser.add_argument('--path_output', type=str, default="./runs/detect/exp/labels_2" , help='paht out of results')
    args = parser.parse_args()
    return args
    _
def yolobbox2bbox(x,y,w,h, path_image):
    image = Image.open(path_image)
    width,height = image.size
    x1, y1 = x-w/2, y-h/2
    x2, y2 = x+w/2, y+h/2
    return int(x1*width), int(y1*height), int(x2*width), int(y2*height)

def draw_box(list_box,image,idx):
    if len(list_box)<=0:
        return
    save_path = './draw_box'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for box in list_box:
        # print(box)
        start = (int(box[0]),int(box[1]))
        end = (int(box[2]),int(box[3]))
        print(start, end)
        color = (255, 255, 0)
        image = cv2.rectangle(image,start,end,color,2)
    path_save = os.path.join(save_path,f'{idx}.jpg')
    cv2.imwrite(path_save, image)

def cover_4poin(x,y,w,h,path_image, pre):
    l, t, r, b = yolobbox2bbox(x,y,w,h,path_image)
    # draw_box([[l, t, r, b]],Image.open(path_image),20)
    str_save = f"{l},{t},{r},{t},{r},{b},{l},{b}\t{pre}\n"
    return str_save

def update_label(path_file_label,path_file_image,path_label_save):
    with open(path_file_label, 'r') as f:
        data = f.read().split("\n")
        if data[-1] == "":
            data = data[:-1]
    f.close()
    log_save = ""
    for line in data:
            line = line.split(" ")
            box = [float(i) for i in line[1:-1]]
            pre = float(line[-1])
            if pre > 0.5:
                str_save = cover_4poin(box[0], box[1], box[2],box[3], path_file_image, pre)
                # print(str_save)
                log_save += str_save
        # print(data)
    f = open(path_label_save,'w')
    f.write(log_save)
    f.close()
def main(path_label,path_image,path_save):
    if not(os.path.exists(path_save)):
        os.makedirs(path_save)
    for file in os.listdir(path_label):
        print(file)
        path_file_label = os.path.join(path_label,file)
        path_file_image = os.path.join(path_image,file.split(".")[0]+".jpg")
        path_label_save = os.path.join(path_save,file)
        update_label(path_file_label,path_file_image,path_label_save)

if __name__ == "__main__":
    args = parse_args()
    path_label = args.path_detect
    path_image = args.path_img
    path_save = args.path_output
    # path_label = "./runs/detect/exp4/labels"
    # path_image = "../dataset/private_test/uaic2022_private_test/images"
    # path_save = "./runs/detect/exp4/labels_2"
    main(path_label,path_image,path_save)
