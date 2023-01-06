CUDA_VISIBLE_DEVICES=2 \
   python3.7 detect.py \
   --weights './runs/train/yolov7-epoch/weights/best.pt' \
   --img-size 640 \
   --source '../dataset/private_test/uaic2022_private_test/images' \
   --conf-thres 0.5 \
   --iou-thres 0.5  \
   --device 2 \
   --save-txt \
   --save-conf \
