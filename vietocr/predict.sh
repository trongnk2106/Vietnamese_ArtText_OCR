python3 predict.py \
--device cuda:4 \
--weights ./weights/reg_final_argu.pth \
--path_detect  ../yolov7/runs/detect/exp4/labels_2 \
--path_img ../dataset/private_test/uaic2022_private_test/images \
--path_output ../Results/final \
--submission 1


