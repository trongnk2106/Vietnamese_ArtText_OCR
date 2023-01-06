# VIETNAMESE ARTTEXT OCR - TIU TEAM


## Chia bài toán thành 2 task :
### Detection : Yolov7
### Regconition : Vietocr
### Sử dụng thêm ESRGAN thực hiện super resolution cho những ảnh nhỏ, nhòe trước khi cho 

#### Chúng tôi thực hiện argument thêm data, có thể xem thử repo :
[Markdown](https://github.com/UIT-20521888/argu_data.git)


#### Cài đặt:

##### Gitclone repo: 

##### Cài đặt yolov7 theo hướng dẫn tại repo gốc 
[Markdown](https://github.com/WongKinYiu/yolov7.git)


#### link pretrain yolov7 : 
[Markdown](https://drive.google.com/file/d/16lWdKYaMSemKlswkJD7Gr4S1zjtVx8vK/view?usp=share_link)

##### Train: 
> bash train.sh
 
##### Test:
> bash infer.sh

##### Trường hợp ảnh không nhận dạng được bất kì đối tượng nào,YOLOv7 không lưu lại output của ảnh đó. Nên khắc phục bằng cách thực hiện:

##### Fill miss file :
> bash fill_miss_file.sh

##### Convert tạo độ:
> bash convert.sh 

##### Lưu ý : Thay đổi các đường dẫn trước khi chạy trong các file.sh

#### Cài đặt vietocr theo hướng dẫn của repo gốc, hoặc build docker image từ dockerfile trong folder vietocr.Chúng tôi đã tích hợp super resolution bằng ESRGAN vào quá trình infer 

[Markdown](https://github.com/pbcquoc/vietocr.git)

##### train: 
> python3 train.py 

##### Lưu ý thay đổi đường dẫn config trong file  train.py

##### test :
> bash predict.sh
##### Lưu ý: Thay đổi các tham số trong file.sh 

#### Link pretrain vietocr: 

[Markdown](https://drive.google.com/file/d/1M4LVXNV71CGJGtkgjv5ax9Xy2QhXZPer/view?usp=share_link)

