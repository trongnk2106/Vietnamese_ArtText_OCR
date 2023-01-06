CUDA_VISIBLE_DEVICES=2 \
	python3 train.py \
	--workers 0 \
	--device 2 \
	--batch-size 8 \
	--data ./data/data.yaml \
	--img 640 640 \
	--cfg cfg/training/yolov7.yaml \
	--weights '' \
	--name yolov7-epoch \
	--hyp data/hyp.scratch.p5.yaml	\
	--epochs 1000	\
	--sync-bn