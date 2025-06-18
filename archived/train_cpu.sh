$env:KMP_DUPLICATE_LIB_OK = "TRUE"
yolo detect train data=dataset/screw/dataset.yaml model=models/yolov8n.pt epochs=20 batch=2 imgsz=320 device=cpu workers=0 cache=False
mv runs/detect/train/weights/best.pt models/best_screw.pt
