yolo detect train \
  data=dataset/screw/yolo_format/dataset.yaml \
  model=models/yolov8n.pt \
  epochs=50 \
  batch=16 \
  imgsz=640 \
  device=0 \
  workers=4 \
  cache=ram
mv runs/detect/train/weights/best.pt models/best_screw.pt
