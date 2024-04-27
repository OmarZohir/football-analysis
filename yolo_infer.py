from ultralytics import YOLO

model = YOLO('yolov8x')
results = model.predict('clips/08fd33_4.mp4', save=True, stream=True)
print('================================')
for r in results:
    boxes = r.boxes
    masks = r.masks
    probs = r.probs

for box in boxes:
    print(box)