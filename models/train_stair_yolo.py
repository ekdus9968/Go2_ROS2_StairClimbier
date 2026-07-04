from ultralytics import YOLO

model = YOLO('yolo11n.pt')  # COCO pretrained

results = model.train(
    data='/home/jen/projects/go2-stair-climber/datasets/stairnet_yolo/data.yaml',
    epochs=50,
    imgsz=640,
    batch=32,
    device=0,
    project='/home/jen/projects/go2-stair-climber/models',
    name='stair_yolo11n',
    exist_ok=True,
    patience=10,
    save=True,
)

metrics = model.val()
print(f'mAP50: {metrics.box.map50:.4f}, mAP50-95: {metrics.box.map:.4f}')
