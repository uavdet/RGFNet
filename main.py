from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO(r"ultralytics/cfg/models/v8/yolov8.yaml")
    # model.train(data=r"ultralytics/cfg/datasets/mydata.yaml")  # 训练

    model = YOLO(r"C:\Users\Desktop\YOLOv8多模态带分割\best.pt")
    # model.val(data=r"ultralytics/cfg/datasets/mydata.yaml",batch=1)  # 验证
    model.predict(source=r"C:\Users\Desktop\YOLOv8多模态带分割\datasets\111\images", save=True)  # 检测
