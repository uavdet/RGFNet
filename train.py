from ultralytics import YOLO
# import torch

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# m = m.to(device)

model = YOLO('ultralytics\cfg\mamba_end_obb\yolov8s_MBVSS2x3_add_malign_refrgb_new.yaml').load('yolov8s-obb.pt')
# build from YAML and transfer weights   通过模型配置和预训练模型将预训练模型权重转到模型上

# Train the model
results = model.train(data='dataset/DroneVehicle/data.yaml', epochs=150, imgsz=640, device = 0, batch =8, 
                       name='DV_MBVSS2x3_add_malign_refrgb')


#开始训练，注意data参数是我们的数据集配置，imgsz在训练和测试时都需要指定6363  save_period=5,