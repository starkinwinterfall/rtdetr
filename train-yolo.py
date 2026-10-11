import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import RTDETR


if __name__ == '__main__':
    model = RTDETR('/root/autodl-tmp/ultralytics/ultralytics/cfg/models/11/yolo11l.yaml')
    # model.load('') # loading pretrain weights

    model.train(
        data='/root/autodl-tmp/RTDETR-MK/dates_cizhuan_6714.yaml',
        cache=False,
        imgsz=640,
        epochs=250,
        batch=4,
        workers=4,
        patience=0,
        project='runs/train',
        name='exp',
        mosaic=0.1,
        mixup=0.05,
        close_mosaic=10,

    )
