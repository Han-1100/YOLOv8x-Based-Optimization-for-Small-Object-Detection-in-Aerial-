import warnings
import numpy as np
import os
import subprocess

from mpmath import fraction
from ultralytics import YOLO

warnings.filterwarnings('ignore')


# def autoChooseCudaDevice():
#     try:
#         cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
#         result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
#         os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmin([int(x.split()[2]) for x in result[:-1]]))
#
#         os.system('echo $CUDA_VISIBLE_DEVICES')
#     except:
#         pass
#
#
# autoChooseCudaDevice()

if __name__ == '__main__':
    model = YOLO('../ultralytics/cfg/models/v8/yolov8x-ST.yaml')

    results = model.train(
        data='../ultralytics/cfg/datasets/VisDrone.yaml',
        epochs=100,
        batch=4,
        imgsz=640,
        device=3,
        optimizer='SGD',
        amp=False,
        cache=False,
    )
