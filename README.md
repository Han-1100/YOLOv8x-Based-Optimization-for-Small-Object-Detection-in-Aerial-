# Enhanced YOLOv8 for Small Object Detection in UAV Aerial Images

## Overview

Small object detection in UAV aerial imagery remains challenging due to loss of fine details after multiple downsampling operations in standard YOLOv8, which only performs detection at three scales (P3, P4, P5). To address this issue, we propose an improved YOLOv8 model that enhances feature representation and detection accuracy for small targets.

## Method

1. **Backbone modification:**
   Replace the last two C2f modules with C3STR blocks that integrate Swin Transformer to preserve global context and enable cross-window attention.

2. **Multi-scale feature aggregation:**
   Replace the original SPPF with SPPCSPC to strengthen multi-scale contextual fusion.

3. **Four-scale detection:**
   Add a new P2 (stride=4) branch in the PAN-FPN and an additional detection head, forming a four-scale detection structure (P2–P5) to improve small object detection.

4. **Loss function optimization:**
   Use WIoU loss to stabilize small box regression and improve localization accuracy.

## Results on VisDrone2019

| Metric       | YOLOv8 Baseline | Improved Model |
| :----------- | :-------------: | :------------: |
| Recall       |      0.420      |    **0.479**   |
| mAP@0.5:0.95 |      0.267      |    **0.310**   |
| Precision    |      0.548      |    **0.570**   |

## Conclusion

By integrating Swin Transformer (C3STR), replacing SPPF with SPPCSPC, adding a P2 detection branch, and using WIoU loss, our enhanced YOLOv8 significantly improves small object detection performance on the VisDrone2019 dataset, achieving higher recall, precision, and mAP compared with the standard YOLOv8.