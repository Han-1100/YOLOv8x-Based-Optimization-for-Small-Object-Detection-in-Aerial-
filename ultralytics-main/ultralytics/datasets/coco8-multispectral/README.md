# Ultralytics COCO8-Multispectral Dataset

Ultralytics COCO8-Multispectral is a small, but versatile object detection dataset composed of the first 8 images of the COCO train 2017 set, converted to multispectral format with 10 spectral bands. The dataset includes 4 images for training and 4 for validation. This dataset is ideal for testing and debugging multispectral object detection models, or for experimenting with new multispectral detection approaches.

The images have been converted from RGB to 10-channel multispectral data using spectral interpolation across the visible light spectrum (450-700nm). Each band represents a specific wavelength, providing more detailed spectral information than traditional RGB images. With 8 images, it is small enough to be easily manageable, yet diverse enough to test training pipelines for errors and act as a sanity check before training larger multispectral datasets.

This dataset is intended for use with Ultralytics YOLO and serves as a foundation for developing models capable of processing multispectral imagery.

Docs: https://docs.ultralytics.com
Community: https://community.ultralytics.com
GitHub: https://github.com/ultralytics/ultralytics
[README.md](../dota8-multispectral/README.md)
