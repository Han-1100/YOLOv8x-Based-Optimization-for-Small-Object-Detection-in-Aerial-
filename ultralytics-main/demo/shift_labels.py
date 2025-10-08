import os
import numpy as np
import cv2


def get_classes(labels_dir: str):
    """
    Count and return classes
    :arg labels_dir: directory of labels
    :return list of sorted classes
    """
    classes = set()
    for file in os.listdir(labels_dir):
        file_path = os.path.join(labels_dir, file)
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = parts[-1]  # get class
                    classes.add(cls)
    return sorted(classes)


def convert_to_yolo_format(labels_dir: str, output_dir: str):
    """
    Convert the labels to YOLO format, with categories sorted and encoded according to their initial letters
    :arg labels_dir: directory of original labels
    :arg output_dir: directory of converted labels
    :return list of encoded and sorted labels
    """
    # get classes
    classes = get_classes(labels_dir)

    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(labels_dir):
        file_path = os.path.join(labels_dir, file)
        output_path = os.path.join(output_dir, file)

        with open(file_path, "r", encoding="utf-8") as f_in, \
                open(output_path, "w", encoding="utf-8") as f_out:

            for line in f_in:
                parts = line.strip().split()
                if len(parts) >= 5:
                    x1, y1, x2, y2 = map(float, parts[:4])
                    cls_name = parts[4]

                    img_width = 800
                    img_height = 800

                    center_x = (x1 + x2) / 2 / img_width
                    center_y = (y1 + y2) / 2 / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height

                    cls_idx = class_to_idx[cls_name]

                    f_out.write(f"{cls_idx} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")

    return classes


def visualize_yolo_annotations(image_path, label_path, classes_list):
    """
    Visualize annotations to check
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"No image at: {image_path}")
        return

    orig_h, orig_w = img.shape[:2]

    with open(label_path, 'r', encoding='utf-8') as f:
        annotations = f.readlines()

    # Bbox
    for ann in annotations:
        parts = ann.strip().split()
        if len(parts) >= 5:
            cls_idx = int(parts[0])
            center_x = float(parts[1]) * orig_w
            center_y = float(parts[2]) * orig_h
            width = float(parts[3]) * orig_w
            height = float(parts[4]) * orig_h

            x1 = int(center_x - width / 2)
            y1 = int(center_y - height / 2)
            x2 = int(center_x + width / 2)
            y2 = int(center_y + height / 2)

            color = (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Add class
            if cls_idx < len(classes_list):
                label = f"{classes_list[cls_idx]}"
                cv2.putText(img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('YOLO Annotation Check', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    labels_folder = "D:/YOLO/AI-TOD/train/labels_backup"
    classes_list = get_classes(labels_folder)
    print("Classes: ", classes_list)

    labels_dir_train = "D:/YOLO/AI-TOD/train/labels_backup"
    output_dir_train = "../ultralytics/datasets/ai-tod/labels/train"
    labels_dir_val = "D:/YOLO/AI-TOD/val/labels_backup"
    output_dir_val = "../ultralytics/datasets/ai-tod/labels/val"

    classes_train = convert_to_yolo_format(labels_dir_train, output_dir_train)
    classes_val = convert_to_yolo_format(labels_dir_val, output_dir_val)

    print("Mapping of classes：")
    for idx, cls_name in enumerate(classes_train):
        print(f"{idx}: {cls_name}")

    image_path = "D:/YOLO/AI-TOD/val/images/88__2577_600.png"
    label_path = "D:/YOLO/AI-TOD/val/labels/88__2577_600.txt"
    visualize_yolo_annotations(image_path, label_path, classes_list)
