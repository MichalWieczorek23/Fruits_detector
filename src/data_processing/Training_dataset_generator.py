import json
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import re
import shutil

# Arguments box1, box2 passed in the following format: [x1, y1, x2, y2]
def calculate_iou(box1, box2):
    # Boundary determination
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = (area_of_box(box1) + area_of_box(box2) - intersection)

    return intersection / union if union > 0 else 0

# Argument box passed in the following format: [x1, y1, x2, y2]
def area_of_box(box):
    if len(box) != 4:
        raise ValueError("The coordinates of the rectangle must be in the format [x1, y1, x2, y2].")
    x1, y1, x2, y2 = box

    width = max(0, x2 - x1)
    height = max(0, y2 - y1)

    return width * height

# TODO It has to be tested on /test and /val directory
def generate_roi_from_images_dataset(images_dir, roi_dir):
    # with open(r"/data/images/train/_annotations.coco.json", 'r', encoding='utf-8') as file:
    ann_path = images_dir / "_annotations.coco.json"
    with open(ann_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    categories = data['categories']
    categories_iter = []
    for cat in categories[1:]:
        categories_iter.append(cat['name'])
    images = data['images']
    annotations = data['annotations']

    # Iteration through categories.
    for i, cat in enumerate(categories_iter):
        category_folder = roi_dir / cat
        os.makedirs(category_folder, exist_ok=True)

        # Iteration through images
        for image_info in images:
            ann = []
            for a in annotations:
                if a['image_id'] == image_info['id']:
                    ann.append(a['bbox'] + [a['category_id']])

            # Conversion from coordinates in float format to int
            ground_truth_boxes = [list(map(int, bbox)) for bbox in ann]

            # Every row of ann data structure has following format [x, y, w, h, category_number], so if we take
            # first row and its last element it will give us the category number
            if i == ann[0][4]-1:
                # Read the image
                img = Image.open(images_dir / str(image_info['file_name']))

                # Convert to format accepted by CV2
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

                # Generate Rois for individual images using Selective Search
                ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
                ss.setBaseImage(img)
                ss.switchToSelectiveSearchFast()
                rois = ss.process()

                print(f"Class:{i+1} Nr.Img:{image_info['id']} {len(rois)} proposals of region.")
                # output_image = img.copy()
                # output_path = os.path.join(roi_dir, f"imgId{image_info['id']}_{i}.jpg")
                # s = cv2.imwrite(output_path, output_image)

                for j, (x, y, w, h) in enumerate(rois):
                    max_iou = 0
                    best_label = 0

                    for gt_box in ground_truth_boxes:
                        gt_box_lbl = gt_box[4]          # gt_box[4] contains a number describing the category of the object
                        gt_box = [gt_box[0], gt_box[1], gt_box[0] + gt_box[2], gt_box[1] + gt_box[3]]   # [x1, y1, x2, y2]
                        temp_roi = [x, y, x+w, y+h]                                                     # [x1, y1, x2, y2]

                        iou = calculate_iou(temp_roi, gt_box)  # Calculate IOU
                        if iou > max_iou:
                            max_iou = iou
                            best_label = gt_box_lbl

                    # If IOU is greater than 40%, it means that on the image there is a part containing interesting object
                    # of the class
                    if max_iou > 0.4:
                        # save_positive_example(roi, best_label)
                        # cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cropped_region = img[y:y + h, x:x + w]
                        cropped_rgb = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2RGB)
                        img_pil = Image.fromarray(cropped_rgb)
                        # Images are saved in the following format "img_nrOfImage_x1_y1_x2_y2_nrOfCategory"
                        output_path = category_folder / f"img{image_info['id']}_{j}_x{x}_y{y}_{x + w}_{y + h}_{best_label}.jpg"
                        img_pil.save(output_path)
                        # s = cv2.imwrite(output_path, cropped_region)      # Problem with polish characters in path. That's
                                                                          # why it has to be transformed to pillow format


                    elif max_iou < 0.05:
                        if j < 100:
                            # save_negative_example(roi, 'background')
                            cropped_region = img[y:y + h, x:x + w]
                            cropped_rgb = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2RGB)
                            img_pil = Image.fromarray(cropped_rgb)
                            # Images are saved in the following format "img_nrOfImage_x1_y1_x2_y2_nrOfCategory"
                            # For the background system assumes that number of category is equal 0
                            output_path = category_folder / f"img{image_info['id']}_{j}_x{x}_y{y}_{x + w}_{y + h}_0.jpg"
                            img_pil.save(output_path)
                            # s = cv2.imwrite(output_path, cropped_region)      # Problem with polish characters in path. That's
                                                                             # why it has to be transformed to pillow format

# Counting of images which represent class object and ones which represent background
def images_with_object_and_background_counter(folder_path, which_class = 1):
    # folder_path = f"data/roi_data/train/{categories_iter[which_class-1]}"
    if not folder_path.exists():
        print("Folder doesn't exist.", file=sys.stderr)
        exit(1)

    pattern_class = re.compile(rf"img\d+_\d+_x\d+_y\d+_\d+_\d+_{which_class}\.jpg")
    pattern_bkgnd = re.compile(r"img\d+_\d+_x\d+_y\d+_\d+_\d+_0\.jpg")

    count_class = 0
    count_bkgnd = 0

    for subfolder in folder_path.iterdir():
        if not subfolder.is_dir():
            continue

        for file_name in os.listdir(subfolder):
            if pattern_class.match(file_name):
                count_class += 1
            elif pattern_bkgnd.match(file_name):
                count_bkgnd += 1

    print(f"Statistics for folder {folder_path.parent.name}/{folder_path.name}")
    print(f"Number of files matching the class pattern: {count_class}")
    print(f"Number of files matching the background pattern: {count_bkgnd}")

def split_samples_to_cls_and_bkgnd_folder(folder_path, which_class = 1):
    if not folder_path.exists():
        print("Folder doesn't exist.", file=sys.stderr)
        exit(1)

    pattern_class = re.compile(rf"img\d+_\d+_x\d+_y\d+_\d+_\d+_{which_class}\.jpg")
    pattern_bkgnd = re.compile(r"img\d+_\d+_x\d+_y\d+_\d+_\d+_0\.jpg")

    dest_bkgnd = folder_path / "0"
    dest_class_dir = folder_path / str(which_class)

    dest_bkgnd.mkdir(exist_ok=True)
    dest_class_dir.mkdir(exist_ok=True)

    moved_class = 0
    moved_bkgnd = 0
    skipped = 0

    for file_name in os.listdir(folder_path):
        src = folder_path / file_name
        if not src.is_file():
            continue

        if pattern_class.match(file_name):
            dst = dest_class_dir / file_name
            # If the file already exists in the destination, we do not overwrite it.
            if dst.exists():
                skipped += 1
                continue
            shutil.move(str(src), str(dst))
            moved_class += 1
            print(str(dst.name), "was moved.")

        elif pattern_bkgnd.match(file_name):
            dst = dest_bkgnd / file_name
            if dst.exists():
                skipped += 1
                continue
            shutil.move(str(src), str(dst))
            moved_bkgnd += 1
            print(str(dst.name), "was moved.")


parent_path = Path(__file__).resolve().parent.parent.parent
images_dir = parent_path / "data" / "images" / "train"
roi_dir = parent_path / "data" / "roi_data" / "train"

# generate_roi_from_images_dataset(images_dir, roi_dir)
images_with_object_and_background_counter(roi_dir / "Orange", which_class=4)
# split_samples_to_cls_and_bkgnd_folder(roi_dir / "Banana", which_class=2)