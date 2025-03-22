import json
import os
import numpy as np
from PIL import Image
import cv2
from pathlib import Path

# To get to data/images/train we have to go one level higher
parent_path = Path(__file__).resolve().parent.parent.parent
file_path = parent_path/'data'/'images'/'train'/'_annotations.coco.json'
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

categories = data['categories']
images = data['images']
annotations = data['annotations']
#
# # Data exploration
# for image_info in images[0:5]:
#     print(image_info)
# # {'id': 0, 'license': 1, 'file_name': 'fa41bc94efed37b6_jpg.rf.c0a1f29aace8ef3d7a282afcfefcaa17.jpg', 'height': 1024, 'width': 768, 'date_captured': '2022-11-29T07:28:32+00:00'}
#
# for categories_info in categories[0:5]:
#     print(categories_info)
# # {'id': 1, 'name': 'Apple', 'supercategory': 'Fruits'}
#
# for annotations_info in annotations[0:5]:
#     print(annotations_info)
# # {'id': 4, 'image_id': 1, 'category_id': 1, 'bbox': [259, 783, 63.36, 60.8], 'area': 3852.288, 'segmentation': [], 'iscrowd': 0}
#
# # Function to draw bounding boxes
# def draw_bboxes(image, bboxes):
#     for bbox in bboxes:
#         bbox = bbox[0:4]
#         bbox = [int(x) for x in bbox]
#         x, y, w, h = bbox
#         image = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
#     return image
#
#
# # Displaying the first few images with the rectangle marked according to the bbox annotation
# for image_info in images[0:10]:
#
#     # Image read
#     file_path = parent_path/'data'/'images'/'train'/str(image_info['file_name'])
#     img = Image.open(file_path)
#
#     # Converted to a format acceptable by cv2
#     img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#
#     ann = []
#     for a in data['annotations']:
#         if a['image_id'] == image_info['id']:
#             ann.append(a['bbox'] + [a['category_id']])
#
#     image_with_bboxes = draw_bboxes(img, ann)
#
#     # print(categories[int(annotations[0][4])-1])
#     cv2.imshow(f"{categories[ann[0][4]]}", image_with_bboxes)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

file_path = parent_path/'data'/'images'/'train'/str(images[0]['file_name'])
img = Image.open(file_path)
img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
cv2.imwrite("Test.jpg", img)

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(img)
ss.switchToSelectiveSearchFast()
rects = ss.process()

print(f"Found {len(rects)} region proposals.")

output_dir = "cut_regions"  # Folder for saved regions
os.makedirs(output_dir, exist_ok=True)  # Create folder if it doesn't exist

output_image = img.copy()
# cv2.imwrite("Test.jpg", output_image)

for i, (x, y, w, h) in enumerate(rects[:100]):
    # cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cropped_region = img[y:y + h, x:x + w]
    output_path = os.path.join(output_dir, f"region_{i}.jpg")
    cv2.imwrite(output_path, cropped_region)

