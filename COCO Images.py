from pycocotools.coco import COCO
import requests
import os

# Set paths and COCO API URL
dataType = 'train2017'
annFile = f'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'

# Initialize COCO api for instance annotations
coco = COCO(annFile)

# Get all image ids
imgIds = coco.getImgIds()
imgIds = imgIds[:3000]  # Select first 3000 images

# Set destination directory
destDir = 'coco_dataset/images'

# Function to download images
def download_image(url, file_path):
    try:
        img_data = requests.get(url).content
        with open(file_path, 'wb') as handler:
            handler.write(img_data)
        print(f'Successfully downloaded {file_path}')
    except Exception as e:
        print(f'Failed to download {file_path}: {e}')

# Download images
for img_id in imgIds:
    img = coco.loadImgs(img_id)[0]
    img_url = img['coco_url']
    img_name = os.path.join(destDir, img['file_name'])
    download_image(img_url, img_name)

