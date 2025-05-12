import torch
from torchvision.datasets import VOCDetection
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms.v2 import ToImageTensor
import matplotlib.pyplot as plt
import os

# Download VOC data (default: 2007 trainval split)
dataset = VOCDetection(
    root='voc_data',
    year='2007',
    image_set='train',
    download=True,
    transform=lambda img: pil_to_tensor(img),  # Convert PIL image to tensor
)

# Access a sample
img, target = dataset[0]

# Print structure of target annotation
print("Image shape:", img.shape)
print("Target keys:", target.keys())
print("Annotation example:", target['annotation']['object'])

# Display image and draw one bounding box
from PIL import ImageDraw

def show_img_with_bbox(img, target):
    img = img.permute(1, 2, 0).numpy()
    objects = target['annotation']['object']
    if not isinstance(objects, list):
        objects = [objects]

    img_pil = Image.fromarray((img * 255).astype('uint8'))
    draw = ImageDraw.Draw(img_pil)

    for obj in objects:
        bbox = obj['bndbox']
        box = [
            int(bbox['xmin']),
            int(bbox['ymin']),
            int(bbox['xmax']),
            int(bbox['ymax']),
        ]
        draw.rectangle(box, outline='red', width=2)
        draw.text((box[0], box[1]), obj['name'], fill='red')

    img_pil.show()

show_img_with_bbox(img / 255.0, target)  # Normalize for display
