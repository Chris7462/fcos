import os
import time
import torch

import multiprocessing

from a4_helper import VOC2007DetectionTiny

from a4_helper import train_detector
from a4_helper import inference_with_detector
from one_stage_detector import FCOS


one_stage_detector_path = os.path.join("./", "one_stage_detector.py")
one_stage_detector_edit_time = time.ctime(
    os.path.getmtime(one_stage_detector_path)
)

DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {DEVICE} device")

# Set a few constants related to data loading.
NUM_CLASSES = 20
BATCH_SIZE = 16
IMAGE_SHAPE = (224, 224)
NUM_WORKERS = multiprocessing.cpu_count()

# NOTE: Set `download=True` for the first time when you set up Google Drive folder.
# Turn it back to `False` later for faster execution in the future.
# If this hangs, download and place data in your drive manually as shown above.
train_dataset = VOC2007DetectionTiny(
    "./", "train", image_size=IMAGE_SHAPE[0],
    download=False  # True (for the first time)
)
val_dataset = VOC2007DetectionTiny("./", "val", image_size=IMAGE_SHAPE[0])

# `pin_memory` speeds up CPU-GPU batch transfer, `num_workers=NUM_WORKERS` loads data
# on the main CPU process, suitable for Colab.
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, pin_memory=True, num_workers=NUM_WORKERS
)

# Use batch_size = 1 during inference - during inference we do not center crop
# the image to detect all objects, hence they may be of different size. It is
# easier and less redundant to use batch_size=1 rather than zero-padding images.
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, pin_memory=True, num_workers=NUM_WORKERS
)

# Slightly larger detector than in above cell.
detector = FCOS(
    num_classes=NUM_CLASSES,
    fpn_channels=128,
    stem_channels=[128, 128],
)
detector = detector.to(DEVICE)

train_detector(
    detector,
    train_loader,
    learning_rate=8e-3,
    max_iters=9000,
    log_period=100,
    device=DEVICE,
)

# After you've trained your model, save the weights for submission.
#weights_path = os.path.join(GOOGLE_DRIVE_PATH, "fcos_detector.pt")
weights_path = os.path.join("./", "fcos_detector.pt")
torch.save(detector.state_dict(), weights_path)



weights_path = os.path.join("./", "fcos_detector.pt")

# Re-initialize so this cell is independent from prior cells.
detector = FCOS(
    num_classes=NUM_CLASSES, fpn_channels=128, stem_channels=[128, 128]
)
detector.to(device=DEVICE)
detector.load_state_dict(torch.load(weights_path, weights_only=True, map_location="cpu"))

inference_with_detector(
    detector,
    val_loader,
    val_dataset.idx_to_class,
    score_thresh=0.4,
    nms_thresh=0.6,
    device=DEVICE,
    dtype=torch.float32,
    output_dir="./mAP/input",
)
