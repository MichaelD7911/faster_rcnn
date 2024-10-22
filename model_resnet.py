import random
import json
import datetime
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision.ops import box_iou
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
    # FasterRCNN_ResNet50_FPN_V2_Weights
)
from torchmetrics.detection import MeanAveragePrecision
# import wandb




# wandb.init(project="Fast RCNN pure pytorch")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


NUM_CLASSES = 6  # background=0 included, Suzanne = 1

def get_faster_rcnn_model(num_classes):
    """return model and preprocessing transform"""
    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    model.roi_heads.box_predictor.cls_score = torch.nn.Linear(
        in_features=model.roi_heads.box_predictor.cls_score.in_features,
        out_features=num_classes,
        bias=True,
    )
    model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(
        in_features=model.roi_heads.box_predictor.bbox_pred.in_features,
        out_features=num_classes * 4,
        bias=True,
    )
    preprocess = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.transforms()
    return model, preprocess


model, preprocess = get_faster_rcnn_model(num_classes=NUM_CLASSES)
model.to(device)


print(model.transform)

class CocoDataset(Dataset):
    """PyTorch dataset for COCO annotations."""

    # adapted from https://github.com/pytorch/vision/issues/2720

    def __init__(self, root, annFile, transform=None):
        """Load COCO annotation data."""
        self.data_dir = Path(root)
        self.transform = transform

        # load the COCO annotations json
        anno_file_path = annFile
        with open(str(anno_file_path)) as file_obj:
            self.coco_data = json.load(file_obj)
        # put all of the annos into a dict where keys are image IDs to speed up retrieval
        self.image_id_to_annos = defaultdict(list)
        for anno in self.coco_data["annotations"]:
            image_id = anno["image_id"]
            self.image_id_to_annos[image_id] += [anno]

    def __len__(self):
        return len(self.coco_data["images"])

    def __getitem__(self, index):
        """Return tuple of image and labels as torch tensors."""
        image_data = self.coco_data["images"][index]
        image_id = image_data["id"]
        image_path = self.data_dir / image_data["file_name"]
        image = Image.open(image_path)

        annos = self.image_id_to_annos[image_id]
        anno_data = {
            "boxes": [],
            "labels": [],
            "area": [],
            "iscrowd": [],
        }
        for anno in annos:
            coco_bbox = anno["bbox"]
            left = coco_bbox[0]
            top = coco_bbox[1]
            right = coco_bbox[0] + coco_bbox[2]
            bottom = coco_bbox[1] + coco_bbox[3]
            area = coco_bbox[2] * coco_bbox[3]
            anno_data["boxes"].append([left, top, right, bottom])
            anno_data["labels"].append(anno["category_id"])
            anno_data["area"].append(area)
            anno_data["iscrowd"].append(anno["iscrowd"])

        target = {
            "boxes": torch.as_tensor(anno_data["boxes"], dtype=torch.float32),
            "labels": torch.as_tensor(anno_data["labels"], dtype=torch.int64),
            "image_id": torch.tensor([image_id]),
            "area": torch.as_tensor(anno_data["area"], dtype=torch.float32),
            "iscrowd": torch.as_tensor(anno_data["iscrowd"], dtype=torch.int64),
        }


        # if self.transform:
        #     input_data = self.transform(input_data)

        if self.transform is not None:
            image = self.transform(image)

        return image, target
    


# create datasets
training_dataset = CocoDataset(
    root="/home/michael/sardet100k/dataset/val",
    annFile="/home/michael/sardet100k/dataset/Annotations_corrected/val.json",
    transform=preprocess,
)
validation_dataset = CocoDataset(
    root="/home/michael/sardet100k/dataset/val",
    annFile="/home/michael/sardet100k/dataset/Annotations_corrected/val.json",
    transform=preprocess,
)

print(f"training dataset size: {training_dataset.__len__()}")


print(f"validation dataset size: {validation_dataset.__len__()}")



# get a random training sample
# img, label = training_dataset[random.randint(0, len(training_dataset) - 1)]
# print(f"random training label: {label}")

# display image with bbox label
# transform = T.ToPILImage()
# img = transform(img)
# x1, y1, x2, y2 = label["boxes"].numpy()[0]
# draw = ImageDraw.Draw(img)
# draw.rectangle([x1, y1, x2, y2], fill=None, outline="#ff0000cc", width=2)
# # display(img)
# img.show()

BATCH_SIZE = 2

def collate(batch):
    """return tuple data"""
    return tuple(zip(*batch))

train_loader = torch.utils.data.DataLoader(
    training_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    # prefetch_factor=4,
    pin_memory=True,
    collate_fn=collate,
)

validation_loader = torch.utils.data.DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    # prefetch_factor=4,
    pin_memory=True,
    collate_fn=collate,
)

params = [p for p in model.parameters() if p.requires_grad]


optimizer = torch.optim.SGD(
    params, 
    lr=0.00001, 
    momentum=0.9, 
    weight_decay=0.0001
)


def calculate_map(predictions, targets, iou_threshold=0.5):
    APs = []
    for prediction, target in zip(predictions, targets):
        pred_boxes = prediction['boxes']
        target_boxes = target['boxes']

        if len(pred_boxes) == 0 or len(target_boxes) == 0:
            APs.append(0)
            continue

        iou_matrix = box_iou(pred_boxes, target_boxes)
        ious = iou_matrix.max(dim=1)[0]  # max IoU for each prediction
        true_positive = (ious >= iou_threshold).sum().item()
        
        # Precision: TP / (TP + FP), FP = len(pred_boxes) - TP
        if len(pred_boxes) > 0:
            precision = true_positive / len(pred_boxes)
        else:
            precision = 0
        
        APs.append(precision)
    
    return sum(APs) / len(APs) if APs else 0.0

num_epochs = 12
train_loss_list = []
validation_loss_list = []

model.train() # set model in training mode
for epoch in range(num_epochs):
    model.train()
    N = len(train_loader.dataset)
    current_train_loss = 0
    total_train_mAP_1 = 0
    torchmetrics_mAP = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
    # train loop
    for i, (images, targets) in enumerate(train_loader):
        # move data to device and build the right input format for our model
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} 
                   for t in targets]


        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        current_train_loss += losses

        model.eval()
        with torch.no_grad():
            predictions = model(images)
            batch_mAP_1 = calculate_map(predictions, targets)
            
            total_train_mAP_1 += batch_mAP_1
        model.train()
        if (i + 1) % 500 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {current_train_loss / (i+1):.4f}")
    train_loss_list.append(current_train_loss / N)
    train_mAP_1 = total_train_mAP_1 / (i+1)


    # validation loop
    model.train()
    best_val_map = 0
    total_val_mAP = 0
    N = len(validation_loader.dataset)
    current_validation_loss = 0
    with torch.no_grad():
        for images, targets in validation_loader:
            images = list(image.to(device) for image in images)
            targets = [
                {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()
                }
                for t in targets
            ]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            current_validation_loss += losses
    validation_loss_list.append(current_validation_loss / N)


    model.eval()
    for images, targets in validation_loader:
        with torch.no_grad():
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} 
                   for t in targets]

            predictions = model(images)  # Get predictions for validation
            batch_mAP = calculate_map(predictions, targets)  # mAP calculation function
            torchmetrics_mAP.update(predictions, targets)
            total_val_mAP += batch_mAP

    val_mAP = total_val_mAP / len(validation_loader)

    if val_mAP < best_val_map:
        print (f'Best model is epoch: {epoch+1}')
        best_val_map = val_mAP
        torch.save(model.state_dict(), f'best_model_{epoch+1}.pth')  # Save the best model


    # Print training and validation metrics
    print(f'Epoch [{epoch+1}/{num_epochs}]')
    print(f'Training | Validation Loss: {train_loss_list[-1]:.4f} <> {validation_loss_list[-1]:.4f}')
    print(f'Train | Validation mAP (IoU 0.5): {train_mAP_1:.4f} <> {torchmetrics_mAP.compute()["map_50"]:.4f} <> {val_mAP:.4f}')
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Current time: {current_time}")
    print("=" * 80)

train_loss = [x.cpu().detach().numpy() for x in train_loss_list]
validation_loss = [x.cpu().detach().numpy() for x in validation_loss_list]



plt.plot(train_loss, "-o", label="train loss")
plt.plot(validation_loss, "-o", label="validation loss")
plt.title(current_time)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.savefig("loss.png") 

