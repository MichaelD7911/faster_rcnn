import torch
from torch.utils.data import DataLoader
from utils import get_faster_rcnn_model
from utils import calculate_map
from utils import CocoDataset




device = 'cpu'
NUM_CLASSES = 6
model, preprocess = get_faster_rcnn_model(num_classes=NUM_CLASSES)


model.to(device)


print ('='*80)
print (model.transform)
print ('='*80)

mpath = "/home/michael/sardet100k/faster_rcnn/checkpoints/best_model_1.pth"
model.load_state_dict(torch.load(mpath))
model.to(device)
# model = torch.load(mpath, map_location=torch.device('cpu')) 
validation_dataset = CocoDataset(
    root="/home/michael/sardet100k/dataset/val",
    annFile="/home/michael/sardet100k/dataset/Annotations_corrected/val.json",
    transform=preprocess,
)


def collate(batch):
    """return tuple data"""
    return tuple(zip(*batch))


# val_loader = DataLoader(validation_dataset, batch_size=4, shuffle=False, num_workers=0)
val_loader = torch.utils.data.DataLoader(
    validation_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0,
    # prefetch_factor=4,
    pin_memory=True,
    collate_fn=collate,
)

predictions = []
targets = []
model.eval()

with torch.no_grad():
    for images, labels in val_loader:  # You can ignore targets during inference
        images = [img.to(device) for img in images]  # Move images to the GPU/CPU
        predictions = model(images)  # Get predictions
        predictions.extend(predictions)
        targets.extend(labels)
print ('finished')
# # Run evaluation and print mAP

mAP = calculate_map(predictions, targets)

print(f"Validation mAP: {mAP:.4f}")
