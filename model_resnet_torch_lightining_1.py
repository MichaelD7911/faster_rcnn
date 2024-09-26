import json
from pathlib import Path
from collections import defaultdict
from lightning.pytorch.callbacks import TQDMProgressBar
from PIL import Image
import torch
from torchvision.models.detection.faster_rcnn import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)
from torch import tensor, as_tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader 
from torch.optim import SGD
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection import MeanAveragePrecision
import lightning as L






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
            "boxes": as_tensor(anno_data["boxes"], dtype=torch.float32),
            "labels": as_tensor(anno_data["labels"], dtype=torch.int64),
            "image_id": tensor([image_id]),
            "area": as_tensor(anno_data["area"], dtype=torch.float32),
            "iscrowd": as_tensor(anno_data["iscrowd"], dtype=torch.int64),
        }


        # if self.transform:
        #     input_data = self.transform(input_data)

        if self.transform is not None:
            image = self.transform(image)

        return image, target
    

class FasterRCNN_ResNet50_Lightning(L.LightningModule):

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     transforms.Resize(size=(800,), max_size=1333),])
      
    def __init__(self):
        super().__init__()
        self.model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        num_classes = 6
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.learning_rate = 1e-3
        self.train_map = MeanAveragePrecision()
        self.val_map = MeanAveragePrecision()

    def forward(self, x):
        return self.model(x)
    

    
    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())
        
        # Log training loss
        # self.log('train_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('train_loss', total_loss, on_step=False, on_epoch=True)
        # self.log('train_loss', total_loss, on_step=False, on_epoch=True)
        # # self.log('loss_step', loss, on_step=True, on_epoch=False)

        # self.model.eval()
        # if batch_idx % 10 == 0:
        #     self.logger.history['loss_step']
            
  
        return total_loss

    # def on_training_epoch_end(self):
    #     # Log mAP for training set
    #     train_map_value = self.train_map.compute()['map_50']
    #     self.log('train_map', train_map_value, on_epoch=True, prog_bar=True)
    #     self.train_map.reset()

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        predictions = self.model(images)
                
        self.val_map.update(predictions, targets)
        self.log({'val_map': self.val_map.compute()['map_50']}, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_map.reset()


    # def on_validation_epoch_end(self):
    #     # Log mAP for validation set
    #     val_map_value = self.val_map.compute()['map_50']
    #     self.log('val_map', val_map_value, on_epoch=True, prog_bar=True, logger=True)
    #     # self.log_dict({'validation_loss': loss, 'validation_accuracy': acc}, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #     self.val_map.reset()


    # def predict_step(self, batch, batch_idx, dataloader_idx=0):
    #     x, y = batch
    #     y_hat = self.model(x)
    #     return y_hat
    
    # def configure_optimizers(self):
    #     optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.95, weight_decay=1e-5, nesterov=True)
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6, eta_min=0, verbose=True)
    #     return [optimizer], [scheduler]
    
    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        return SGD(params, lr=0.02)


preprocess = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.transforms()

def collate_fn(batch):
    """Define a collate function to handle batches."""
    return tuple(zip(*batch))


class CocoLightningDataModule(L.LightningDataModule):

    def __init__(self, batch_size, num_workers):
        super().__init__()

        self.save_hyperparameters()
        # self.csv_path = csv_path
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    @staticmethod
    def _collate_fn(batch):
        """Define a collate function to handle batches."""
        return tuple(zip(*batch))

    
    def train_dataloader(self):
        train_dataset = CocoDataset(root="/home/michael/sardet100k/dataset/val",
                                    annFile="/home/michael/sardet100k/dataset/Annotations_corrected/sample50.json",
                                    transform=preprocess)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, 
                          num_workers=self.num_workers, collate_fn=self._collate_fn)

    def train_dataloader(self):
        val_dataset = CocoDataset(root="/home/michael/sardet100k/dataset/val",
                                  annFile="/home/michael/sardet100k/dataset/Annotations_corrected/sample50.json",
                                  transform=preprocess)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, 
                          num_workers=self.num_workers, collate_fn=self._collate_fn)


def train_faster_rcnn():
    # Initialize dataset and dataloaders
    data = CocoLightningDataModule(batch_size=2, num_workers=4)
    model = FasterRCNN_ResNet50_Lightning()
    trainer = L.Trainer(max_epochs=2, accelerator='gpu',
                        num_nodes=1, log_every_n_steps=2)
    
    # trainer = L.Trainer(max_epochs=10, accelerator='gpu',
    #                     num_nodes=1, 
    #                     callbacks=[TQDMProgressBar(refresh_rate=2)])
    
    trainer.fit(model, data)


if __name__ == '__main__':
    train_faster_rcnn()


 



