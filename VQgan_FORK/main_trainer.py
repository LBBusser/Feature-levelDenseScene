import os
import time
import torch
import copy
from omegaconf import OmegaConf
import argparse
from torch import optim
from torch.nn import functional as F
from tqdm import tqdm
from transformers import get_constant_schedule_with_warmup
from main_inference import *
import lightning as L
# from pytorch_fid import fid_score
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from autoregressive_model import *
from data_loader import *
from my_utils import *


class MainTrainer:
    def __init__(self, args, train_loader, val_loader,  test_loader):
        super(MainTrainer, self).__init__()
        self.num_epochs = args.num_epochs
        self.warm_up_steps = args.warm_up_steps
        self.lr = args.lr
        self.img_size = args.img_size
        self.patch_size = args.patch_size
        self.model = MetaTrainer(args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       

        self.params_with_grad = [param for name, param in self.model.named_parameters() if param.requires_grad]
        print(f"The model has {count_model_params(self.params_with_grad)} trainable parameters.")

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        ###############
        self.checkpoint_callback = ModelCheckpoint(monitor= 'val_loss')
        # self.checkpoint_callback = ModelCheckpoint(monitor= 'train_loss')
        ###############
        # self.early_stopping = EarlyStopping(args)
        # self.optimizer = optim.AdamW(params=self.params_with_grad, lr=self.lr)
        # self.scheduler = get_constant_schedule_with_warmup(self.optimizer, num_warmup_steps=self.warm_up_steps)
    
        
    def train(self):
        print(f"START TRAINING {args.trained_model_name}")
        #############
        lightning_trainer = L.Trainer(devices = 4, 
                                      accelerator = "gpu", max_epochs = self.num_epochs, accumulate_grad_batches = 4, num_nodes=1 ,callbacks=[self.checkpoint_callback], precision='bf16-mixed', strategy='ddp_find_unused_parameters_true')
        lightning_trainer.fit(self.model,self.train_loader, self.val_loader)
        #############
        return self.model

    def forward_batch(self, batch):
        x_spt, y_spt, x_qry, y_qry = batch
        x_spt, y_spt, x_qry, y_qry = x_spt.to(self.device), y_spt.to(self.device), x_qry.to(self.device), y_qry.to(self.device)
        loss = self.model(x_spt, y_spt, x_qry, y_qry)
        return loss


    def inference(self):
        all_predictions = []
        all_labels = []
        all_images = []
        # color_mapping = load_color_mapping('/home/lbusser/hbird_scripts/hbird_eval/data/color_map.json')
        # class_ids = list(color_mapping.values())
        # total_ious = []
        model = MetaTrainer.load_from_checkpoint('/home/lbusser/lightning_logs/version_6833430/checkpoints/epoch=36-step=92500.ckpt', args=args)
        ######
        trainer = L.Trainer(accelerator= 'gpu', devices=1)
        out = trainer.predict(model, self.test_loader)
        for result in out:
            all_predictions.append(result['pred'])
            all_labels.append(result['mask'])
            all_images.append(result['image'])
            
            
        # model = model.to(device)
        # print("Num. of tasks: {}".format(len(self.test_loader)))
        all_predictions = torch.cat(all_predictions, dim=0)
 
        all_labels = torch.cat(all_labels, dim=0)
        all_images = torch.cat(all_images,dim=0)
        # Normalize using global min and max
        all_predictions_normalized = (all_predictions - all_predictions.min()) / (all_predictions.max() - all_predictions.min())
        all_labels_normalized = (all_labels - all_labels.min()) / (all_labels.max() - all_labels.min())
        # for pred, label in zip(all_predictions_normalized, all_labels_normalized):
        #         quantized_pred = quantize_colors(pred, color_mapping)
        #         pred_label_mask = color_mask_to_label_mask(quantized_pred, color_mapping)
        #         label = label.squeeze(0).permute(1, 2, 0).numpy()
        #         true_label_mask = color_mask_to_label_mask(label, color_mapping)
        #         # print("Before quantized prediction:", np.unique(pred))
        #         # print("Quantized Prediction:", np.unique(quantized_pred))
        #         # print("Prediction Label Mask:", np.unique(pred_label_mask))
        #         # print("True Label Mask:", np.unique(true_label_mask))
        #         ious = calculate_iou(pred_label_mask, label, class_ids)
              
        #         total_ious.append(ious)
        total_mae = F.l1_loss(all_predictions_normalized, all_labels_normalized, reduction= 'mean').item()
        print(f"Average Mean Absolute Error (MAE): {total_mae:.4f}")
        # print(all_predictions_normalized.max())
        # print(all_labels_normalized.max())

        # total_ious = np.array(total_ious)
        # mean_iou_per_class = np.nanmean(total_ious, axis=0)
        # mean_iou = np.nanmean(mean_iou_per_class)
        # print(f"Mean Intersection over Union (mIoU): {mean_iou:.4f}")
      
        print("Saving first few test entries for visualisation...")
        torch.save(all_labels[0:20], '/home/lbusser/hbird_scripts/hbird_eval/data/models/predictions/' + f"{args.trained_model_name}_gts.pt")
        torch.save(all_images[0:20], '/home/lbusser/hbird_scripts/hbird_eval/data/models/predictions/' + f"{args.trained_model_name}_imgs.pt")
        torch.save(all_predictions[0:20], '/home/lbusser/hbird_scripts/hbird_eval/data/models/predictions/' + f"{args.trained_model_name}_preds.pt")
        # miou_score = miou_calculator.compute()
        # print("Mean IoU:", miou_score)
        # # accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
      
        # print("Mean MAE:", sum(mae_scores)/len(mae_scores)


    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease """
        print("Validation loss decreased ({:.4f}  --> {:.4f}). Saving model {} ..."
              .format(self.val_loss_min, val_loss, self.model_name))
        torch.save(model.state_dict(), os.path.join("/home/lbusser/hbird_scripts/hbird_eval/data", "models", self.model_name+".pt"))
        self.val_loss_min = val_loss


def load_color_mapping(json_file):
    with open(json_file, 'r') as f:
        color_mapping = json.load(f)
    # Convert string keys to integers
    return {tuple(map(int, v)): int(k) for k, v in color_mapping.items()}

def quantize_colors(reconstructed_color, color_mapping):
    # Transpose the color to have shape [224, 224, 3]
    if reconstructed_color.shape[0] == 3:
        reconstructed_color = reconstructed_color.permute(1, 2, 0).numpy()
    
    quantized_mask = np.zeros((reconstructed_color.shape[0], reconstructed_color.shape[1], 3), dtype=np.uint8)
    
    for i in range(reconstructed_color.shape[0]):
        for j in range(reconstructed_color.shape[1]):
            pixel = reconstructed_color[i, j]
            if isinstance(pixel, torch.Tensor):
                pixel = pixel.numpy()
            print(pixel)
            closest_color = min(color_mapping.keys(), key=lambda color: np.linalg.norm(np.array(color) - pixel))
            print(closest_color)
            quantized_mask[i, j, :] = closest_color
    return quantized_mask

def color_mask_to_label_mask(color_mask, color_mapping):
    label_mask = np.zeros((color_mask.shape[0], color_mask.shape[1]), dtype=np.int32)
    for color, label in color_mapping.items():
        color_mask_bool = np.all(color_mask == np.array(color), axis=-1)
        label_mask[color_mask_bool] = label
    return label_mask

def calculate_iou(pred_mask, true_mask, class_ids):
    ious = []

    for cls in class_ids:
        pred_cls = (pred_mask == cls)
        true_cls = (true_mask == cls)
        
        intersection = np.logical_and(pred_cls, true_cls).sum()
        union = np.logical_or(pred_cls, true_cls).sum()
        
        if union == 0:
            ious.append(np.nan)
        else:
            iou = intersection / union
            ious.append(iou)
    return ious

def count_model_params(model_parameters):
    params = list(filter(lambda p: p.requires_grad, model_parameters))
    params_summed = sum(p.numel() for p in params)
    return params_summed

class CheckParameterDeviceCallback(L.Callback):
    def on_train_start(self, trainer, pl_module):
        for name, param in pl_module.named_parameters():
            print(f"{name} is on {param.device}")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=3)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=1)
    argparser.add_argument('--testk_spt', type=int, help='k shot for test support set', default=3)
    argparser.add_argument('--testk_qry', type=int, help='k shot for test query set', default=1)
    argparser.add_argument('--img_size', type=int, help='img_size', default=224)
    argparser.add_argument('--num_workers', type=int, default=8)
    argparser.add_argument('--data_size', type=int, default=10000)
    argparser.add_argument('--test_size', type=int, default=100)
    argparser.add_argument('--val_data_size', type=int, default=100)

    # COCO training / non-episodic image captioning
    argparser.add_argument('--num_epochs', type=int, help='epoch number for training', default=10)
    argparser.add_argument('--batch_size', type=int, help='batch size for few shot training', default=4)
    argparser.add_argument('--test_batch_size', type=int, help='batch size for few shot testing', default=4)
    argparser.add_argument('--coco_annotations_path', type=str, default='/scratch-shared/combined_hbird/mscoco_hbird/annotations/')
    argparser.add_argument('--early_stop_patience', type=int, help='#epochs w/o improvement', default=3)
    argparser.add_argument('--delta', type=float, help='min change in the monitored val loss', default=0.01)
    argparser.add_argument('--lr', type=float, help='LR for training', default=5e-04)
    argparser.add_argument('--warm_up_steps', type=int, help='warm up steps', default=500)
    argparser.add_argument('--trained_model_name', type=str, default='default')
    argparser.add_argument('--patch_size', type=int, default=14)
    args = argparser.parse_args()
    
   
    min_scale_factor = 0.5
    max_scale_factor = 2.0
    brightness_jitter_range = 0.1
    contrast_jitter_range = 0.1
    saturation_jitter_range = 0.1
    hue_jitter_range = 0.1
    brightness_jitter_probability = 0.5
    contrast_jitter_probability = 0.5
    saturation_jitter_probability = 0.5
    hue_jitter_probability = 0.5
    input_size = args.img_size
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    #----------------------------TRAIN---------------------------------------
    image_train_transform = trn.Compose([
            trn.RandomApply([trn.ColorJitter(brightness=brightness_jitter_range)], p=brightness_jitter_probability),
            trn.RandomApply([trn.ColorJitter(contrast=contrast_jitter_range)], p=contrast_jitter_probability),
            trn.RandomApply([trn.ColorJitter(saturation=saturation_jitter_range)], p=saturation_jitter_probability),
            trn.RandomApply([trn.ColorJitter(hue=hue_jitter_probability)], p=hue_jitter_probability),
            trn.ToTensor(),
            trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255]),
        ])
    shared_train_transform = Compose([
            Resize(size=(input_size, input_size)),
            # RandomHorizontalFlip(probability=0.1),
        ])
    
    image_val_transform = trn.Compose([ trn.ToTensor(), trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])])
    shared_val_transform = Compose([
            Resize(size=(input_size, input_size)),
        ])
    
    # cluster_index = faiss.read_index("/home/lbusser/hbird_scripts/hbird_eval/data/MSCOCO_dinov2_vitb14_1500_cluster_results/train/cluster_index.index")
    # image_ids = sorted(os.listdir("/scratch-shared/combined_hbird/mscoco_hbird/train2017/"))
    # train_idx, val_idx = train_test_split(np.arange(len(image_ids)), test_size=0.2, random_state=0)
    # train_set_COCO = CocoMemoryTasksDataLoader(data_path="/scratch-shared/combined_hbird/mscoco_hbird", mode = 'train',  setsz=args.data_size, k_shot=args.k_spt, k_query=args.k_qry, resize=input_size, cluster_index= cluster_index, cluster_assignment= '/home/lbusser/hbird_scripts/hbird_eval/data/MSCOCO_dinov2_vitb14_1500_cluster_results/train/cluster_assignments.pkl', transforms = (image_train_transform, shared_train_transform), mode_idx=train_idx)
    # val_set_COCO = CocoMemoryTasksDataLoader(data_path="/scratch-shared/combined_hbird/mscoco_hbird", mode = 'train', setsz=args.val_data_size, k_shot=args.k_spt, k_query=args.k_qry, resize=input_size, cluster_index= cluster_index,cluster_assignment= '/home/lbusser/hbird_scripts/hbird_eval/data/MSCOCO_dinov2_vitb14_1500_cluster_results/train/cluster_assignments.pkl', transforms = (image_val_transform, shared_val_transform), mode_idx=val_idx)
    # # train_set_KP = KeyPointMemoryTasksDataLoader(data_path="/scratch-shared/combined_hbird/mscoco_hbird", mode="train", setsz=args.data_size, k_shot=args.k_spt, k_query=args.k_qry, resize=input_size, cluster_index = cluster_index, cluster_assignment ='/home/lbusser/hbird_scripts/hbird_eval/data/MSCOCO_dinov2_vitb14_1500_cluster_results/train/cluster_assignments.pkl',  transforms = (image_train_transform, shared_train_transform), mode_idx = train_idx)
    # # val_set_KP = KeyPointMemoryTasksDataLoader(data_path="/scratch-shared/combined_hbird/mscoco_hbird", mode="train", setsz=args.val_data_size, k_shot=args.k_spt, k_query=args.k_qry, resize=input_size, cluster_index = cluster_index, cluster_assignment ='/home/lbusser/hbird_scripts/hbird_eval/data/MSCOCO_dinov2_vitb14_1500_cluster_results/train/cluster_assignments.pkl',  transforms = (image_train_transform, shared_train_transform), mode_idx= val_idx)
    # cluster_index_nyu = faiss.read_index("/home/lbusser/hbird_scripts/hbird_eval/data/NYUv2_dinov2_vitb14_500_cluster_results/cluster_index.index")
    # cluster_assignment = '/home/lbusser/hbird_scripts/hbird_eval/data/NYUv2_dinov2_vitb14_500_cluster_results/cluster_assignments.pkl'
    # image_paths = pd.read_csv("/scratch-shared/combined_hbird/nyu_hbird/nyu_data/data/nyu2_train.csv")
    # train_idx, val_idx = train_test_split(np.arange(len(image_paths)), test_size=0.2, random_state = 0)
    # train_set_NYU = NYUMemoryTasksDataLoader(data_path="/scratch-shared/combined_hbird/nyu_hbird/nyu_data/data/", mode="train", setsz=args.data_size, k_shot=args.k_spt, k_query=args.k_qry, resize=input_size, cluster_index= cluster_index_nyu, cluster_assignment= cluster_assignment,  transforms = (image_train_transform, shared_train_transform), mode_idx = train_idx)
    # val_set_NYU = NYUMemoryTasksDataLoader(data_path="/scratch-shared/combined_hbird/nyu_hbird/nyu_data/data/", mode="train", setsz=args.val_data_size, k_shot=args.k_spt, k_query=args.k_qry, resize= input_size, cluster_index= cluster_index_nyu, cluster_assignment= cluster_assignment,  transforms = (image_train_transform, shared_train_transform),mode_idx = val_idx)
    
    # train_set = CombinedDataset([train_set_COCO, train_set_NYU], [3,1])
    # val_set = CombinedDataset([val_set_COCO, val_set_NYU],[3,1])
    # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False,
    #                           num_workers=args.num_workers, pin_memory=True, drop_last=True)
    # val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
    #                         num_workers=args.num_workers, pin_memory=True, drop_last=True)
    #------------------------------------------TEST------------------------------------
    test_coco_cluster_index = faiss.read_index('/home/lbusser/hbird_scripts/hbird_eval/data/MSCOCO_dinov2_vitb14_100_cluster_results/cluster_index.index')
    test_nyu_cluster_index = faiss.read_index('/home/lbusser/hbird_scripts/hbird_eval/data/NYUv2_dinov2_vitb14_100_cluster_results/cluster_index.index')
    
    image_val_transform = trn.Compose([ trn.ToTensor(), trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])])
    shared_val_transform = Compose([
                Resize(size=(input_size, input_size)),
            ]) 
    panoptic = True
    coco_test_set = CocoMemoryTasksDataLoader("/scratch-shared/combined_hbird/mscoco_hbird",'val', args.test_size, args.testk_spt, args.testk_qry, args.img_size, cluster_index= test_coco_cluster_index, cluster_assignment= '/home/lbusser/hbird_scripts/hbird_eval/data/MSCOCO_dinov2_vitb14_100_cluster_results/cluster_assignments.pkl', transforms = (image_val_transform, shared_val_transform), panoptic=panoptic)
    nyu_test_set = NYUMemoryTasksDataLoader("/scratch-shared/combined_hbird/nyu_hbird/nyu_data/data/", 'test', args.test_size, args.testk_spt, args.testk_qry, args.img_size, cluster_index= test_nyu_cluster_index, cluster_assignment='/home/lbusser/hbird_scripts/hbird_eval/data/NYUv2_dinov2_vitb14_100_cluster_results/cluster_assignments.pkl', transforms = (image_val_transform, shared_val_transform))
    test_set = CombinedDataset([coco_test_set], [1])

    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    #------------------------initalise model and train--------------------------------
    trainer = MainTrainer(args, None, None, test_loader)
    # trainer.train()
    print("-------------------------------------------------------INFERENCE-------------------------------------------------------")
    trainer.inference()
    print(f"DONE TESTING {args.trained_model_name}")
