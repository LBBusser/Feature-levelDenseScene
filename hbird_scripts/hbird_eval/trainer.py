import logging
import os
import time

import torch
import argparse
from torch import optim
from models import FeatureExtractorBeta as FeatureExtractor
from torch.nn import functional as F
from tqdm import tqdm
from transformers import get_constant_schedule_with_warmup
from main_inference import *

from meta_learner import MetaLearner
from data_loader import *
from meta_trainer import MetaTrainer
from my_utils import *

LOG_PATH = "/home/lbusser/hbird_scripts/hbird_eval/data/logs_few_shot/"
class Trainer:
    def __init__(self, args, train_loader, val_loader, feature_extractor, device):
        super(Trainer, self).__init__()
        self.num_epochs = args.num_epochs
        self.warm_up_steps = args.warm_up_steps
        self.lr = args.lr
        self.img_size = args.img_size
        self.patch_size = args.patch_size
        self.device = device
        
        self.feature_extractor = feature_extractor
        self.model = MetaTrainer(args, self.feature_extractor)
        self.model.to(device)

        self.params_with_grad = [param for param in self.model.parameters() if param.requires_grad]
        print(f"The model has {count_model_params(self.params_with_grad)} trainable parameters.")

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.early_stopping = EarlyStopping(args)
        self.optimizer = optim.AdamW(params=self.params_with_grad, lr=self.lr)
        self.scheduler = get_constant_schedule_with_warmup(self.optimizer, num_warmup_steps=self.warm_up_steps)
        
    def train(self):
        train_losses, val_losses = [], []
        start_train = time.time()
        grad_clip = 0
        for epoch_id in range(0, self.num_epochs):
            print('------ START Epoch [{}/{}] ------'.format(epoch_id + 1, self.num_epochs))
            self.model.train()
            start = time.time()
            with tqdm(desc='Epoch [{}/{}] - training'.format(str(epoch_id + 1), self.num_epochs), unit='it',
                      total=len(self.train_loader), position=0, leave=True) as pbar:
                for batch_id, batch in enumerate(self.train_loader):
                    loss = self.forward_batch(batch)
                    self.optimizer.zero_grad()
                    
                    # Compute gradients only for parameters that require them
                    gradients = torch.autograd.grad(loss, self.params_with_grad, allow_unused=True)
                    # Apply the gradients to the parameters
                    for param, grad in zip(self.params_with_grad, gradients):
                        if grad is not None:
                            param.grad = grad
                    # Gradient clipping
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_value_(self.params_with_grad, clip_value=grad_clip)
                    self.optimizer.step()
                    self.scheduler.step()
                    train_losses.append(loss.item())
                    pbar.set_postfix(loss=loss.item())
                    pbar.update()

                with torch.no_grad():
                    self.model.eval()
                    for batch_id, batch in enumerate(self.val_loader):
                        print("start validation")
                        val_loss = self.forward_batch(batch)
                        val_losses.append(val_loss.item())

            end = time.time()
            train_log = "Training: Epoch [{}/{}], Mean Epoch Loss: {:.4f}, LR = {}" \
                .format(epoch_id + 1, self.num_epochs, np.mean(train_losses), self.optimizer.param_groups[0]['lr'])
            val_log = "Validation: Epoch [{}/{}], Mean Epoch Loss: {:.4f}, LR = {}" \
                .format(epoch_id + 1, self.num_epochs, np.mean(val_losses), self.optimizer.param_groups[0]['lr'])

            print(train_log)
            print(val_log)

            print('Total time train + val of epoch {}: {:.4f} seconds'.format(epoch_id + 1, (end - start)))
            print('------ END Epoch [{}/{}] ------'.format(epoch_id + 1, self.num_epochs))
            self.early_stopping(val_loss=np.mean(val_losses), model=self.model)
            if self.early_stopping.check_early_stop:
                print("Early stopping ...")
                break

        print("End of training.")
        end_train = time.time()
        train_time_info = 'Total training took {:.4f} minutes'.format((end_train - start_train) / 60)

        print(train_time_info)

        return self.model

    def forward_batch(self, batch):
        x_spt, y_spt, x_qry, y_qry = batch
        loss = self.model(x_spt, y_spt, x_qry,y_qry)
        return loss


class EarlyStopping:
    """ Adapted from:
    Title: Early Stopping for PyTorch
    Availability: https://github.com/Bjarten/early-stopping-pytorch """
    """ Early stops the training if validation loss doesn't improve after a given patience """

    def __init__(self, args):
        # How long to wait after last time validation loss improved
        self.patience = args.early_stop_patience
        # Minimum change in the monitored quantity to qualify as an improvement
        self.delta = args.delta
        self.counter = 0
        self.best_score = None
        self.check_early_stop = False
        self.val_loss_min = np.Inf
        self.model_name = args.trained_model_name

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print("Early stopping counter: {}/{}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.check_early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease """
        print("Validation loss decreased ({:.4f}  --> {:.4f}). Saving model {} ..."
              .format(self.val_loss_min, val_loss, self.model_name))
        torch.save(model.state_dict(), os.path.join("/home/lbusser/hbird_scripts/hbird_eval/data", "models", self.model_name))
        self.val_loss_min = val_loss


def count_model_params(model_parameters):
    params = list(filter(lambda p: p.requires_grad, model_parameters))
    params_summed = sum(p.numel() for p in params)
    return params_summed

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=3)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=2)
    argparser.add_argument('--img_size', type=int, help='img_size', default=504)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--num_workers', type=int, default=8)
    argparser.add_argument('--dataset', type=str, default='MSCOCO')

    # COCO training / non-episodic image captioning
    argparser.add_argument('--num_epochs', type=int, help='epoch number for training', default=20)
    argparser.add_argument('--batch_size', type=int, help='batch size for few shot training', default=16)
    argparser.add_argument('--coco_annotations_path', type=str, default='/scratch-shared/combined_hbird/mscoco_hbird/annotations/')
    argparser.add_argument('--early_stop_patience', type=int, help='#epochs w/o improvement', default=5)
    argparser.add_argument('--delta', type=float, help='min change in the monitored val loss', default=0.01)
    argparser.add_argument('--lr', type=float, help='LR for training', default=2e-04)
    argparser.add_argument('--warm_up_steps', type=int, help='warm up steps', default=100)
    argparser.add_argument('--trained_model_name', type=str, default='default.pt')
    argparser.add_argument('--patch_size', type=int, default=14)
    #SCANN setup
    argparser.add_argument('--num_leaves', type=int, default=1)
    argparser.add_argument('--num_leaves_to_search', type=int, default=1)
    argparser.add_argument('--reorder', type=int, default=1800)
    argparser.add_argument('--num_neighbors', type=int, default=5)
    args = argparser.parse_args()
    device = set_device()
    
    MODEL = 'dinov2_vitb14'
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
    
    if args.dataset =='MSCOCO':
        print("--------MSCOCO mode----------")
        cluster_index = faiss.read_index("/home/lbusser/hbird_scripts/hbird_eval/data/MSCOCO_dinov2_vitb14_1500_cluster_results/train/cluster_index.index")
        image_ids = sorted(os.listdir("/scratch-shared/combined_hbird/mscoco_hbird/train2017/"))
        train_idx, val_idx = train_test_split(np.arange(len(image_ids)), test_size=0.2, random_state=0)
        train_set = CocoMemoryTasksDataLoader(data_path="/scratch-shared/combined_hbird/mscoco_hbird", mode = 'train',  batchsz=1000, k_shot=args.k_spt, k_query=args.k_qry, resize=input_size, cluster_index= cluster_index, cluster_assignment= '/home/lbusser/hbird_scripts/hbird_eval/data/MSCOCO_dinov2_vitb14_1500_cluster_results/train/cluster_assignments.pkl', transforms = (image_train_transform, shared_train_transform),mode_idx=train_idx)
        val_set = CocoMemoryTasksDataLoader(data_path="/scratch-shared/combined_hbird/mscoco_hbird", mode = 'train', batchsz=100, k_shot=args.k_spt, k_query=args.k_qry, resize=input_size, cluster_index= cluster_index,cluster_assignment= '/home/lbusser/hbird_scripts/hbird_eval/data/MSCOCO_dinov2_vitb14_1500_cluster_results/train/cluster_assignments.pkl', transforms = (image_val_transform, shared_val_transform), mode_idx=val_idx)
        # coco_data_test = CocoMemoryTasksDataLoader(data_path="/scratch-shared/mscoco_hbird", mode="val", batchsz=10, k_shot=6, k_query=2, resize=504, cluster_index= cluster_index, transforms = (image_val_transform, shared_val_transform))
        # test_load = DataLoader(coco_data_test, batch_size=args.coco_batch_size, shuffle=False, num_workers=args.num_workers,
    #                         pin_memory=True, drop_last=True)
    elif args.dataset == 'NYUv2':
        print("--------NYUv2 mode----------")
        cluster_index_nyu = faiss.read_index("/home/lbusser/hbird_scripts/hbird_eval/data/NYUv2_dinov2_vitb14_500_cluster_results/cluster_index.index")
        image_paths = pd.read_csv("/scratch-shared/combined_hbird/nyu_hbird/nyu_data/data/nyu2_train.csv")
        train_idx, val_idx = train_test_split(np.arange(len(image_paths)), test_size=0.2, random_state = 0)
        cluster_assignment = '/home/lbusser/hbird_scripts/hbird_eval/data/NYUv2_dinov2_vitb14_500_cluster_results/cluster_assignments.pkl'
        train_set = NYUMemoryTasksDataLoader(data_path="/scratch-shared/combined_hbird/nyu_hbird/nyu_data/data/", mode="train", mode_idx = train_idx, batchsz=1000, k_shot=3, k_query=2, resize=504, cluster_index= cluster_index_nyu, cluster_assignment= cluster_assignment,  transforms = (image_train_transform, shared_train_transform))
        val_set = NYUMemoryTasksDataLoader(data_path="/scratch-shared/combined_hbird/nyu_hbird/nyu_data/data/", mode="train", mode_idx = val_idx, batchsz=100, k_shot=3, k_query=2, resize=504, cluster_index= cluster_index_nyu, cluster_assignment= cluster_assignment,  transforms = (image_train_transform, shared_train_transform))
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                            pin_memory=True, drop_last=True)
    
    vit_model = torch.hub.load('facebookresearch/dinov2', MODEL)
    feature_extractor = FeatureExtractor(vit_model)
    trainer = Trainer(args, train_loader, val_loader, feature_extractor, device)
    trained_net = trainer.train()
    main_inference(args, model_name = args.trained_model_name, cluster_index_test = f'/home/lbusser/hbird_scripts/hbird_eval/data/{args.dataset}test_dinov2_vitb14_100_cluster_results/cluster_index.index')
    print("DONE")
