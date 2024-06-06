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
        # self.checkpoint_callback = ModelCheckpoint(monitor= 'val_loss')
        self.checkpoint_callback = ModelCheckpoint(monitor= 'train_loss')
        ###############
        # self.early_stopping = EarlyStopping(args)
        # self.optimizer = optim.AdamW(params=self.params_with_grad, lr=self.lr)
        # self.scheduler = get_constant_schedule_with_warmup(self.optimizer, num_warmup_steps=self.warm_up_steps)
    
        
    def train(self):
        print(f"START TRAINING {args.trained_model_name}")
        start_train = time.time()
        # early_stop_callback  = EarlyStopping(monitor='val_loss', patience=3, min_delta = 0.10)

        # lightning_trainer = L.Trainer(devices = 2, 
                                    #   accelerator = "gpu", strategy='ddp_find_unused_parameters_true', accumulate_grad_batches=8, max_epochs = self.num_epochs, num_nodes=1, callbacks=[self.checkpoint_callback])
        # lightning_trainer.fit(self.model, self.train_loader, self.val_loader)
        #############
        lightning_trainer = L.Trainer(devices = 1, 
                                      accelerator = "gpu", max_epochs = self.num_epochs, num_nodes=1, callbacks=[self.checkpoint_callback])
        lightning_trainer.fit(self.model,self.train_loader)
        #############
        # train_losses = []
        # val_losses = []
   
        # for epoch_id in range(0, self.num_epochs):
        #     print('------ START Epoch [{}/{}] ------'.format(epoch_id + 1, self.num_epochs))
        #     self.model.train()
        #     start = time.time()
        #     with tqdm(desc='Epoch [{}/{}] - training'.format(str(epoch_id + 1), self.num_epochs), unit='it',
        #               total=len(self.train_loader), position=0, leave=True) as pbar:
        #         for batch_id, batch in enumerate(self.train_loader):
                
        #             loss = self.forward_batch(batch)
        #             self.optimizer.zero_grad()
        #             loss.backward()
        #             # Compute gradients only for parameters that require them
        #             # gradients = torch.autograd.grad(loss, self.params_with_grad, allow_unused=True)
        #             # # Apply the gradients to the parameters
        #             # for param, grad in zip(self.params_with_grad, gradients):
        #             #     if grad is not None:
        #             #         param.grad = grad
        #             self.optimizer.step()
        #             self.scheduler.step()
        #             train_losses.append(loss.item())
        #             pbar.set_postfix(loss=loss.item())
        #             pbar.update()

        #         with torch.no_grad():
        #             self.model.eval()
        #             print("Start validation")
        #             for batch in tqdm(self.val_loader):
        #                 val_loss = self.forward_batch(batch)
        #                 val_losses.append(val_loss.item())

        #     end = time.time()
        #     train_log = "Training: Epoch [{}/{}], Mean Epoch Loss: {:.4f}, LR = {}" \
        #         .format(epoch_id + 1, self.num_epochs, np.mean(train_losses), self.optimizer.param_groups[0]['lr'])

        #     print(train_log)

        #     print('Total time train + val of epoch {}: {:.4f} seconds'.format(epoch_id + 1, (end - start)))
        #     print('------ END Epoch [{}/{}] ------'.format(epoch_id + 1, self.num_epochs))
        #     self.early_stopping(val_loss=np.mean(val_losses), model=self.model)
        #     if self.early_stopping.check_early_stop:
        #         print("Early stopping ...")
        #         break
        # print("End of training.")
        # end_train = time.time()
        # train_time_info = 'Total training took {:.4f} minutes'.format((end_train - start_train) / 60)

        # print(train_time_info)

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
        model = MetaTrainer.load_from_checkpoint(self.checkpoint_callback.best_model_path, args=args)
        # trainer = L.Trainer(accelerator='gpu', strategy='ddp_find_unused_parameters_true', devices=2)	
        # [out] = trainer.predict(model, self.test_loader)
        ######
        trainer = L.Trainer(accelerator= 'gpu', strategy='ddp_find_unused_parameters_true', devices=1, precision="bf16-mixed")
        [out] = trainer.predict(model, self.train_loader)
        all_predictions.append(out['pred'])
        all_labels.append(out['mask'])
        all_images.append(out['image'])
        # model = model.to(device)
        # print("Num. of tasks: {}".format(len(self.test_loader)))
        # preds_all_test= []
        # mae_scores = []
        # all_labels = []
        # all_images = []
        # model.eval()
        # miou_calculator = simple_PredsmIoU(81)
        # for (x_spt, y_spt, x_qry, y_qry) in tqdm(self.test_loader):
        #     x_spt, y_spt, x_qry, y_qry = x_spt.to(self.device), y_spt.to(self.device), x_qry.to(self.device), y_qry.to(self.device)
        #     prediction = model.generate(x_spt, y_spt, x_qry, y_qry)
        #     preds_all_test.append(prediction)
        #     all_labels.append(y_qry)
        #     all_images.append(x_qry)
        #     print("------ Test {}-shot ({}-query) ------".format(args.k_spt, args.k_qry))
        #     # my_utils.write_data_to_txt(file_path=log_file_path, data="Step: {} \tTest acc: {}\n".format(step, accs))
  
      
        print("Saving first few test entries for visualisation...")
        torch.save(all_labels[0:10], '/home/lbusser/hbird_scripts/hbird_eval/data/models/predictions/' + f"{args.trained_model_name}_gts.pt")
        torch.save(all_images[0:10], '/home/lbusser/hbird_scripts/hbird_eval/data/models/predictions/' + f"{args.trained_model_name}_imgs.pt")
        torch.save(all_predictions[0:10], '/home/lbusser/hbird_scripts/hbird_eval/data/models/predictions/' + f"{args.trained_model_name}_preds.pt")
        # miou_score = miou_calculator.compute()
        # print("Mean IoU:", miou_score)
        # # accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
      
        # print("Mean MAE:", sum(mae_scores)/len(mae_scores)

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
        torch.save(model.state_dict(), os.path.join("/home/lbusser/hbird_scripts/hbird_eval/data", "models", self.model_name+".pt"))
        self.val_loss_min = val_loss


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
    argparser.add_argument('--batch_size', type=int, help='batch size for few shot training', default=8)
    argparser.add_argument('--test_batch_size', type=int, help='batch size for few shot testing', default=4)
    argparser.add_argument('--coco_annotations_path', type=str, default='/scratch-shared/combined_hbird/mscoco_hbird/annotations/')
    argparser.add_argument('--early_stop_patience', type=int, help='#epochs w/o improvement', default=3)
    argparser.add_argument('--delta', type=float, help='min change in the monitored val loss', default=0.01)
    argparser.add_argument('--lr', type=float, help='LR for training', default=5e-04)
    argparser.add_argument('--warm_up_steps', type=int, help='warm up steps', default=100)
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
    
    cluster_index = faiss.read_index("/home/lbusser/hbird_scripts/hbird_eval/data/MSCOCO_dinov2_vitb14_1500_cluster_results/train/cluster_index.index")
    image_ids = sorted(os.listdir("/scratch-shared/combined_hbird/mscoco_hbird/train2017/"))
    train_idx, val_idx = train_test_split(np.arange(len(image_ids)), test_size=0.2, random_state=0)
    train_set_COCO = CocoMemoryTasksDataLoader(data_path="/scratch-shared/combined_hbird/mscoco_hbird", mode = 'train',  setsz=args.data_size, k_shot=args.k_spt, k_query=args.k_qry, resize=input_size, cluster_index= cluster_index, cluster_assignment= '/home/lbusser/hbird_scripts/hbird_eval/data/MSCOCO_dinov2_vitb14_1500_cluster_results/train/cluster_assignments.pkl', transforms = (image_train_transform, shared_train_transform), mode_idx=train_idx)
    val_set_COCO = CocoMemoryTasksDataLoader(data_path="/scratch-shared/combined_hbird/mscoco_hbird", mode = 'train', setsz=args.val_data_size, k_shot=args.k_spt, k_query=args.k_qry, resize=input_size, cluster_index= cluster_index,cluster_assignment= '/home/lbusser/hbird_scripts/hbird_eval/data/MSCOCO_dinov2_vitb14_1500_cluster_results/train/cluster_assignments.pkl', transforms = (image_val_transform, shared_val_transform), mode_idx=val_idx)
    # train_set_KP = KeyPointMemoryTasksDataLoader(data_path="/scratch-shared/combined_hbird/mscoco_hbird", mode="train", setsz=args.data_size, k_shot=args.k_spt, k_query=args.k_qry, resize=input_size, cluster_index = cluster_index, cluster_assignment ='/home/lbusser/hbird_scripts/hbird_eval/data/MSCOCO_dinov2_vitb14_1500_cluster_results/train/cluster_assignments.pkl',  transforms = (image_train_transform, shared_train_transform), mode_idx = train_idx)
    # val_set_KP = KeyPointMemoryTasksDataLoader(data_path="/scratch-shared/combined_hbird/mscoco_hbird", mode="train", setsz=args.val_data_size, k_shot=args.k_spt, k_query=args.k_qry, resize=input_size, cluster_index = cluster_index, cluster_assignment ='/home/lbusser/hbird_scripts/hbird_eval/data/MSCOCO_dinov2_vitb14_1500_cluster_results/train/cluster_assignments.pkl',  transforms = (image_train_transform, shared_train_transform), mode_idx= val_idx)
    # cluster_index_nyu = faiss.read_index("/home/lbusser/hbird_scripts/hbird_eval/data/NYUv2_dinov2_vitb14_500_cluster_results/cluster_index.index")
    # cluster_assignment = '/home/lbusser/hbird_scripts/hbird_eval/data/NYUv2_dinov2_vitb14_500_cluster_results/cluster_assignments.pkl'
    # image_paths = pd.read_csv("/scratch-shared/combined_hbird/nyu_hbird/nyu_data/data/nyu2_train.csv")
    # train_idx, val_idx = train_test_split(np.arange(len(image_paths)), test_size=0.2, random_state = 0)
    # train_set_NYU = NYUMemoryTasksDataLoader(data_path="/scratch-shared/combined_hbird/nyu_hbird/nyu_data/data/", mode="train", setsz=args.data_size, k_shot=args.k_spt, k_query=args.k_qry, resize=input_size, cluster_index= cluster_index_nyu, cluster_assignment= cluster_assignment,  transforms = (image_train_transform, shared_train_transform), mode_idx = train_idx)
    # val_set_NYU = NYUMemoryTasksDataLoader(data_path="/scratch-shared/combined_hbird/nyu_hbird/nyu_data/data/", mode="train", setsz=args.val_data_size, k_shot=args.k_spt, k_query=args.k_qry, resize= input_size, cluster_index= cluster_index_nyu, cluster_assignment= cluster_assignment,  transforms = (image_train_transform, shared_train_transform),mode_idx = val_idx)
    
    train_set = CombinedDataset([train_set_COCO])
    val_set = CombinedDataset([val_set_COCO])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=True)
    #------------------------------------------TEST------------------------------------
    test_coco_cluster_index = faiss.read_index('/home/lbusser/hbird_scripts/hbird_eval/data/MSCOCO_dinov2_vitb14_100_cluster_results/cluster_index.index')
    # test_nyu_cluster_index = faiss.read_index('/home/lbusser/hbird_scripts/hbird_eval/data/NYUv2_dinov2_vitb14_100_cluster_results/cluster_index.index')
    
    image_val_transform = trn.Compose([ trn.ToTensor(), trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])])
    shared_val_transform = Compose([
                Resize(size=(input_size, input_size)),
            ]) 

    coco_test_set = CocoMemoryTasksDataLoader("/scratch-shared/combined_hbird/mscoco_hbird",'val', args.test_size, args.testk_spt, args.testk_qry, args.img_size, cluster_index= test_coco_cluster_index, cluster_assignment= '/home/lbusser/hbird_scripts/hbird_eval/data/MSCOCO_dinov2_vitb14_100_cluster_results/cluster_assignments.pkl', transforms = (image_val_transform, shared_val_transform))
    # nyu_test_set = NYUMemoryTasksDataLoader("/scratch-shared/combined_hbird/nyu_hbird/nyu_data/data/", 'test', args.test_size, args.testk_spt, args.testk_qry, args.img_size, cluster_index= test_nyu_cluster_index, cluster_assignment='/home/lbusser/hbird_scripts/hbird_eval/data/NYUv2_dinov2_vitb14_100_cluster_results/cluster_assignments.pkl', transforms = (image_val_transform, shared_val_transform))
    # kp_test_set = KeyPointMemoryTasksDataLoader("/scratch-shared/combined_hbird/mscoco_hbird", 'val', args.test_size, args.testk_spt, args.testk_qry, args.img_size, cluster_index= test_coco_cluster_index, cluster_assignment= '/home/lbusser/hbird_scripts/hbird_eval/data/MSCOCO_dinov2_vitb14_100_cluster_results/cluster_assignments.pkl', transforms = (image_val_transform, shared_val_transform))
    test_set = CombinedDataset([coco_test_set])

    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    #------------------------initalise model and train--------------------------------
    trainer = MainTrainer(args, train_loader, val_loader, test_loader)
    trainer.train()
    print("-------------------------------------------------------INFERENCE-------------------------------------------------------")
    trainer.inference()
    print(f"DONE TESTING {args.trained_model_name}")
