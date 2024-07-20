import argparse
import datetime

from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from models import FeatureExtractorBeta as FeatureExtractor
import my_utils
from torch.nn import functional as F
from data_loader import *
from fs_trainer import MetaTrainer
from tqdm import tqdm
from eval_metrics import *
from sklearn.metrics import mean_absolute_error

PATH = str(Path.cwd().parent.parent.parent)  # root directory
MODELS_PATH = '/home/lbusser/hbird_scripts/hbird_eval/data/models'
MODEL = 'dinov2_vitb14'

torch.manual_seed(222)
torch.cuda.manual_seed_all(222)
np.random.seed(222)
device = set_device()


def main_inference(args, model_name):
  
    input_size = 504
    vit_model = torch.hub.load('facebookresearch/dinov2', MODEL)
    feature_extractor = FeatureExtractor(vit_model)
    model = MetaTrainer(args, feature_extractor, mode = 'test')
    model_dict = torch.load('data/models/' + model_name, map_location=torch.device(device))
    model.load_state_dict(model_dict)
    params = list(filter(lambda p: p.requires_grad, model.model.parameters()))
    params_summed = sum(p.numel() for p in params)
    print("Total num of params: {} ".format(params_summed))
 
    image_val_transform = trn.Compose([ trn.ToTensor(), trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])])
    shared_val_transform = Compose([
            Resize(size=(input_size, input_size)),
        ])

    coco_cluster_index = faiss.read_index('data/MSCOCO_dinov2_vitb14_100_cluster_results/cluster_index.index')
    nyu_cluster_index = faiss.read_index('data/NYUv2_dinov2_vitb14_100_cluster_results/cluster_index.index')
    coco_test_set = CocoMemoryTasksDataLoader("mscoco_hbird",'val', 100, args.k_spt, args.k_qry, args.img_size, cluster_index= coco_cluster_index, cluster_assignment= '/home/lbusser/hbird_scripts/hbird_eval/data/MSCOCO_dinov2_vitb14_100_cluster_results/cluster_assignments.pkl',transforms = (image_val_transform, shared_val_transform))
    nyu_test_set = NYUMemoryTasksDataLoader("nyu_hbird/nyu_data/data/", 'test', 100, args.k_spt, args.k_qry, args.img_size, cluster_index= nyu_cluster_index, cluster_assignment='/home/lbusser/hbird_scripts/hbird_eval/data/NYUv2_dinov2_vitb14_100_cluster_results/cluster_assignments.pkl', transforms = (image_val_transform, shared_val_transform))
    
    test_set = CombinedDataset([coco_test_set, nyu_test_set])
    db_test = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print("Num. of tasks: {}".format(len(db_test)))
    preds_all_test= []
    mae_scores = []
    all_labels = []
    all_images = []
    model.eval()
    miou_calculator = simple_PredsmIoU(81)
    for (x_spt, y_spt, x_qry, y_qry), task_id in tqdm(db_test):
        prediction = model(x_spt, y_spt, x_qry, y_qry, task_id)
        print(prediction.shape)
        if task_id ==1:
            mae_scores.append(mean_absolute_error(y_qry[0].squeeze(), prediction.cpu().squeeze()))
        elif task_id == 0:
            miou_calculator.update(prediction, y_qry[0])
        preds_all_test.append(prediction)
        all_labels.append(y_qry[0])
        all_images.append(x_qry)
        print("------ Test {}-shot ({}-query) ------".format(args.k_spt, args.k_qry))
        # my_utils.write_data_to_txt(file_path=log_file_path, data="Step: {} \tTest acc: {}\n".format(step, accs))
   
    miou_score = miou_calculator.compute()
    print("Mean IoU:", miou_score)
    # accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
    print("Mean MAE:", sum(mae_scores)/len(mae_scores))
    torch.save(all_labels, 'data/models/predictions/' + f"{model_name}_gts.pt")
    torch.save(all_images, 'data/models/predictions/' + f"{model_name}_imgs.pt")
    torch.save(preds_all_test, 'data/models/predictions/' + f"{model_name}_preds.pt")
    # my_utils.write_data_to_txt(file_path=log_file_path, data="FINAL: \tTest acc: {} \n".format(accs))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=3)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=1)
    argparser.add_argument('--img_size', type=int, help='img_size', default=504)
    argparser.add_argument('--seq_len', type=int, default=16)  # for padding batch
    argparser.add_argument('--batch_size', type=int, default=1)
    argparser.add_argument('--trained_model_name', type=str, default='default.pt')

    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--num_workers', type=int, default=8)
    argparser.add_argument('--patch_size', type=int, default=14)
    argparser.add_argument('--num_leaves', type=int, default=1)
    argparser.add_argument('--num_leaves_to_search', type=int, default=1)
    argparser.add_argument('--reorder', type=int, default=1800)
    argparser.add_argument('--num_neighbors', type=int, default=5)

    args = argparser.parse_args()

    model_name = args.trained_model_name

    main_inference(args, model_name = model_name)
