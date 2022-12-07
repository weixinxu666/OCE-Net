import joblib, copy
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch, sys
from tqdm import tqdm

from collections import OrderedDict
from lib.visualize import save_img, group_images, concat_result, single_result, single_result_prob
import os
import argparse
from lib.logger import Logger, Print_Logger
from lib.extract_patches import *
from os.path import join
from lib.dataset import TestDataset
from lib.metrics import Evaluate
import models
from lib.common import setpu_seed, dict_round
from config_predict import parse_args
from lib.pre_processing import my_PreProc

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

setpu_seed(2021)


class Test():
    def __init__(self, args):
        self.args = args
        assert (args.stride_height <= args.test_patch_height and args.stride_width <= args.test_patch_width)
        # save path
        self.path_experiment = join(args.outf, args.save)

        self.patches_imgs_test, self.test_imgs, self.new_height, self.new_width = get_data_test_overlap_wo_fov_gt(
            test_data_path_list=args.test_data_path_list,
            patch_height=args.test_patch_height,
            patch_width=args.test_patch_width,
            stride_height=args.stride_height,
            stride_width=args.stride_width
        )
        self.img_height = self.test_imgs.shape[2]
        self.img_width = self.test_imgs.shape[3]

        test_set = TestDataset(self.patches_imgs_test)
        self.test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=3)

    # Inference prediction process
    def inference(self, net):
        net.eval()
        preds = []
        with torch.no_grad():
            for batch_idx, inputs in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                inputs = inputs.cuda()
                outputs = net(inputs)
                outputs = outputs[:, 1].data.cpu().numpy()
                preds.append(outputs)
        predictions = np.concatenate(preds, axis=0)
        self.pred_patches = np.expand_dims(predictions, axis=1)

    # Evaluate ate and visualize the predicted images
    def evaluate(self):
        self.pred_imgs = recompone_overlap(
            self.pred_patches, self.new_height, self.new_width, self.args.stride_height, self.args.stride_width)
        ## restore to original dimensions
        self.pred_imgs = self.pred_imgs[:, :, 0:self.img_height, 0:self.img_width]



    # save segmentation imgs
    def save_segmentation_result(self):
        img_path_list = load_file_path_txt_wo_fov_gt(self.args.test_data_path_list)
        img_name_list = [item.split(' ')[0].split('/')[-1].split('.')[0] for item in img_path_list]

        print(img_path_list)
        print(img_name_list)

        # kill_border(self.pred_imgs, self.test_FOVs)  # only for visualization
        self.save_img_path = join('/home/wxx/Retinal_Fundus/VesselSeg-Pytorch-master/result_predict_img')
        if not os.path.exists(join(self.save_img_path)):
            os.makedirs(self.save_img_path)
        # self.test_imgs = my_PreProc(self.test_imgs) # Uncomment to save the pre processed image
        for i in range(self.test_imgs.shape[0]):
            # total_img = concat_result(self.test_imgs[i],self.pred_imgs[i],self.test_masks[i])
            total_img = single_result(self.test_imgs[i], self.pred_imgs[i], self.pred_imgs[i])
            # total_img = single_result_prob(self.test_imgs[i], self.pred_imgs[i], self.pred_imgs[i])
            # save_img(total_img,join(self.save_img_path, "Result_"+img_name_list[i]+'.png'))
            save_img(total_img, join(self.save_img_path, "Result_" + img_name_list[i] + '.png'))



if __name__ == '__main__':
    args = parse_args()
    # save_path = args.outf
    save_path = join(args.outf, args.save)
    # sys.stdout = Print_Logger(os.path.join(save_path, 'test_log.txt'))
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    net = models.UNetFamily.U_Net_small(1,2).to(device)

    cudnn.benchmark = True

    # Load checkpoint
    print('==> Loading checkpoint...')
    checkpoint = torch.load(join(save_path, 'best_model.pth'))
    net.load_state_dict(checkpoint['net'])

    eval = Test(args)
    eval.inference(net)
    print(eval.evaluate())
    eval.save_segmentation_result()
