import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # in/out
    parser.add_argument('--outf', default='./experiments',
                        help='trained model will be saved at here')



    parser.add_argument('--save', default='oce_net', help='save name of experiment in args.outf directory')



    # data   DRIVE
    parser.add_argument('--train_data_path_list',
                        default='/home/wxx/Retinal_Fundus/VesselSeg-Pytorch-master/prepare_dataset/data_path_list/DRIVE/train.txt')
    parser.add_argument('--test_data_path_list',
                        default='/home/wxx/Retinal_Fundus/VesselSeg-Pytorch-master/prepare_dataset/data_path_list/DRIVE/test.txt')

    # # data   STARE
    # parser.add_argument('--train_data_path_list',
    #                     default='/home/wxx/Retinal_Fundus/VesselSeg-Pytorch-master/prepare_dataset/data_path_list/STARE/train.txt')
    # parser.add_argument('--test_data_path_list',
    #                     default='/home/wxx/Retinal_Fundus/VesselSeg-Pytorch-master/prepare_dataset/data_path_list/STARE/test.txt')

    # # data   CHASEDB1
    # parser.add_argument('--train_data_path_list',
    #                     default='/home/wxx/Retinal_Fundus/VesselSeg-Pytorch-master/prepare_dataset/data_path_list/CHASEDB1/train.txt')
    # parser.add_argument('--test_data_path_list',
    #                     default='/home/wxx/Retinal_Fundus/VesselSeg-Pytorch-master/prepare_dataset/data_path_list/CHASEDB1/test.txt')




    parser.add_argument('--train_patch_height', default=48)
    parser.add_argument('--train_patch_width', default=48)
    parser.add_argument('--N_patches', default=15000,
                        help='Number of training image patches')
    parser.add_argument('--inside_FOV', default='not',
                        help='Choose from [not,center,all]')
    parser.add_argument('--val_ratio', default=0.05,
                        help='The ratio of the validation set in the training set')
    parser.add_argument('--sample_visualization', default=True,
                        help='Visualization of training samples')
    # model parameters
    parser.add_argument('--in_channels', default=1,type=int,
                        help='input channels of model')
    parser.add_argument('--classes', default=2,type=int, 
                        help='output channels of model')

    # training
    parser.add_argument('--N_epochs', default=100, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=8,
                        type=int, help='batch size')
    parser.add_argument('--early-stop', default=8, type=int,
                        help='early stopping')
    parser.add_argument('--lr', default=0.0005, type=float,
                        help='initial learning rate')
    parser.add_argument('--val_on_test', default=True, type=bool,
                        help='Validation on testset')

    # for pre_trained checkpoint
    parser.add_argument('--start_epoch', default=1, 
                        help='Start epoch')
    parser.add_argument('--pre_trained', default=None,
                        help='(path of trained _model)load trained model to continue train')

    # testing
    parser.add_argument('--test_patch_height', default=64)
    parser.add_argument('--test_patch_width', default=64)
    parser.add_argument('--stride_height', default=16)
    parser.add_argument('--stride_width', default=16)

    # hardware setting
    parser.add_argument('--cuda', default=True, type=bool,
                        help='Use GPU calculating')

    args = parser.parse_args()

    return args
