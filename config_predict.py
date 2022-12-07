import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # in/out
    parser.add_argument('--outf', default='/home/wxx/Retinal_Fundus/VesselSeg-Pytorch-master/experiments',
                        help='trained model will be saved at here')

    # parser.add_argument('--save', default='LadderNet_vessel_seg', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='LadderNet_vessel_seg_val_on_test', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='UNet_vessel..._seg', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='UNet_downsize2', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='UNet_vessel..._seg', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='UNet_vessel_seg_val_on_test', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='UNet_downsize2', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='DR_UNet_vessel_seg_val_on_test', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='DR_UNet_downsize2_vessel_seg_val_on_test', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='Att_UNet_vessel_seg', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='Att_UNet_small', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='U_Net_small', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='Att_UNet_downsize2', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='Att_UNet_downsize2_vessel_seg_val_on_test', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='Dense_UNet_vessel_seg', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='Dense_UNet_vessel_seg_val_on_test', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='R2U_Net_vessel_seg_val_on_test', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='DU_Net_vessel_seg_val_on_test', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='DU_Net_downsize2_vessel_seg_val_on_test', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='Direction_UNet_vessel_seg', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='Example',help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='drive_sk_all_att_apf_fusion',help='save name of experiment in args.outf directory')
    parser.add_argument('--save', default='drive_sk_all_att_apf_fusion',help='save name of experiment in args.outf directory')



    # parser.add_argument('--save', default='Self_att_map_U_Net_small', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='Self_att_map_U_Net_downsize2', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='U_Net_apf_downsize2', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='U_Net_apf_conv_downsize2', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='U_Net_apf_df_downsize2', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='U_Net_apf_self_att_downsize2', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='U_Net_apf_DNL_downsize2', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='Self_att_U_Net_downsize2', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='Self_att_gate_U_Net_downsize2', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='Self_att_bot_U_Net_downsize2', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='DR_U_Net_downsize2', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='DR_Gabor_UNet_downsize2', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='Dynamic_conv_K4_UNet_downsize2', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='Dynamic_conv_K8_UNet_downsize2', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='Deform_UNet_downsize2', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='Cc_att_UNet_downsize2', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='UNet_bot_dilated_conv', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='Deform_modulate_UNet_downsize2', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='CondConv_UNet_downsize2', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='R2_UNet_downsize2', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='R2Att_UNet_downsize2', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='Dense_UNet_downsize2', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='DNL_unary_Bot_UNet_downsize2', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='DNL_unary_Bot_UNet_downsize2', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='Dynamic_Gabor_M4_UNet_downsize2', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='Dynamic_Gabor_M8_UNet_downsize2', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='Dynamic_Gabor_Scale_UNet_downsize2', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='Dynamic_Gabor_all_UNet_downsize2', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='LadderNet_downsize2', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='LadderNet_k16', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='LadderNet_k16_apf_DNL', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='k32_LadderNet', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='Dynamic_Gabor_LadderNet', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='Dynamic_Gabor_LadderNet', help='save name of experiment in args.outf directory')
    # parser.add_argument('--save', default='FCN', help='save name of experiment in args.outf directory')


    parser.add_argument('--test_data_path_list',
                        default='/home/wxx/Retinal_Fundus/VesselSeg-Pytorch-master/prepare_dataset/data_path_list/my_predict/test.txt')
    parser.add_argument('--train_patch_height', default=48)
    parser.add_argument('--train_patch_width', default=48)
    parser.add_argument('--N_patches', default=15000,
                        help='Number of training image patches')
    parser.add_argument('--inside_FOV', default='center',
                        help='Choose from [not,center,all]')
    parser.add_argument('--val_ratio', default=0.1,
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
    parser.add_argument('--batch_size', default=16,
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
