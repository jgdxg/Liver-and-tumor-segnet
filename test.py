# -*- coding: utf-8 -*-

import os
from datetime import datetime

import SimpleITK as sitk
import pandas as pd
from time import time
import argparse
import warnings
import numpy as np
import torch
import collections
from cs import Metirc
from net import Unet,fct,snpt1
from net.comper import unet2,attentionnet
import joblib
import copy
torch.cuda.current_device()
torch.cuda._initialized = True

test_ct_path = './LITS2017/test'   #需要预测的CT图像
seg_result_path = './LITS2017/seg' #需要预标签
pred_path = './pred_result/attentionnet'
# pred_path = './pred_result/unet2'
model_name = "UNet2d"
result_path1 = './data/val_result/'

file_name = []  # 文件名称
time_pre_case = []  # 单例数据消耗时间
# 定义评价指标
liver_score = collections.OrderedDict()
liver_score['dice'] = []
liver_score['jacard'] = []
liver_score['voe'] = []
liver_score['fnr'] = []
liver_score['fpr'] = []
liver_score['assd'] = []
liver_score['rmsd'] = []
liver_score['msd'] = []

tumor_score = collections.OrderedDict()
tumor_score['dice'] = []
tumor_score['jacard'] = []
tumor_score['voe'] = []
tumor_score['fnr'] = []
tumor_score['fpr'] = []
tumor_score['assd'] = []
tumor_score['rmsd'] = []
tumor_score['msd'] = []

liver_dice_intersection = 0
if not os.path.exists(pred_path):
    os.mkdir(pred_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default=None,
                        help='')
    parser.add_argument('--training', type=bool, default=False,
                    help='whthere dropout or not')

    args = parser.parse_args()

    return args

# 为了计算dice_global定义的两个变量
def main():
    val_args = parse_args()
    dice_intersection = 0.0
    dice_union = 0.0
    liver_dice_intersection = 0.0
    liver_dice_union = 0.0
    liver_voe_intersection = 0.0
    liver_voe_union = 0.0
    tumor_dice_intersection = 0.0
    tumor_dice_union = 0.0

    args = joblib.load('models/LiTS_Unet_lym/attention/args.pkl')
    # args = joblib.load('models/LiTS_Unet_lym/unet++/args.pkl')
    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')
    joblib.dump(args, 'models/LiTS_Unet_lym/attention/args.pkl')
    # joblib.dump(args, 'models/LiTS_Unet_lym/unet++/args.pkl')

    # create model
    print("=> creating model %s" %args.arch)
    print(torch.cuda.is_available())
    # model = Unet.U_Net()
    # model = fct.FCT(args)
    # model = snpt1.FCT(args)
    model = attentionnet.AttU_Net()
    # model = unet2.UnetPlusPlus()
    model = torch.nn.DataParallel(model).cuda()

    model.load_state_dict(torch.load('models/LiTS_Unet_lym/attention/epoch90-0.9651-0.7257_model.pth'))
    model.eval()
    # model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
    
    for file_index, file in enumerate(os.listdir(test_ct_path)):
        start = time()

        # 将CT读入内存
        ct = sitk.ReadImage(os.path.join(test_ct_path, file), sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)

        mask = sitk.ReadImage(os.path.join(seg_result_path, file.replace('volume', 'segmentation')), sitk.sitkUInt8)
        mask_array = sitk.GetArrayFromImage(mask)

        mask_array[mask_array > 0] = 1

        print('start predict file:',file)

        # 阈值截取
        ct_array[ct_array > 200] = 200
        ct_array[ct_array < -200] = -200

        ct_array = ct_array.astype(np.float32)
        ct_array = ct_array / 200

        # 找到肝脏区域开始和结束的slice，并各向外扩张slice
        z = np.any(mask_array, axis=(1, 2))
        start_slice, end_slice = np.where(z)[0][[0, -1]]

        start_slice = max(0, start_slice - 10)
        end_slice = min(mask_array.shape[0]-1, end_slice + 10)

        ct_crop = ct_array[start_slice:end_slice+1,32:480,32:480]

        slice_predictions = np.zeros((ct_array.shape[0],512,512),dtype=np.int16)

        with torch.no_grad():
            for n_slice in range(ct_crop.shape[0]-3):
                ct_tensor = torch.FloatTensor(ct_crop[n_slice: n_slice + 3]).cuda()
                ct_tensor = ct_tensor.unsqueeze(dim=0)
                # print('ct_tensor',ct_tensor.shape,n_slice)
                # print('input:',ct_tensor.shape)
                output = model(ct_tensor)
                output = torch.sigmoid(output).data.cpu().numpy()
                # print('output:',output.shape)
                probability_map = np.zeros([1, 448, 448], dtype=np.uint8)
                #预测值拼接回去
                # i = 0
                for idz in range(output.shape[1]):
                    for idx in range(output.shape[2]):
                        for idy in range(output.shape[3]):
                            if (output[0,0, idx, idy] > 0.5):
                                probability_map[0, idx, idy] = 1        
                            if (output[0,1, idx, idy] > 0.5):
                                # 大于肿瘤
                                probability_map[0, idx, idy] = 2


                slice_predictions[n_slice+start_slice+1,32:480,32:480] = probability_map


            pred_seg = slice_predictions
            pred_seg = pred_seg.astype(np.uint8)

            # liver_metric = Metirc(mask_array, pred_seg, ct.GetSpacing())
            #
            # liver_score['dice'].append(liver_metric.get_dice_coefficient()[0])
            # print('dice:',liver_metric.get_dice_coefficient()[0])
            # liver_score['jacard'].append(liver_metric.get_jaccard_index())
            # print('JAC:',liver_metric.get_jaccard_index())
            # liver_score['voe'].append(liver_metric.get_VOE())
            # print('VOE:',liver_metric.get_VOE())
            # liver_score['rvd'].append(liver_metric.get_RVD())
            # print('RVD',liver_metric.get_RVD())
            # liver_score['fnr'].append(liver_metric.get_FNR())
            # liver_score['fpr'].append(liver_metric.get_FPR())
            # liver_score['assd'].append(liver_metric.get_ASSD())
            # print('ASD:',liver_metric.get_ASSD())
            # liver_score['rmsd'].append(liver_metric.get_RMSD())
            # print('rmsd:',liver_metric.get_RMSD())
            # liver_score['msd'].append(liver_metric.get_MSD())
            # print('msd:',liver_metric.get_MSD())
            #
            # dice_intersection += liver_metric.get_dice_coefficient()[1]
            # dice_union += liver_metric.get_dice_coefficient()[2]


            # 将预测的结果保存为nii数据
            pred_seg = sitk.GetImageFromArray(pred_seg)
            pred_seg.SetDirection(ct.GetDirection())
            pred_seg.SetOrigin(ct.GetOrigin())
            pred_seg.SetSpacing(ct.GetSpacing())
            sitk.WriteImage(pred_seg, os.path.join(pred_path, file.replace('volume', 'segmentation').replace('nii', 'nii.gz')))
            speed = time() - start

            print(file, 'this case use {:.3f} s'.format(speed))
            print('-----------------------')
            torch.cuda.empty_cache()


        seg = sitk.ReadImage(os.path.join(seg_result_path, file.replace('volume', 'segmentation')),
                                 sitk.sitkUInt8)
        seg_array = sitk.GetArrayFromImage(seg)

            # 将预测读入内存
        pred_seg = sitk.ReadImage(os.path.join(pred_path, file.replace('volume', 'segmentation')),
                                      sitk.sitkUInt8)
        pred_seg_array = sitk.GetArrayFromImage(pred_seg)

            # 计算分割评价指标：
            # 肝脏分割指标计算
        liver_seg_array = copy.deepcopy(seg_array)  # 金标准
        liver_seg = copy.deepcopy(pred_seg_array)  # 预测值
        liver_seg_array[liver_seg_array > 0] = 1  # 肝脏金标准
        liver_seg[liver_seg > 0] = 1  # 肝脏预测标签

        liver_metric = Metirc(liver_seg_array, liver_seg, ct.GetSpacing())

        liver_score['dice'].append(liver_metric.get_dice_coefficient()[0])
        liver_score['jacard'].append(liver_metric.get_jaccard_index())
        liver_score['voe'].append(liver_metric.get_VOE())
        liver_score['fnr'].append(liver_metric.get_FNR())
        liver_score['fpr'].append(liver_metric.get_FPR())
        liver_score['assd'].append(liver_metric.get_ASSD())
        liver_score['rmsd'].append(liver_metric.get_RMSD())
        liver_score['msd'].append(liver_metric.get_MSD())

        liver_dice_intersection += liver_metric.get_dice_coefficient()[1]
        liver_dice_union += liver_metric.get_dice_coefficient()[2]
        # liver_voe_intersection += liver_metric.get_VOE()[1]
        # liver_voe_union += liver_metric.get_VOE()[2]
        print(file, "肝脏预测评估完成")

        # 肿瘤分割指标计算
        tumor_seg_array = copy.deepcopy(seg_array)  # 金标准
        tumor_seg = copy.deepcopy(pred_seg_array)  # 预测值
        tumor_seg_array[tumor_seg_array == 1] = 0  # 肿瘤金标准
        tumor_seg_array[tumor_seg_array > 1] = 1
        tumor_seg[tumor_seg == 1] = 0  # 肿瘤预测标签
        tumor_seg[tumor_seg > 1] = 1

        tumor_metric = Metirc(tumor_seg_array, tumor_seg, ct.GetSpacing())

        tumor_score['dice'].append(tumor_metric.get_dice_coefficient()[0])
        tumor_score['jacard'].append(tumor_metric.get_jaccard_index())
        tumor_score['voe'].append(tumor_metric.get_VOE())
        tumor_score['fnr'].append(tumor_metric.get_FNR())
        tumor_score['fpr'].append(tumor_metric.get_FPR())
        tumor_score['assd'].append(tumor_metric.get_ASSD())
        tumor_score['rmsd'].append(tumor_metric.get_RMSD())
        tumor_score['msd'].append(tumor_metric.get_MSD())

        tumor_dice_intersection += tumor_metric.get_dice_coefficient()[1]
        tumor_dice_union += tumor_metric.get_dice_coefficient()[2]
        print(file, "肿瘤预测评估完成")

            # 计时
        speed = time() - start
        time_pre_case.append(speed)

        print(file, 'this case use {:.3f} s'.format(speed))
        print('-----------------------')

        # 打印dice global
    if liver_dice_union != 0:
            print('liver dice global:', liver_dice_intersection / liver_dice_union)
            # print('liver voe global:', liver_voe_intersection / liver_voe_union)
    if tumor_dice_union != 0:
            print('tumor dice global:', tumor_dice_intersection / tumor_dice_union)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    main()


        
