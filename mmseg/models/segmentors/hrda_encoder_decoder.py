# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------


import numpy as np
import torch
import torch.nn as nn
from mmseg.ops import resize
from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder

import torchvision.transforms as transforms
from PIL import Image
import os
import torch.nn.functional as F  # 导入功能模块
import cv2
import imageio

from torchvision import models

# 保存平均后的特征图
def save_average_feature_map(feature_map, base_path):
    feature_map = feature_map.cpu().numpy()
    # 对所有通道求平均
    average_feature_map = np.mean(feature_map, axis=1)

    # 归一化到[0, 1]范围
    min_val = np.min(average_feature_map)
    max_val = np.max(average_feature_map)
    normalized_feature_map = (average_feature_map - min_val) / (max_val - min_val)
    
    # 转换为[0, 255]范围的整数
    int_feature_map = (normalized_feature_map * 255).astype(np.uint8)

    # 查找递增的保存地址
    save_path = base_path
    index = 1
    for img_tensor in int_feature_map:
        while os.path.exists(save_path):
            save_path = f"{base_path.rsplit('.', 1)[0]}_{index}.{base_path.rsplit('.', 1)[1]}"
            index += 1

        # 保存图像
        imageio.imwrite(save_path, img_tensor)


def denormalize(image, mean, std):
    mean = np.array(mean)
    std = np.array(std)
    
    # 反归一化
    denormalized_image = (image * std) + mean
    # 四舍五入并裁剪
    denormalized_image = np.clip(np.round(denormalized_image), 0, 255)
    
    # 转换为 uint8 类型
    denormalized_image = denormalized_image.astype(np.uint8)
    return denormalized_image

def build_laplacian_pyramid(image, levels):
    # 构建高斯金字塔
    G = [image]
    image = cv2.pyrDown(image)
    G.append(image)

    # 构建拉普拉斯金字塔
    GE = cv2.pyrUp(G[1])
    GE = cv2.resize(GE, (G[0].shape[1], G[0].shape[0]))  # 调整大小以匹配上一层的大小
    residual = cv2.subtract(G[0], GE)

    return residual

# 将彩色图像转换为灰度图
def convert_to_grayscale(batch_images):
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]
    batch_gray_images = []
    for img in batch_images:
        if img.is_cuda:
            img = img.cpu().numpy()
        img = np.transpose(img, (1, 2, 0))  # (channels, height, width) -> (height, width, channels)
        denormalized_image = denormalize(img, mean, std)

        gray_img = cv2.cvtColor(denormalized_image, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
        batch_gray_images.append(gray_img)
    batch_gray_images = np.stack(batch_gray_images, axis=0)  # List of (height, width) -> (batch, height, width)
    return batch_gray_images


def get_crop_bbox(img_h, img_w, crop_size, divisible=1):
    """Randomly get a crop bounding box."""
    assert crop_size[0] > 0 and crop_size[1] > 0
    if img_h == crop_size[-2] and img_w == crop_size[-1]:
        return (0, img_h, 0, img_w)
    margin_h = max(img_h - crop_size[-2], 0)
    margin_w = max(img_w - crop_size[-1], 0)
    offset_h = np.random.randint(0, (margin_h + 1) // divisible) * divisible
    offset_w = np.random.randint(0, (margin_w + 1) // divisible) * divisible
    crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
    crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

    return crop_y1, crop_y2, crop_x1, crop_x2


def crop(img, crop_bbox):
    """Crop from ``img``"""
    crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
    if img.dim() == 4:
        img = img[:, :, crop_y1:crop_y2, crop_x1:crop_x2]
    elif img.dim() == 3:
        img = img[:, crop_y1:crop_y2, crop_x1:crop_x2]
    elif img.dim() == 2:
        img = img[crop_y1:crop_y2, crop_x1:crop_x2]
    else:
        raise NotImplementedError(img.dim())
    return img


@SEGMENTORS.register_module()
class HRDAEncoderDecoder(EncoderDecoder):
    last_train_crop_box = {}

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 scales=[1],
                 hr_crop_size=None,
                 hr_slide_inference=True,
                 hr_slide_overlapping=True,
                 crop_coord_divisible=1,
                 blur_hr_crop=False,
                 feature_scale=1):
        self.feature_scale_all_strs = ['all']
        if isinstance(feature_scale, str):
            assert feature_scale in self.feature_scale_all_strs
        scales = sorted(scales)
        decode_head['scales'] = scales
        decode_head['enable_hr_crop'] = hr_crop_size is not None
        decode_head['hr_slide_inference'] = hr_slide_inference
        super(HRDAEncoderDecoder, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        self.scales = scales
        self.feature_scale = feature_scale
        self.crop_size = hr_crop_size
        self.hr_slide_inference = hr_slide_inference
        self.hr_slide_overlapping = hr_slide_overlapping
        self.crop_coord_divisible = crop_coord_divisible
        self.blur_hr_crop = blur_hr_crop
        self.sigmoid = nn.Sigmoid()


    def extract_unscaled_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def extract_slide_feat(self, img, lr_feature):


        if self.hr_slide_overlapping:
            h_stride, w_stride = [e // 2 for e in self.crop_size]
        else:
            h_stride, w_stride = self.crop_size
        h_crop, w_crop = self.crop_size
        bs, _, h_img, w_img = img.size()
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        crop_imgs, crop_feats, crop_boxes = [], [], []
        save_path = '/gemini/code/crop/crop.png'
        base_path = save_path

        index = 1
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_imgs.append(img[:, :, y1:y2, x1:x2])
                crop_boxes.append([y1, y2, x1, x2])

                # for img_tensor in img:
                #     while os.path.exists(save_path):
                #         save_path = f"{base_path.rsplit('.', 1)[0]}_{index}.{base_path.rsplit('.', 1)[1]}"
                #         index += 1
                #     to_pil_image = transforms.ToPILImage()
                #     img_tensor = img_tensor[:3, y1:y2, x1:x2]
                #     img1 = to_pil_image(img_tensor)    # 转换为 PIL 图像
                #     img1.save(save_path)  # 保存图像

        crop_imgs = torch.cat(crop_imgs, dim=0)
        crop_feats = self.extract_unscaled_feat(crop_imgs)
        # shape: feature levels, crops * batch size x c x h x w
        
        m, c, orig_h, orig_w = img.shape
        b, c, feat_h, feat_w = lr_feature.shape
        b, c, feat_h2, feat_w2 = crop_feats[3].shape
        batch_gray_images = convert_to_grayscale(img)
        # 为批次中的每张图像构建拉普拉斯金字塔
        laplacian_pyramids = []
        for image in batch_gray_images:
            laplacian_pyramid = build_laplacian_pyramid(image, 1)
            # cv2.imwrite('/gemini/code/HRDA-master/photo/1.png', laplacian_pyramid)
            laplacian_pyramids.append(laplacian_pyramid)  # 只取第一层的拉普拉斯图像
        laplacian_pyramids_batch = np.stack(laplacian_pyramids, axis=0)  # (batch, height, width)
        # 将numpy数组转换为PIL Image对象
        image = Image.fromarray(laplacian_pyramids_batch[0].astype('uint8'))  # 假设取batch中的第1张图像
        # 保存图像
        image.save('output_image.png')
        layer_tensor = torch.tensor(laplacian_pyramids_batch, dtype=torch.float32) / 255.0  # 归一化到 [0, 1]
        layer_tensor = layer_tensor.unsqueeze(1).cuda()
        for i in range(0, len(crop_boxes), m):
            crop_y1, crop_y2, crop_x1, crop_x2 = crop_boxes[i]
            y1_feat = int(crop_y1 * feat_h / orig_h)
            y2_feat = int(crop_y2 * feat_h / orig_h)
            x1_feat = int(crop_x1 * feat_w / orig_w)
            x2_feat = int(crop_x2 * feat_w / orig_w)
            new_cropbox = []
            new_cropbox.append(y1_feat)
            new_cropbox.append(y2_feat)
            new_cropbox.append(x1_feat)
            new_cropbox.append(x2_feat)
            scaled_feature = crop(lr_feature, new_cropbox) #在特征层裁处对应crop区域
            size = (feat_h2, feat_w2)
            upsampled_feature_map = F.interpolate(scaled_feature, size=size, mode='bilinear', align_corners=False)
            # residual_feature = torch.abs(self.sigmoid(mres_feats[1][0])- self.sigmoid(upsampled_feature_map))
            # img_tensor = residual_feature[0]  # 选择第 i 个 batch
            # to_pil_image = transforms.ToPILImage()
            # # 选择前3个通道作为RGB图像
            # img_tensor = img_tensor[:3, :, :]
            # img = to_pil_image(img_tensor)    # 转换为 PIL 图像
            # img.save('/gemini/code/HRDA-master/photo/1.png')  # 保存图像
            # 将批次图像转换为灰度图

            layer_tensor1 = crop(layer_tensor, crop_boxes[i])
            layer_tensor1 = F.interpolate(layer_tensor1, size=size, mode='bilinear', align_corners=False)


            # # 保存平均后的特征图
            # base_save_path = '/gemini/code/feature_hr/average_feature_map.png'
            # save_average_feature_map(crop_feats[3][i:i+m], base_save_path)

            # # 保存平均后的特征图
            # base_save_path = '/gemini/code/feature_lr/average_feature_map.png'
            # save_average_feature_map(lr_feature, base_save_path)

            # # 保存平均后的特征图
            # base_save_path = '/gemini/code/up_feature_lr/average_feature_map.png'
            # save_average_feature_map(lr_feature, base_save_path)

            crop_feats[3][i:i+m] = layer_tensor1 * crop_feats[3][i:i+m] + (1-layer_tensor1)*upsampled_feature_map


        # else:
        #     for i in range(0, len(crop_boxes)):
        #         crop_y1, crop_y2, crop_x1, crop_x2 = crop_boxes[i]
        #         y1_feat = int(crop_y1 * feat_h / orig_h)
        #         y2_feat = int(crop_y2 * feat_h / orig_h)
        #         x1_feat = int(crop_x1 * feat_w / orig_w)
        #         x2_feat = int(crop_x2 * feat_w / orig_w)
        #         new_cropbox = []
        #         new_cropbox.append(y1_feat)
        #         new_cropbox.append(y2_feat)
        #         new_cropbox.append(x1_feat)
        #         new_cropbox.append(x2_feat)
        #         scaled_feature = crop(lr_feature, new_cropbox) #在特征层裁处对应crop区域
        #         size = (feat_h2, feat_w2)
        #         upsampled_feature_map = F.interpolate(scaled_feature, size=size, mode='bilinear', align_corners=False)
        #         # residual_feature = torch.abs(self.sigmoid(mres_feats[1][0])- self.sigmoid(upsampled_feature_map))
        #         # img_tensor = residual_feature[0]  # 选择第 i 个 batch
        #         # to_pil_image = transforms.ToPILImage()
        #         # # 选择前3个通道作为RGB图像
        #         # img_tensor = img_tensor[:3, :, :]
        #         # img = to_pil_image(img_tensor)    # 转换为 PIL 图像
        #         # img.save('/gemini/code/HRDA-master/photo/1.png')  # 保存图像
        #         # 将批次图像转换为灰度图

        #         layer_tensor1 = crop(layer_tensor, crop_boxes[i])
        #         layer_tensor1 = F.interpolate(layer_tensor1, size=size, mode='bilinear', align_corners=False)
        #         print(layer_tensor1.shape)
        #         print(crop_feats[3][i].shape)
        #         print(upsampled_feature_map.shape)
        #         crop_feats[3][i] = layer_tensor1 * crop_feats[3][i] + (1-layer_tensor1)*upsampled_feature_map
            
        return {'features': crop_feats, 'boxes': crop_boxes}

    def blur_downup(self, img, s=0.5):
        img = resize(
            input=img,
            scale_factor=s,
            mode='bilinear',
            align_corners=self.align_corners)
        img = resize(
            input=img,
            scale_factor=1 / s,
            mode='bilinear',
            align_corners=self.align_corners)
        return img

    def resize(self, img, s):
        if s == 1:
            return img
        else:
            with torch.no_grad():
                return resize(
                    input=img,
                    scale_factor=s,
                    mode='bilinear',
                    align_corners=self.align_corners)

    def extract_feat(self, img):
        if self.feature_scale in self.feature_scale_all_strs:
            mres_feats = []
            for i, s in enumerate(self.scales):
                if s == 1 and self.blur_hr_crop:
                    scaled_img = self.blur_downup(img)
                else:
                    scaled_img = self.resize(img, s)
                if self.crop_size is not None and i >= 1:
                    tt = scaled_img
                    scaled_img = crop(
                        scaled_img, HRDAEncoderDecoder.last_train_crop_box[i])

                mres_feats.append(self.extract_unscaled_feat(scaled_img))
            # b, c, orig_h, orig_w = img.shape()
            # b, c, feat_h, feat_w = mres_feats[0].shape()
            # crop_y1, crop_y2, crop_x1, crop_x2 = HRDAEncoderDecoder.last_train_crop_box[1]
            # y1_feat = int(crop_y1 * feat_h / orig_h)
            # y2_feat = int(crop_y2 * feat_h / orig_h)
            # x1_feat = int(crop_x1 * feat_w / orig_w)
            # x2_feat = int(crop_x2 * feat_w / orig_w)
            # new_cropbox = []
            # new_cropbox.append(y1_feat)
            # new_cropbox.append(y2_feat)
            # new_cropbox.append(x1_feat)
            # new_cropbox.append(x2_feat)
            # scaled_feature = crop(mres_feats[0], new_cropbox)
            # size = (feat_h, feat_w)
            # upsampled_feature_map = F.interpolate(scaled_feature, size=size, mode='bilinear', align_corners=False)
            # residual_feature = torch.abs(self.sigmoid(mres_feats[1])- self.sigmoid(upsampled_feature_map))
            # to_pil_image = transforms.ToPILImage()
            # img_tensor = residual_feature[0]  # 选择第i个batch
            # img = to_pil_image(img_tensor)    # 转换为PIL图像
            # img.save('/gemini/code/HRDA-master/photo/1.png')  # 保存图像

            return mres_feats
        else:
            scaled_img = self.resize(img, self.feature_scale)
            return self.extract_unscaled_feat(scaled_img)

    def encode_decode(self, img, img_metas):

        # save_path = '/gemini/code/origin/origin.png'
        # base_path = save_path
        # index = 1
        # for img_tensor in img:
        #     while os.path.exists(save_path):
        #         save_path = f"{base_path.rsplit('.', 1)[0]}_{index}.{base_path.rsplit('.', 1)[1]}"
        #         index += 1
        #     to_pil_image = transforms.ToPILImage()
        #     img1 = to_pil_image(img_tensor)    # 转换为 PIL 图像
        #     img1.save(save_path)  # 保存图像
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        mres_feats = []
        self.decode_head.debug_output = {}
        for i, s in enumerate(self.scales):
            if s == 1 and self.blur_hr_crop:
                scaled_img = self.blur_downup(img)
            else:
                scaled_img = self.resize(img, s)
            if i >= 1 and self.hr_slide_inference:
                mres_feats.append(self.extract_slide_feat(scaled_img, mres_feats[0][3]))

            else:
                # save_path = '/gemini/code/resize/origin.png'
                # base_path = save_path
                # index = 1
                # for img_tensor in scaled_img:
                #     while os.path.exists(save_path):
                #         save_path = f"{base_path.rsplit('.', 1)[0]}_{index}.{base_path.rsplit('.', 1)[1]}"
                #         index += 1
                #     to_pil_image = transforms.ToPILImage()
                #     img1 = to_pil_image(img_tensor)    # 转换为 PIL 图像
                #     img1.save(save_path)  # 保存图像
                mres_feats.append(self.extract_unscaled_feat(scaled_img))
            if self.decode_head.debug:
                self.decode_head.debug_output[f'Img {i} Scale {s}'] = \
                    scaled_img.detach()
        out = self._decode_head_forward_test(mres_feats, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _forward_train_features(self, img):
        mres_feats = []
        self.decode_head.debug_output = {}
        assert len(self.scales) <= 2, 'Only up to 2 scales are supported.'
        prob_vis = None
        for i, s in enumerate(self.scales):
            if s == 1 and self.blur_hr_crop:
                scaled_img = self.blur_downup(img)
            else:
                scaled_img = resize(
                    input=img,
                    scale_factor=s,
                    mode='bilinear',
                    align_corners=self.align_corners)
            if self.crop_size is not None and i >= 1:
                crop_box = get_crop_bbox(*scaled_img.shape[-2:],
                                         self.crop_size,
                                         self.crop_coord_divisible)
                if self.feature_scale in self.feature_scale_all_strs:
                    HRDAEncoderDecoder.last_train_crop_box[i] = crop_box
                self.decode_head.set_hr_crop_box(crop_box)
                scaled_img = crop(scaled_img, crop_box)
            if self.decode_head.debug:
                self.decode_head.debug_output[f'Img {i} Scale {s}'] = \
                    scaled_img.detach()
            mres_feats.append(self.extract_unscaled_feat(scaled_img))
        b, c, orig_h, orig_w = img.shape
        # 多尺度特征 print(mres_feats[0][0].shape, mres_feats[0][1].shape, mres_feats[0][2].shape)
        b, c, feat_h, feat_w = mres_feats[0][3].shape
        b, c, feat_h2, feat_w2 = mres_feats[1][3].shape
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_box
        y1_feat = int(crop_y1 * feat_h / orig_h)
        y2_feat = int(crop_y2 * feat_h / orig_h)
        x1_feat = int(crop_x1 * feat_w / orig_w)
        x2_feat = int(crop_x2 * feat_w / orig_w)
        new_cropbox = []
        new_cropbox.append(y1_feat)
        new_cropbox.append(y2_feat)
        new_cropbox.append(x1_feat)
        new_cropbox.append(x2_feat)
        scaled_feature = crop(mres_feats[0][3], new_cropbox) #在特征层裁处对应crop区域
        size_lr_feat = (feat_h, feat_w)
        size_hr_feat = (feat_h2, feat_w2)
        upsampled_feature_map = F.interpolate(scaled_feature, size=size_hr_feat, mode='bilinear', align_corners=False)
        # residual_feature = torch.abs(self.sigmoid(mres_feats[1][0])- self.sigmoid(upsampled_feature_map))
        # img_tensor = residual_feature[0]  # 选择第 i 个 batch
        # to_pil_image = transforms.ToPILImage()
        # # 选择前3个通道作为RGB图像
        # img_tensor = img_tensor[:3, :, :]
        # img = to_pil_image(img_tensor)    # 转换为 PIL 图像
        # img.save('/gemini/code/HRDA-master/photo/1.png')  # 保存图像
        # 将批次图像转换为灰度图
        batch_gray_images = convert_to_grayscale(img)

        # 为批次中的每张图像构建拉普拉斯金字塔
        laplacian_pyramids = []
        for image in batch_gray_images:
            laplacian_pyramid = build_laplacian_pyramid(image, 1)
            # cv2.imwrite('/gemini/code/HRDA-master/photo/1.png', laplacian_pyramid)

            laplacian_pyramids.append(laplacian_pyramid)  # 只取第一层的拉普拉斯图像
        laplacian_pyramids_batch = np.stack(laplacian_pyramids, axis=0)  # (batch, height, width)

        layer_tensor = torch.tensor(laplacian_pyramids_batch, dtype=torch.float32) / 255.0  # 归一化到 [0, 1]
        layer_tensor = layer_tensor.unsqueeze(1).cuda()

        layer_tensor1 = crop(layer_tensor, crop_box)
        layer_tensor1 = F.interpolate(layer_tensor1, size=size_hr_feat, mode='bilinear', align_corners=False)
        # print(layer_tensor1.shape)
        # print(mres_feats[1][3].shape)
        # print(upsampled_feature_map.shape)
        mres_feats[1][3] = layer_tensor1 * mres_feats[1][3] + (1-layer_tensor1)* upsampled_feature_map
        # mres_feats[0][3] = layer_tensor2 * mres_feats[0][3] +  mres_feats[0][3]

#两种方案：一种加权融合，一种残差突出重点
        return mres_feats, prob_vis

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      top_50_percent_mask=None,
                      seg_weight=None,
                      return_feat=False
                      ):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()

        mres_feats, prob_vis = self._forward_train_features(img)
        for i, s in enumerate(self.scales):
            if return_feat and self.feature_scale in \
                    self.feature_scale_all_strs:
                if 'features' not in losses:
                    losses['features'] = []
                losses['features'].append(mres_feats[i])
            if return_feat and s == self.feature_scale:
                losses['features'] = mres_feats[i]
                break

        loss_decode = self._decode_head_forward_train(mres_feats, img_metas,
                                                      gt_semantic_seg,
                                                      top_50_percent_mask,
                                                      seg_weight)
        losses.update(loss_decode)

        if self.decode_head.debug and prob_vis is not None:
            self.decode_head.debug_output['Crop Prob.'] = prob_vis

        if self.with_auxiliary_head:
            raise NotImplementedError

        return losses

    def forward_with_aux(self, img, img_metas):
        assert not self.with_auxiliary_head
        mres_feats, _ = self._forward_train_features(img)
        out = self.decode_head.forward(mres_feats)
        # out = resize(
        #     input=out,
        #     size=img.shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        return {'main': out}
