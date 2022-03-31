import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
from models.feature_backbones.VGG_features import VGGPyramid
from models.utils.mod import FeatureL2Norm

def EPE(input_flow, target_flow, sparse=False, mean=True, sum=False):

    EPE_map = torch.norm(target_flow-input_flow, 2, 1)
    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)

        EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    elif sum:
        return EPE_map.sum()
    else:
        return EPE_map.sum()/batch_size


def L1_loss(input_flow, target_flow):
    L1 = torch.abs(input_flow-target_flow)
    L1 = torch.sum(L1, 1)
    return L1


def L1_charbonnier_loss(input_flow, target_flow, sparse=False, mean=True, sum=False):

    batch_size = input_flow.size(0)
    epsilon = 0.01
    alpha = 0.4
    L1 = L1_loss(input_flow, target_flow)
    norm = torch.pow(L1 + epsilon, alpha)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)

        norm = norm[~mask]
    if mean:
        return norm.mean()
    elif sum:
        return norm.sum()
    else:
        return norm.sum()/batch_size


def sparse_max_pool(input, size):
    '''Downsample the input by considering 0 values as invalid.
    Unfortunately, no generic interpolation mode can resize a sparse map correctly,
    the strategy here is to use max pooling for positive values and "min pooling"
    for negative values, the two results are then summed.
    This technique allows sparsity to be minized, contrary to nearest interpolation,
    which could potentially lose information for isolated data points.'''

    positive = (input > 0).float()
    negative = (input < 0).float()
    output = F.adaptive_max_pool2d(input * positive, size) - F.adaptive_max_pool2d(-input * negative, size)
    return output


def multiscaleEPE(network_output, target_flow, robust_L1_loss=False, mask=None, weights=None,
                  sparse=False, mean=False):
    '''
    here the ground truth flow is given at the higest resolution and it is just interpolated
    at the different sized (without rescaling it)
    :param network_output:
    :param target_flow:
    :param weights:
    :param sparse:
    :return:
    '''

    def one_scale(output, target, sparse, robust_L1_loss=False, mask=None, mean=False):
        b, _, h, w = output.size()
        if sparse:
            target_scaled = sparse_max_pool(target, (h, w))

            if mask is not None:
                mask = sparse_max_pool(mask.float().unsqueeze(1), (h, w))
        else:
            target_scaled = F.interpolate(target, (h, w), mode='bilinear')

            if mask is not None:
                # mask can be byte or float or uint8 or int
                # resize first in float, and then convert to byte/int to remove the borders which are values between 0 and 1
                mask = F.interpolate(mask.float().unsqueeze(1), (h, w), mode='bilinear').byte()

        if robust_L1_loss:
            if mask is not None:
                return L1_charbonnier_loss(output * mask.float(), target_scaled * mask.float(), sparse, mean=mean, sum=False)
            else:
                return L1_charbonnier_loss(output, target_scaled, sparse, mean=mean, sum=False)
        else:
            if mask is not None:
                return EPE(output * mask.float(), target_scaled * mask.float(), sparse, mean=mean, sum=False)
            else:
                return EPE(output, target_scaled, sparse, mean=mean, sum=False)

    if type(network_output) not in [tuple, list]:
        network_output = [network_output]
    if weights is None:
        weights = [0.32, 0.08, 0.02, 0.01, 0.005]  # as in original article
    assert(len(weights) == len(network_output))

    loss = 0
    for output, weight in zip(network_output, weights):
        # from smallest size to biggest size (last one is a quarter of input image size
        loss += weight * one_scale(output, target_flow, sparse, robust_L1_loss=robust_L1_loss, mask=mask, mean=mean)
    return loss


def realEPE(output, target, mask_gt, ratio_x=None, ratio_y=None, sparse=False, mean=True, sum=False):
    '''
    in this case real EPE, the network output is upsampled to the size of
    the target (without scaling) because it was trained without the scaling, it should be equal to target flow
    mask_gt can be uint8 tensor or byte or int
    :param output:
    :param target: flow in range [0, w-1]
    :param sparse:
    :return:
    '''
    # mask_gt in shape bxhxw, can be torch.byte or torch.uint8 or torch.int
    b, _, h, w = target.size()
    if ratio_x is not None and ratio_y is not None:
        upsampled_output = F.interpolate(output, (h,w), mode='bilinear', align_corners=False)
        upsampled_output[:,0,:,:] *= ratio_x
        upsampled_output[:,1,:,:] *= ratio_y
    else:
        upsampled_output = F.interpolate(output, (h, w), mode='bilinear', align_corners=False)
    # output interpolated to original size (supposed to be in the right range then)

    flow_target_x = target.permute(0, 2, 3, 1)[:, :, :, 0]
    flow_target_y = target.permute(0, 2, 3, 1)[:, :, :, 1]
    flow_est_x = upsampled_output.permute(0, 2, 3, 1)[:, :, :, 0]  # BxH_xW_
    flow_est_y = upsampled_output.permute(0, 2, 3, 1)[:, :, :, 1]

    flow_target = \
        torch.cat((flow_target_x[mask_gt.bool()].unsqueeze(1),
                   flow_target_y[mask_gt.bool()].unsqueeze(1)), dim=1)
    flow_est = \
        torch.cat((flow_est_x[mask_gt.bool()].unsqueeze(1),
                   flow_est_y[mask_gt.bool()].unsqueeze(1)), dim=1)
    return EPE(flow_est, flow_target, sparse, mean=mean, sum=sum)



def cal_IOU(pred_label,gt_label):
    b,c,h,w = pred_label.shape
    # print("gt_label.max:",gt_label.max())

    smooth = 1.0  # may change
    # smooth的目的是为了防止score除数为0,如果有为0的情况可以改成smooth=1或者其他
    if b>1:
        i = torch.sum(gt_label)
        j = torch.sum(pred_label)
        intersection = torch.sum(gt_label * pred_label)
    else:
        i = gt_label.sum(1).sum(1).sum(1)
        j = pred_label.sum(1).sum(1).sum(1)
        intersection = (gt_label * pred_label).sum(1).sum(1).sum(1)
    # print("intersection:",intersection)
    iou = (intersection + smooth) / (i + j - intersection + smooth)  # iou
    return -iou.mean()


def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow

    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = Variable(grid) + flo
    # makes a mapping out of the flow

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output * mask


def multiscale_IOU(source_label,target_label, network_output, target_flow, weights=None, mask=None, sparse=False):
    def one_IOU(source_label,target_label, output, target_flow, sparse, mask=mask):
        # import cv2
        # warpped_sat_label_gt = warp(source_label,target_flow)
        # print("warpped_uav_label_gt_noscale.shape:", warpped_sat_label_gt.shape)  # torch.Size([1, 2, 64, 64])
        # cv2.imwrite("debug_semantic/sat_label_gt_noscale.jpg",source_label.data.cpu().numpy()[0].transpose(1, 2, 0) * 255)
        # cv2.imwrite("debug_semantic/warpped_sat_label_gt_noscale.jpg",warpped_sat_label_gt.data.cpu().numpy()[0].transpose(1, 2, 0) * 255)

        # downsample the source_label, target_label and target_flow to 32*32 or 64*64 or
        b, _, h, w = output.size()

        if sparse:
            target_label_scaled = sparse_max_pool(target_label, (h, w))

            if mask is not None:
                mask = sparse_max_pool(mask.float().unsqueeze(1), (h, w))
        else:
            source_label_scaled = F.interpolate(source_label, (h, w), mode='bilinear')
            target_label_scaled = F.interpolate(target_label, (h, w), mode='bilinear')
            output_flow_scaled = F.interpolate(output, (h, w), mode='bilinear')
            target_flow_scaled = F.interpolate(target_flow, (h, w), mode='bilinear')
            # print("source_label_scaled.shape:", source_label_scaled.shape)  # torch.Size([1, 2, 64, 64])
            # print("target_label_scaled.shape:", target_label_scaled.shape)  # torch.Size([1, 2, 64, 64])
            # print("output_flow_scaled.shape:", output_flow_scaled.shape)  # torch.Size([1, 2, 64, 64])
            # print("target_flow_scaled.shape:", target_flow_scaled.shape)  # torch.Size([1, 2, 64, 64])

            from dataset_utils.dataset_io import boolean_string, writeFlow

            # writeFlow(target_flow_scaled[0].permute(1,2,0).cpu().numpy(), 'debug_semantic/scaled_sat2uav_flow.flo', "")
            # writeFlow(target_flow[0].permute(1,2,0).cpu().numpy(), 'debug_semantic/original_sat2uav_flow.flo', "")
            if mask is not None:
                # mask can be byte or float or uint8 or int
                # resize first in float, and then convert to byte/int to remove the borders which are values between 0 and 1
                mask = F.interpolate(mask.float().unsqueeze(1), (h, w), mode='bilinear').byte()

        downsample_warpped_sat_label_pred = warp(source_label_scaled,output_flow_scaled)
        # print("warpped_sat_label_pred.shape:", downsample_warpped_sat_label_pred.shape)  # torch.Size([1, 2, 64, 64])
        downsample_warpped_sat_label_gt = warp(source_label_scaled,target_flow_scaled)
        # print("warpped_uav_label_gt.shape:", warpped_sat_label_gt.shape)  # torch.Size([1, 2, 64, 64])
        # cv2.imwrite("debug_semantic/sat_label_gt.jpg",cv2.resize(source_label_scaled.data.cpu().numpy()[0].transpose(1,2,0)*255,(256,256)))
        # cv2.imwrite("debug_semantic/uav_label_gt.jpg",cv2.resize(target_label_scaled.data.cpu().numpy()[0].transpose(1,2,0)*255,(256,256)))
        # cv2.imwrite("debug_semantic/warpped_sat_label_pred_64.jpg",downsample_warpped_sat_label_pred.data.cpu().numpy()[0].transpose(1,2,0)*255)
        # cv2.imwrite("debug_semantic/warpped_sat_label_pred.jpg",cv2.resize(downsample_warpped_sat_label_pred.data.cpu().numpy()[0].transpose(1,2,0)*255,(256,256)))
        # cv2.imwrite("debug_semantic/warpped_sat_label_gt.jpg",cv2.resize(downsample_warpped_sat_label_gt.data.cpu().numpy()[0].transpose(1,2,0)*255,(256,256)))
        # print(warpped_uav_label_gt)
        downsample_loss = cal_IOU(downsample_warpped_sat_label_pred,downsample_warpped_sat_label_gt)
        # print("downsample_loss:",downsample_loss)
        loss = ori_loss + downsample_loss
        return 10000 * loss

    loss = 0

    # upsample the output flow to 256*256
    original_warpped_sat_label_gt = warp(source_label, target_flow)
    ori_b, ori_, ori_h, ori_w = target_flow.size()
    pred_flow = F.interpolate(source_label, (ori_h, ori_w), mode='bilinear')
    original_warpped_sat_label_pred = warp(source_label, pred_flow)
    ori_loss = cal_IOU(original_warpped_sat_label_pred, original_warpped_sat_label_gt)
    # print("original_loss:", ori_loss)

    loss += ori_loss
    for output, weight in zip(network_output, weights):
        # from smallest size to biggest size (last one is a quarter of input image size
        loss += weight * one_IOU(source_label,target_label,output, target_flow, sparse, mask=mask)
    return loss



def L2_loss(a,b):
    l2 = torch.abs(a - b)
    l2 = torch.sum(l2)
    return l2


def multiscale_feature(image1,image2):
    pyramid = VGGPyramid().cuda()
    l2norm = FeatureL2Norm()
    loss = 0
    im1_pyr = pyramid(image1, eigth_resolution=True)
    im2_pyr = pyramid(image2, eigth_resolution=True)
    for i in range(len(im1_pyr)):
        loss += L2_loss(l2norm(im1_pyr[i]),l2norm(im2_pyr[i]))

    del image1,image2
    return 0.0001 * loss


"""
Normalized local cross-correlation function in Pytorch. Modified from https://github.com/voxelmorph/voxelmorph.
"""
class NCC_Loss(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def __init__(self, win=5, eps=1e-5):
        super(NCC_Loss, self).__init__()
        self.win_raw = win
        self.eps = eps
        self.win = win

    def forward(self, I, J):
        ndims = 3
        win_size = self.win_raw
        self.win = [self.win_raw] * ndims

        weight_win_size = self.win_raw
        weight = torch.ones((1, 3, weight_win_size, weight_win_size), device=I.device, requires_grad=False)
        conv_fn = F.conv2d

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size/2))
        J_sum = conv_fn(J, weight, padding=int(win_size/2))
        I2_sum = conv_fn(I2, weight, padding=int(win_size/2))
        J2_sum = conv_fn(J2, weight, padding=int(win_size/2))
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size/2))

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return -1000 * torch.mean(cc)