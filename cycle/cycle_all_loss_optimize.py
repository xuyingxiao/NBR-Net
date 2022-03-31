import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from dataset_utils.pixel_wise_mapping import remap_using_flow_fields
from cycle.loss import multiscaleEPE, realEPE, sparse_max_pool, multiscale_IOU, multiscale_feature,NCC_Loss
from matplotlib import pyplot as plt


def pre_process_data(source_img, target_img, device):
    '''
    Pre-processes source and target images before passing it to the network
    :param source_img: Torch tensor Bx3xHxW
    :param target_img: Torch tensor Bx3xHxW
    :param device: cpu or gpu
    :return:
    source_img_copy: Torch tensor Bx3xHxW, source image scaled to 0-1 and mean-centered and normalized
                     using mean and standard deviation of ImageNet
    target_img_copy: Torch tensor Bx3xHxW, target image scaled to 0-1 and mean-centered and normalized
                     using mean and standard deviation of ImageNet
    source_img_256: Torch tensor Bx3x256x256, source image rescaled to 256x256, scaled to 0-1 and mean-centered and normalized
                    using mean and standard deviation of ImageNet
    target_img_256: Torch tensor Bx3x256x256, target image rescaled to 256x256, scaled to 0-1 and mean-centered and normalized
                    using mean and standard deviation of ImageNet
    '''
    # img has shape bx3xhxw
    b, _, h_scale, w_scale = target_img.shape
    mean_vector = np.array([0.485, 0.456, 0.406])
    std_vector = np.array([0.229, 0.224, 0.225])

    # original resolution
    source_img_copy = source_img.float().to(device).div(255.0)
    target_img_copy = target_img.float().to(device).div(255.0)
    mean = torch.as_tensor(mean_vector, dtype=source_img_copy.dtype, device=source_img_copy.device)
    std = torch.as_tensor(std_vector, dtype=source_img_copy.dtype, device=source_img_copy.device)
    source_img_copy.sub_(mean[:, None, None]).div_(std[:, None, None])
    target_img_copy.sub_(mean[:, None, None]).div_(std[:, None, None])

    # resolution 256x256
    source_img_256 = torch.nn.functional.interpolate(input=source_img.float().to(device),
                                                     size=(256, 256),
                                                     mode='area').byte()
    target_img_256 = torch.nn.functional.interpolate(input=target_img.float().to(device),
                                                     size=(256, 256),
                                                     mode='area').byte()

    source_img_256 = source_img_256.float().div(255.0)
    target_img_256 = target_img_256.float().div(255.0)
    source_img_256.sub_(mean[:, None, None]).div_(std[:, None, None])
    target_img_256.sub_(mean[:, None, None]).div_(std[:, None, None])

    return source_img_copy, target_img_copy, source_img_256, target_img_256


def pre_process_label(source_lbl, target_lbl, device):
    '''
    Pre-processes source and target labels before passing it to the network
    :param source_lbl: Torch tensor Bx1xHxW
    :param target_lbl: Torch tensor Bx1xHxW
    :param device: cpu or gpu
    :return:
    source_lbl_copy: Torch tensor Bx1xHxW, source label scaled to 0-1
    target_lbl_copy: Torch tensor Bx1xHxW, target label scaled to 0-1
    source_lbl_256: Torch tensor Bx1x256x256, source label rescaled to 256x256,
    target_lbl_256: Torch tensor Bx1x256x256, target label rescaled to 256x256, scaled to 0-1
    '''
    b, _, h_scale, w_scale = target_lbl.shape
    # original resolution
    source_lbl_copy = source_lbl.float().to(device).div(255.0)
    target_lbl_copy = target_lbl.float().to(device).div(255.0)
    # print("sat_label_max:",source_lbl_copy.max())
    # resolution 256x256
    source_lbl_256 = torch.nn.functional.interpolate(input=source_lbl.float().to(device),
                                                     size=(256, 256),
                                                     mode='area').byte()
    target_lbl_256 = torch.nn.functional.interpolate(input=target_lbl.float().to(device),
                                                     size=(256, 256),
                                                     mode='area').byte()

    source_lbl_256 = source_lbl_256.float().div(255.0)
    target_lbl_256 = target_lbl_256.float().div(255.0)
    # print("sat_label_256_max:", source_lbl_256.max())
    return source_lbl_copy, target_lbl_copy, source_lbl_256, target_lbl_256


def single_to_channels(gray):
    image = np.zeros((gray.shape[0], gray.shape[1], 3))
    image[:, :, 0] = gray
    image[:, :, 1] = gray
    image[:, :, 2] = gray
    return image


def plot_image(image):
    mean_values = torch.tensor([0.485, 0.456, 0.406], dtype=image.dtype).view(3, 1, 1)
    std_values = torch.tensor([0.229, 0.224, 0.225], dtype=image.dtype).view(3, 1, 1)
    image = (image.detach()[0].cpu() * std_values + mean_values).clamp(0, 1).permute(1, 2, 0)
    return image


def cycle_semantic_plot_during_training(save_path, epoch, batch, apply_mask, h_original, w_original, h_256, w_256,
                                        div_flow,
                                        source_image, target_image, warpped1_source_image_gt_original,
                                        warpped1_source_image_pred_original, warpped2_source_image_gt_original,
                                        warpped2_source_image_pred_original,
                                        source_label, target_label, warpped1_source_label_gt_original,
                                        warpped1_source_label_pred_original, warpped2_source_label_gt_original,
                                        warpped2_source_label_pred_original,
                                        source_image_256, target_image_256, warpped1_source_image_gt_256,
                                        warpped1_source_image_pred_256, warpped2_source_image_gt_256,
                                        warpped2_source_image_pred_256,
                                        source_label_256, target_label_256, warpped1_source_label_gt_256,
                                        warpped1_source_label_pred_256, warpped2_source_label_gt_256,
                                        warpped2_source_label_pred_256,
                                        step1_EPE, step1_EPE_256, step2_EPE, step2_EPE_256, sat2uav=True, train=True,
                                        mask=None, mask_256=None):
    fig, axis = plt.subplots(4, 6, figsize=(20, 20))
    axis[0][0].imshow(plot_image(source_image))
    axis[0][0].set_title("original reso: \nsat image")
    axis[0][1].imshow(plot_image(target_image))
    axis[0][1].set_title("original reso: \nuav image")
    axis[0][2].imshow(plot_image(warpped1_source_image_gt_original))
    axis[0][2].set_title("original reso: \nwarpped sat image with GT")
    axis[0][3].imshow(plot_image(warpped1_source_image_pred_original))
    axis[0][3].set_title("original reso %s: \nwarpped sat image with network" % str(step1_EPE.data.cpu().numpy()))
    axis[0][4].imshow(plot_image(warpped2_source_image_gt_original))
    axis[0][4].set_title("original reso : \nwarpped uav image with GT")
    axis[0][5].imshow(plot_image(warpped2_source_image_pred_original))
    axis[0][5].set_title("original reso %s: \nwarpped uav image with network" % str(step2_EPE.data.cpu().numpy()))

    axis[1][0].imshow(single_to_channels(source_label.data.cpu().numpy()[0].transpose(1, 2, 0)[:, :, 0] * 1.))
    axis[1][0].set_title("original reso: \nsat label")
    axis[1][1].imshow(single_to_channels(target_label.data.cpu().numpy()[0].transpose(1, 2, 0)[:, :, 0] * 1.))
    axis[1][1].set_title("original reso: \nuav label")
    axis[1][2].imshow(
        single_to_channels(warpped1_source_label_gt_original.data.cpu().numpy()[0].transpose(1, 2, 0)[:, :, 0] * 1.))
    axis[1][2].set_title("original reso : \nwarpped sat label with GT")
    axis[1][3].imshow(
        single_to_channels(warpped1_source_label_pred_original.data.cpu().numpy()[0].transpose(1, 2, 0)[:, :, 0] * 1.))
    axis[1][3].set_title("original reso: \nwarpped sat label with network")
    axis[1][4].imshow(
        single_to_channels(warpped2_source_label_gt_original.data.cpu().numpy()[0].transpose(1, 2, 0)[:, :, 0] * 1.))
    axis[1][4].set_title("original reso : \nwarpped uav label with GT")
    axis[1][5].imshow(
        single_to_channels(warpped2_source_label_pred_original.data.cpu().numpy()[0].transpose(1, 2, 0)[:, :, 0] * 1.))
    axis[1][5].set_title("original reso: \nwarpped uav label with network")

    axis[2][0].imshow(plot_image(source_image_256))
    axis[2][0].set_title("reso 256*256: \nsat image")
    axis[2][1].imshow(plot_image(target_image_256))
    axis[2][1].set_title("reso 256*256: \nuav image")
    axis[2][2].imshow(plot_image(warpped1_source_image_gt_256))
    axis[2][2].set_title("reso 256*256: \nwarpped sat image with GT")
    axis[2][3].imshow(plot_image(warpped1_source_image_pred_256))
    axis[2][3].set_title("reso 256*256 %s: \nwarpped sat image with network" % str(step1_EPE_256.data.cpu().numpy()))
    axis[2][4].imshow(plot_image(warpped2_source_image_gt_256))
    axis[2][4].set_title("reso 256*256: \nwarpped uav image with GT")
    axis[2][5].imshow(plot_image(warpped2_source_image_pred_256))
    axis[2][5].set_title("reso 256*256 %s: \nwarpped uav image with network" % str(step2_EPE_256.data.cpu().numpy()))

    axis[3][0].imshow(single_to_channels(source_label_256.data.cpu().numpy()[0].transpose(1, 2, 0)[:, :, 0] * 1.))
    axis[3][0].set_title("reso 256*256: \nsat label")
    axis[3][1].imshow(single_to_channels(target_label_256.data.cpu().numpy()[0].transpose(1, 2, 0)[:, :, 0] * 1.))
    axis[3][1].set_title("reso 256*256: \nuav label")
    axis[3][2].imshow(
        single_to_channels(warpped1_source_label_gt_256.data.cpu().numpy()[0].transpose(1, 2, 0)[:, :, 0] * 1.))
    axis[3][2].set_title("reso 256*256 : \nwarpped sat label with GT")
    axis[3][3].imshow(
        single_to_channels(warpped1_source_label_pred_256.data.cpu().numpy()[0].transpose(1, 2, 0)[:, :, 0] * 1.))
    axis[3][3].set_title("reso 256*256: \nwarpped sat label with network")
    axis[3][4].imshow(
        single_to_channels(warpped2_source_label_gt_256.data.cpu().numpy()[0].transpose(1, 2, 0)[:, :, 0] * 1.))
    axis[3][4].set_title("reso 256*256 : \nwarpped uav label with GT")
    axis[3][5].imshow(
        single_to_channels(warpped2_source_label_pred_256.data.cpu().numpy()[0].transpose(1, 2, 0)[:, :, 0] * 1.))
    axis[3][5].set_title("reso 256*256: \nwarpped uav label with network")
    if sat2uav:
        fig.savefig('{}/epoch{}_batch{}_sat2uav.png'.format(save_path, epoch, batch),
                    bbox_inches='tight')
    else:
        fig.savefig('{}/epoch{}_batch{}_uav2sat.png'.format(save_path, epoch, batch),
                    bbox_inches='tight')

    # print('{}/epoch{}_batch{}_uav2sat.png'.format(save_path, epoch, batch))
    plt.close(fig)


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


def train_source2target(net, save_path, epoch, i, apply_mask, div_flow,
                        target_image, source_image, target_image_256, source_image_256,
                        target_label, source_label, target_label_256, source_label_256,
                        step1_flow_gt_original, step2_flow_gt_original,
                        step1_mask_gt, step2_mask_gt,
                        step1_EPE_array, step2_EPE_array,
                        sparse=False, train=True, sat2uav=True,
                        loss_grid_weights=[0.32, 0.08, 0.02, 0.01], robust_L1_loss=False):
    # at 256x256 resolution, b, _, 256, 256
    bs, _, h_original, w_original = step1_flow_gt_original.shape
    step1_flow_gt_256 = F.interpolate(step1_flow_gt_original, (256, 256), mode='bilinear', align_corners=False)
    step1_flow_gt_256[:, 0, :, :] *= 256.0 / float(w_original)
    step1_flow_gt_256[:, 1, :, :] *= 256.0 / float(h_original)
    step2_flow_gt_256 = F.interpolate(step2_flow_gt_original, (256, 256), mode='bilinear', align_corners=False)
    step2_flow_gt_256[:, 0, :, :] *= 256.0 / float(w_original)
    step2_flow_gt_256[:, 1, :, :] *= 256.0 / float(h_original)

    step1_output_net_256, step1_output_net_original = net(target_image, source_image,
                                                          target_image_256, source_image_256)
    weights_original = loss_grid_weights[-len(step1_output_net_original):]
    weights_256 = loss_grid_weights[:len(step1_output_net_256)]

    # interpolate
    warpped1_source_image_pred_original = warp(source_image,
                                               F.interpolate(step1_output_net_original[-1], (h_original, w_original),
                                                             mode='bilinear', align_corners=False))
    warpped1_source_image_gt_original = warp(source_image,
                                             F.interpolate(step1_flow_gt_original, (h_original, w_original),
                                                           mode='bilinear', align_corners=False))
    warpped1_source_image_pred_256 = warp(source_image_256,
                                          F.interpolate(step1_output_net_256[-1], (256, 256), mode='bilinear',
                                                        align_corners=False))
    warpped1_source_image_gt_256 = warp(source_image_256, F.interpolate(step1_flow_gt_256, (256, 256), mode='bilinear',
                                                                    align_corners=False))

    warpped1_source_label_pred_original = warp(source_label,
                                               F.interpolate(step1_output_net_original[-1], (h_original, w_original),
                                                             mode='bilinear', align_corners=False))
    warpped1_source_label_gt_original = warp(source_label,
                                             F.interpolate(step1_flow_gt_original, (h_original, w_original),
                                                           mode='bilinear', align_corners=False))
    warpped1_source_label_pred_256 = warp(source_label_256,
                                          F.interpolate(step1_output_net_256[-1], (256, 256), mode='bilinear',
                                                        align_corners=False))
    warpped1_source_label_gt_256 = warp(source_label_256, F.interpolate(step1_flow_gt_256, (256, 256), mode='bilinear',
                                                                    align_corners=False))

    if not train:
        # source2target calculating the validation EPE
        for index_reso_original in range(len(step1_output_net_original)):
            step1_EPE = div_flow * realEPE(step1_output_net_original[-(index_reso_original + 1)],
                                           step1_flow_gt_original, step1_mask_gt, sparse=sparse)
            step1_EPE_array[index_reso_original, i] = step1_EPE

        for index_reso_256 in range(len(step1_output_net_256)):
            step1_EPE_256 = div_flow * realEPE(step1_output_net_256[-(index_reso_256 + 1)],
                                               step1_flow_gt_original, step1_mask_gt,
                                               ratio_x=float(w_original) / float(256.0),
                                               ratio_y=float(h_original) / float(256.0),
                                               sparse=sparse)
            step1_EPE_array[(len(step1_output_net_original) + index_reso_256), i] = step1_EPE_256
    else:
        step1_EPE = realEPE(step1_output_net_original[-1],
                            step1_flow_gt_original, step1_mask_gt, sparse=sparse)
        step1_EPE_256 = div_flow * realEPE(step1_output_net_256[-1],
                                           step1_flow_gt_original, step1_mask_gt,
                                           ratio_x=float(w_original) / float(256.0),
                                           ratio_y=float(h_original) / float(256.0),
                                           sparse=sparse)

    # step1_loss
    # multiscale epe and iou(label)
    EPE_loss_1 = multiscaleEPE(step1_output_net_original, step1_flow_gt_original,
                               weights=weights_original, sparse=False, mean=False,  # mask=step1_mask_gt,
                               robust_L1_loss=robust_L1_loss)
    EPE_loss_1 += multiscaleEPE(step1_output_net_256, step1_flow_gt_256,
                                weights=weights_256, sparse=False, mean=False,  # mask=step1_mask_gt,
                                robust_L1_loss=robust_L1_loss)
    IOU_loss_1 = multiscale_IOU(source_label, target_label, step1_output_net_original, step1_flow_gt_original,
                                weights=weights_original)  # , mask=step1_mask_gt)
    IOU_loss_1 += multiscale_IOU(source_label_256, target_label_256, step1_output_net_256, step1_flow_gt_256,
                                 weights=weights_256)  # , mask=step1_mask_gt)

    total_loss_1 = EPE_loss_1 + IOU_loss_1

    step2_target_image = warp(target_image, step2_flow_gt_original)
    step2_target_image_256 = warp(target_image_256, step2_flow_gt_256)

    # step2_output_net_256, step2_output_net_original = net(source_image, warpped1_source_image_pred_original,
    #                                                       source_image_256, warpped1_source_image_pred_256)

    step2_output_net_256, step2_output_net_original = net(step2_target_image, warpped1_source_image_pred_original,
                                                          step2_target_image_256, warpped1_source_image_pred_256)

    warpped2_source_image_pred_original = warp(warpped1_source_image_pred_original,
                                               F.interpolate(step2_output_net_original[-1], (h_original, w_original),
                                                             mode='bilinear', align_corners=False))
    warpped2_source_image_gt_original = warp(warpped1_source_image_gt_original,
                                             F.interpolate(step2_flow_gt_original, (h_original, w_original),
                                                           mode='bilinear', align_corners=False))
    warpped2_source_image_pred_256 = warp(warpped1_source_image_pred_256,
                                          F.interpolate(step2_output_net_256[-1], (256, 256), mode='bilinear',
                                                        align_corners=False))
    warpped2_source_image_gt_256 = warp(warpped1_source_image_gt_256,
                                        F.interpolate(step2_flow_gt_256, (256, 256), mode='bilinear',
                                                      align_corners=False))

    warpped2_source_label_pred_original = warp(warpped1_source_label_pred_original,
                                               F.interpolate(step2_output_net_original[-1], (h_original, w_original),
                                                             mode='bilinear', align_corners=False))
    warpped2_source_label_gt_original = warp(warpped1_source_label_gt_original,
                                             F.interpolate(step2_flow_gt_original, (h_original, w_original),
                                                           mode='bilinear', align_corners=False))
    warpped2_source_label_pred_256 = warp(warpped1_source_label_pred_256,
                                          F.interpolate(step2_output_net_256[-1], (256, 256), mode='bilinear',
                                                        align_corners=False))
    warpped2_source_label_gt_256 = warp(warpped1_source_label_gt_256,
                                        F.interpolate(step2_flow_gt_256, (256, 256), mode='bilinear',
                                                      align_corners=False))

    if not train:
        # target2source calculating the validation EPE
        for index_reso_original in range(len(step2_output_net_original)):
            step2_EPE = div_flow * realEPE(step2_output_net_original[-(index_reso_original + 1)],
                                           step2_flow_gt_original, step2_mask_gt, sparse=sparse)
            step2_EPE_array[index_reso_original, i] = step2_EPE

        for index_reso_256 in range(len(step2_output_net_256)):
            step2_EPE_256 = div_flow * realEPE(step2_output_net_256[-(index_reso_256 + 1)],
                                               step2_flow_gt_original, step2_mask_gt,
                                               ratio_x=float(w_original) / float(256.0),
                                               ratio_y=float(h_original) / float(256.0),
                                               sparse=sparse)
            step2_EPE_array[(len(step2_output_net_original) + index_reso_256), i] = step2_EPE_256
    else:
        step2_EPE = realEPE(step2_output_net_original[-1],
                            step2_flow_gt_original, step2_mask_gt, sparse=sparse)
        step2_EPE_256 = realEPE(step2_output_net_256[-1],
                                step2_flow_gt_original, step2_mask_gt,
                                ratio_x=float(w_original) / float(256.0),
                                ratio_y=float(h_original) / float(256.0),
                                sparse=sparse)
    # step2_loss
    # multiscale epe and iou(label)
    EPE_loss_2 = multiscaleEPE(step2_output_net_original, step2_flow_gt_original,  # mask=step2_mask_gt,
                               weights=weights_original, sparse=False, mean=False, robust_L1_loss=robust_L1_loss)
    EPE_loss_2 += multiscaleEPE(step2_output_net_256, step2_flow_gt_256,  # mask=step2_mask_gt,
                                weights=weights_256, sparse=False, mean=False, robust_L1_loss=robust_L1_loss)
    IOU_loss_2 = multiscale_IOU(target_label, source_label, step2_output_net_original, step2_flow_gt_original,
                                weights=weights_original)  # , mask=step2_mask_gt)
    IOU_loss_2 += multiscale_IOU(target_label_256, source_label_256, step2_output_net_256, step2_flow_gt_256,
                                 weights=weights_256)  # , mask=step2_mask_gt)

    total_loss_2 = EPE_loss_2 + IOU_loss_2
    loss = total_loss_1 + total_loss_2
    if not train:
        if epoch is None:
            cycle_semantic_plot_during_training(save_path, epoch, i, apply_mask, h_original, w_original, 256, 256, div_flow,
                                                source_image, target_image, warpped1_source_image_gt_original,
                                                warpped1_source_image_pred_original, warpped2_source_image_gt_original,
                                                warpped2_source_image_pred_original,
                                                source_label, target_label, warpped1_source_label_gt_original,
                                                warpped1_source_label_pred_original, warpped2_source_label_gt_original,
                                                warpped2_source_label_pred_original,
                                                source_image_256, target_image_256, warpped1_source_image_gt_256,
                                                warpped1_source_image_pred_256, warpped2_source_image_gt_256,
                                                warpped2_source_image_pred_256,
                                                source_label_256, target_label_256, warpped1_source_label_gt_256,
                                                warpped1_source_label_pred_256, warpped2_source_label_gt_256,
                                                warpped2_source_label_pred_256,
                                                step1_EPE, step1_EPE_256, step2_EPE, step2_EPE_256, sat2uav=sat2uav,
                                                train=train, mask=None, mask_256=None)
        elif i < 6:  # log first output of first batches
            # must be both in shape Bx2xHxW
            cycle_semantic_plot_during_training(save_path, epoch, i, apply_mask, h_original, w_original, 256, 256, div_flow,
                                                source_image, target_image, warpped1_source_image_gt_original,
                                                warpped1_source_image_pred_original, warpped2_source_image_gt_original,
                                                warpped2_source_image_pred_original,
                                                source_label, target_label, warpped1_source_label_gt_original,
                                                warpped1_source_label_pred_original, warpped2_source_label_gt_original,
                                                warpped2_source_label_pred_original,
                                                source_image_256, target_image_256, warpped1_source_image_gt_256,
                                                warpped1_source_image_pred_256, warpped2_source_image_gt_256,
                                                warpped2_source_image_pred_256,
                                                source_label_256, target_label_256, warpped1_source_label_gt_256,
                                                warpped1_source_label_pred_256, warpped2_source_label_gt_256,
                                                warpped2_source_label_pred_256,
                                                step1_EPE, step1_EPE_256, step2_EPE, step2_EPE_256, sat2uav=sat2uav,
                                                train=train, mask=None, mask_256=None)

    del source_image, target_image, source_image_256, target_image_256, \
        source_label, target_label, source_label_256, target_label_256, \
        warpped1_source_image_pred_original, warpped1_source_image_gt_original, warpped1_source_image_pred_256, warpped1_source_image_gt_256, \
        warpped1_source_label_pred_original, warpped1_source_label_gt_original, warpped1_source_label_pred_256, warpped1_source_label_gt_256, \
        warpped2_source_label_pred_original, warpped2_source_label_gt_original, warpped2_source_label_pred_256, warpped2_source_label_gt_256, \
        # step1_flow_gt_original, step1_flow_gt_256,step1_output_net_original, step1_output_net_256,\
    # step2_flow_gt_original, step2_flow_gt_256,step2_output_net_original, step2_output_net_256,
    torch.cuda.empty_cache()

    if train:
        return loss, EPE_loss_1, EPE_loss_2, IOU_loss_1, IOU_loss_2
    else:
        return step1_EPE_array, step2_EPE_array, loss, EPE_loss_1, EPE_loss_2, IOU_loss_1, IOU_loss_2


def train_epoch(net,
                optimizer,
                train_loader,
                device,
                epoch,
                train_writer,
                div_flow=1.0,
                save_path=None,
                loss_grid_weights=None,
                apply_mask=False,
                robust_L1_loss=False,
                sparse=False):

    net.train()

    running_EPE_loss = 0
    running_IOU_loss = 0
    running_feature_loss = 0
    running_corr_loss = 0
    running_total_loss = 0

    pbar = tqdm(enumerate(train_loader), total=len(train_loader),ncols=500)
    for i, mini_batch in pbar:
        optimizer.zero_grad()

        # # # pre-process the data
        #sat2uav
        sat_image, uav_image, sat_image_256, uav_image_256 = pre_process_data(mini_batch['sat_image'],
                                                                              mini_batch['uav_image'],
                                                                              device=device)
        sat_label, uav_label, sat_label_256, uav_label_256 = pre_process_label(mini_batch['sat_label'],
                                                                              mini_batch['uav_label'],
                                                                              device=device)
        sat2uav_flow_gt_original = mini_batch['sat2uav_flow'].to(device)
        uav2sat_flow_gt_original = mini_batch['uav2sat_flow'].to(device)

        sat2uav_mask_gt = mini_batch['sat2uav_mask'].to(device)
        uav2sat_mask_gt = mini_batch['uav2sat_mask'].to(device)

        # # pre-process the data
        # # uav2sat
        # sat_image, uav_image, sat_image_256, uav_image_256 = pre_process_data(mini_batch['uav_image'],
        #                                                                       mini_batch['sat_image'],
        #                                                                       device=device)
        # sat_label, uav_label, sat_label_256, uav_label_256 = pre_process_label(mini_batch['uav_label'],
        #                                                                        mini_batch['sat_label'],
        #                                                                        device=device)
        # sat2uav_flow_gt_original = mini_batch['uav2sat_flow'].to(device)
        # uav2sat_flow_gt_original = mini_batch['sat2uav_flow'].to(device)
        #
        # sat2uav_mask_gt = mini_batch['uav2sat_mask'].to(device)
        # uav2sat_mask_gt = mini_batch['sat2uav_mask'].to(device)

        sat2uav_loss, sat2uav_EPE_loss_1, sat2uav_EPE_loss_2, sat2uav_IOU_loss_1, sat2uav_IOU_loss_2 = \
            train_source2target(net, save_path, epoch, i, apply_mask, div_flow,
                                uav_image, sat_image, uav_image_256, sat_image_256,
                                uav_label, sat_label, uav_label_256, sat_label_256,
                                sat2uav_flow_gt_original, uav2sat_flow_gt_original,
                                sat2uav_mask_gt, uav2sat_mask_gt,
                                step1_EPE_array=None, step2_EPE_array=None,
                                sparse=sparse, train=True, sat2uav=True,
                                loss_grid_weights=loss_grid_weights, robust_L1_loss=robust_L1_loss)

        running_EPE_loss += sat2uav_EPE_loss_1.item() + sat2uav_EPE_loss_2.item()
        running_IOU_loss += sat2uav_IOU_loss_1.item() + sat2uav_IOU_loss_2.item()
        running_total_loss += sat2uav_loss#.item()

        sat2uav_loss.backward()
        optimizer.step()

        pbar.set_description(
            'sat2uav_total_loss: %.3f/%.3f'
            'sat2uav_EPE_loss: %.3f/%.3f/%.3f '
            'sat2uav_IOU_loss: %.3f/%.3f/%.3f '
            %
            (running_total_loss / (i + 1), sat2uav_loss.item(),
             running_EPE_loss / (i + 1), sat2uav_EPE_loss_1.item(), sat2uav_EPE_loss_2.item(),
             running_IOU_loss / (i + 1), sat2uav_IOU_loss_1.item(), sat2uav_IOU_loss_2.item()))

    running_total_loss /= len(train_loader)
    running_IOU_loss /= len(train_loader)
    running_EPE_loss /= len(train_loader)

    return running_total_loss, running_EPE_loss, running_IOU_loss


def validate_epoch(net,
                   val_loader,
                   device,
                   epoch,
                   test_writer,
                   save_path,
                   div_flow=1,
                   loss_grid_weights=None,
                   apply_mask=False,
                   sparse=False,
                   robust_L1_loss=False):
    """
    Validation epoch script
    Args:
        net: model architecture
        val_loader: dataloader
        device: `cpu` or `gpu`
        epoch: epoch number for plotting
        val_writer: for tensorboard
        div_flow: multiplicative factor to apply to the estimated flow
        save_path: path to folder to save the plots
        loss_grid_weights: weight coefficients for each level of the feature pyramid
        apply_mask: bool on whether or not to apply a mask for the loss
        robust_L1_loss: bool on the loss to use
        sparse: bool on sparsity of ground truth flow field
    Output:
        running_total_loss: total validation loss,
        EPE_0, EPE_1, EPE_2, EPE_3: EPEs corresponding to each level of the network (after upsampling
        the estimated flow to original resolution and scaling it properly to compare to ground truth).

        here output of the network at every level is flow interpolated but not scaled.
        we only use the ground truth flow as highest resolution and downsample it without scaling.

    """

    net.eval()
    if loss_grid_weights is None:
        loss_grid_weights = [0.32, 0.08, 0.02, 0.01, 0.005]
    running_total_loss = 0
    running_EPE_loss = 0
    running_IOU_loss = 0
    running_feature_loss = 0
    running_corr_loss = 0

    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=len(val_loader),ncols=500)
        sat2uav_step1_EPE_array = torch.zeros([len(loss_grid_weights), len(val_loader)], dtype=torch.float32,
                                              device=device)
        sat2uav_step2_EPE_array = torch.zeros([len(loss_grid_weights), len(val_loader)], dtype=torch.float32,
                                              device=device)
        for i, mini_batch in pbar:
            # sat2uav
            sat_image, uav_image, sat_image_256, uav_image_256 = pre_process_data(
                mini_batch['sat_image'],
                mini_batch['uav_image'],
                device=device)

            sat_label, uav_label, sat_label_256, uav_label_256 = pre_process_label(mini_batch['sat_label'],
                                                                                   mini_batch['uav_label'],
                                                                                   device=device)
            sat2uav_flow_gt_original = mini_batch['sat2uav_flow'].to(device)
            uav2sat_flow_gt_original = mini_batch['uav2sat_flow'].to(device)

            sat2uav_mask_gt = mini_batch['sat2uav_mask'].to(device)
            uav2sat_mask_gt = mini_batch['uav2sat_mask'].to(device)
            #
            # # uav2sat
            # sat_image, uav_image, sat_image_256, uav_image_256 = pre_process_data(mini_batch['uav_image'],
            #                                                                       mini_batch['sat_image'],
            #                                                                       device=device)
            # sat_label, uav_label, sat_label_256, uav_label_256 = pre_process_label(mini_batch['uav_label'],
            #                                                                        mini_batch['sat_label'],
            #                                                                        device=device)
            # sat2uav_flow_gt_original = mini_batch['uav2sat_flow'].to(device)
            # uav2sat_flow_gt_original = mini_batch['sat2uav_flow'].to(device)
            #
            # sat2uav_mask_gt = mini_batch['uav2sat_mask'].to(device)
            # uav2sat_mask_gt = mini_batch['sat2uav_mask'].to(device)

            sat2uav_step1_EPE_array, sat2uav_step2_EPE_array, sat2uav_loss, sat2uav_EPE_loss_1, sat2uav_EPE_loss_2, \
            sat2uav_IOU_loss_1, sat2uav_IOU_loss_2 = \
                train_source2target(net, save_path, epoch, i, apply_mask, div_flow,
                                    uav_image, sat_image, uav_image_256, sat_image_256,
                                    uav_label, sat_label, uav_label_256, sat_label_256,
                                    sat2uav_flow_gt_original, uav2sat_flow_gt_original,
                                    sat2uav_mask_gt, uav2sat_mask_gt,
                                    sat2uav_step1_EPE_array, sat2uav_step2_EPE_array, sparse=False, train=False,
                                    sat2uav=True,
                                    loss_grid_weights=loss_grid_weights, robust_L1_loss=robust_L1_loss)

            running_EPE_loss += sat2uav_EPE_loss_1.item() + sat2uav_EPE_loss_2.item()
            running_IOU_loss += sat2uav_IOU_loss_1.item() + sat2uav_IOU_loss_2.item()
            running_total_loss += sat2uav_loss.item()

            pbar.set_description(
                'sat2uav_total_loss: %.3f/%.3f'
                'sat2uav_EPE_loss: %.3f/%.3f/%.3f '
                'sat2uav_IOU_loss: %.3f/%.3f/%.3f '
                %
                (running_total_loss / (i + 1), sat2uav_loss.item(),
                 running_EPE_loss / (i + 1), sat2uav_EPE_loss_1.item(), sat2uav_EPE_loss_2.item(),
                 running_IOU_loss / (i + 1), sat2uav_IOU_loss_1.item(), sat2uav_IOU_loss_2.item()))

        running_total_loss /= len(val_loader)
        running_IOU_loss /= len(val_loader)
        running_EPE_loss /= len(val_loader)

        mean_sat2uav_step1_epe = torch.mean(sat2uav_step1_EPE_array, dim=1)
        mean_sat2uav_step2_epe = torch.mean(sat2uav_step2_EPE_array, dim=1)

        torch.cuda.empty_cache()
        return running_total_loss, running_EPE_loss, running_IOU_loss,\
               mean_sat2uav_step1_epe[0].item(), mean_sat2uav_step1_epe[1].item(), mean_sat2uav_step1_epe[2].item(), \
               mean_sat2uav_step1_epe[3].item(), \
               mean_sat2uav_step2_epe[0].item(), mean_sat2uav_step2_epe[1].item(), mean_sat2uav_step2_epe[2].item(), \
               mean_sat2uav_step2_epe[3].item()


def test_epoch(net,
               val_loader,
               device,
               epoch,
               save_path,
               div_flow=1,
               loss_grid_weights=None,
               apply_mask=False,
               sparse=False,
               robust_L1_loss=False):
    net.eval()
    if loss_grid_weights is None:
        loss_grid_weights = [0.32, 0.08, 0.02, 0.01, 0.005]
    running_total_loss = 0
    running_EPE_loss = 0
    running_IOU_loss = 0
    running_feature_loss = 0
    running_corr_loss = 0

    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=len(val_loader),ncols=500)
        sat2uav_step1_EPE_array = torch.zeros([len(loss_grid_weights), len(val_loader)], dtype=torch.float32,
                                              device=device)
        sat2uav_step2_EPE_array = torch.zeros([len(loss_grid_weights), len(val_loader)], dtype=torch.float32,
                                              device=device)
        for i, mini_batch in pbar:
            # sat2uav
            sat_image, uav_image, sat_image_256, uav_image_256 = pre_process_data(
                mini_batch['sat_image'],
                mini_batch['uav_image'],
                device=device)

            sat_label, uav_label, sat_label_256, uav_label_256 = pre_process_label(mini_batch['sat_label'],
                                                                                   mini_batch['uav_label'],
                                                                                   device=device)
            sat2uav_flow_gt_original = mini_batch['sat2uav_flow'].to(device)
            uav2sat_flow_gt_original = mini_batch['uav2sat_flow'].to(device)

            sat2uav_mask_gt = mini_batch['sat2uav_mask'].to(device)
            uav2sat_mask_gt = mini_batch['uav2sat_mask'].to(device)
            #
            # # uav2sat
            # sat_image, uav_image, sat_image_256, uav_image_256 = pre_process_data(mini_batch['uav_image'],
            #                                                                       mini_batch['sat_image'],
            #                                                                       device=device)
            # sat_label, uav_label, sat_label_256, uav_label_256 = pre_process_label(mini_batch['uav_label'],
            #                                                                        mini_batch['sat_label'],
            #                                                                        device=device)
            # sat2uav_flow_gt_original = mini_batch['uav2sat_flow'].to(device)
            # uav2sat_flow_gt_original = mini_batch['sat2uav_flow'].to(device)
            #
            # sat2uav_mask_gt = mini_batch['uav2sat_mask'].to(device)
            # uav2sat_mask_gt = mini_batch['sat2uav_mask'].to(device)

            sat2uav_step1_EPE_array, sat2uav_step2_EPE_array, sat2uav_loss, sat2uav_EPE_loss_1, sat2uav_EPE_loss_2, \
            sat2uav_IOU_loss_1, sat2uav_IOU_loss_2,sat2uav_feature_loss_1, sat2uav_feature_loss_2, sat2uav_corr_loss_1 , sat2uav_corr_loss_2 = \
                train_source2target(net, save_path, epoch, i, apply_mask, div_flow,
                                    uav_image, sat_image, uav_image_256, sat_image_256,
                                    uav_label, sat_label, uav_label_256, sat_label_256,
                                    sat2uav_flow_gt_original, uav2sat_flow_gt_original,
                                    sat2uav_mask_gt, uav2sat_mask_gt,
                                    sat2uav_step1_EPE_array, sat2uav_step2_EPE_array, sparse=False, train=False,
                                    sat2uav=True,
                                    loss_grid_weights=loss_grid_weights, robust_L1_loss=robust_L1_loss)

            running_EPE_loss += sat2uav_EPE_loss_1.item() + sat2uav_EPE_loss_2.item()
            running_IOU_loss += sat2uav_IOU_loss_1.item() + sat2uav_IOU_loss_2.item()
            running_feature_loss += sat2uav_feature_loss_1.item() + sat2uav_feature_loss_2.item()
            running_corr_loss += sat2uav_corr_loss_1.item() + sat2uav_corr_loss_2.item()
            running_total_loss += sat2uav_loss.item()

            pbar.set_description(
                'sat2uav_total_loss: %.3f/%.3f'
                'sat2uav_EPE_loss: %.3f/%.3f/%.3f '
                'sat2uav_IOU_loss: %.3f/%.3f/%.3f '
                'sat2uav_feature_loss: %.3f/%.3f/%.3f '
                'sat2uav_corr_loss: %.3f/%.3f/%.3f '
                %
                (running_total_loss / (i + 1), sat2uav_loss.item(),
                 running_EPE_loss / (i + 1), sat2uav_EPE_loss_1.item(), sat2uav_EPE_loss_2.item(),
                 running_IOU_loss / (i + 1), sat2uav_IOU_loss_1.item(), sat2uav_IOU_loss_2.item(),
                 running_feature_loss / (i + 1), sat2uav_feature_loss_1.item(), sat2uav_feature_loss_2.item(),
                 running_corr_loss / (i + 1), sat2uav_corr_loss_1.item(), sat2uav_corr_loss_2.item()))

        running_total_loss /= len(val_loader)
        running_IOU_loss /= len(val_loader)
        running_EPE_loss /= len(val_loader)
        running_feature_loss /= len(val_loader)
        running_corr_loss /= len(val_loader)

        mean_sat2uav_step1_epe = torch.mean(sat2uav_step1_EPE_array, dim=1)
        mean_sat2uav_step2_epe = torch.mean(sat2uav_step2_EPE_array, dim=1)
        # test_writer.add_scalar('test loss', running_total_loss, epoch)
        # test_writer.add_scalar('test epe loss', running_EPE_loss, epoch)
        # test_writer.add_scalar('test iou loss', running_IOU_loss, epoch)
        # test_writer.add_scalar('test feature loss', running_feature_loss, epoch)
        # test_writer.add_scalar('val_sat2uav_step1_epe', mean_sat2uav_step1_epe[0], epoch)
        # test_writer.add_scalar('val_sat2uav_step1_epe_H_8', mean_sat2uav_step1_epe[1].item(), epoch)
        # test_writer.add_scalar('val_sat2uav_step1_epe_32', mean_sat2uav_step1_epe[2], epoch)
        # test_writer.add_scalar('val_sat2uav_step1_epe_16', mean_sat2uav_step1_epe[3], epoch)
        # test_writer.add_scalar('val_uav2sat_step2_epe', mean_sat2uav_step2_epe[0], epoch)
        # test_writer.add_scalar('val_uav2sat_step2_epe_H_8', mean_sat2uav_step2_epe[1], epoch)
        # test_writer.add_scalar('val_uav2sat_step2_epe_32', mean_sat2uav_step2_epe[2], epoch)
        # test_writer.add_scalar('val_uav2sat_step2_epe_16', mean_sat2uav_step2_epe[3], epoch)
        torch.cuda.empty_cache()
        return running_total_loss, running_EPE_loss, running_IOU_loss, running_feature_loss,running_corr_loss,\
               mean_sat2uav_step1_epe[0].item(), mean_sat2uav_step1_epe[1].item(), mean_sat2uav_step1_epe[2].item(), \
               mean_sat2uav_step1_epe[3].item(), \
               mean_sat2uav_step2_epe[0].item(), mean_sat2uav_step2_epe[1].item(), mean_sat2uav_step2_epe[2].item(), \
               mean_sat2uav_step2_epe[3].item()

