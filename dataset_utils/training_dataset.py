"""
Extracted from DGC-Net https://github.com/AaltoVision/DGC-Net/blob/master/data/dataset.py and modified
"""
from os import path as osp
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.spatial.distance import squareform, pdist, cdist
import torch.nn.functional as F
from torch.utils.data import Dataset
from dataset_utils.util import center_crop
from torch.autograd import Variable


def homography_mat_from_4_pts(theta):

    b = theta.size(0)
    if not theta.size() == (b, 8):
        theta = theta.view(b, 8)
        theta = theta.contiguous()

    xp = theta[:, :4].unsqueeze(2);
    yp = theta[:, 4:].unsqueeze(2)

    x = Variable(torch.FloatTensor([-1, -1, 1, 1])).unsqueeze(1).unsqueeze(0).expand(b, 4, 1)
    y = Variable(torch.FloatTensor([-1, 1, -1, 1])).unsqueeze(1).unsqueeze(0).expand(b, 4, 1)
    z = Variable(torch.zeros(4)).unsqueeze(1).unsqueeze(0).expand(b, 4, 1)
    o = Variable(torch.ones(4)).unsqueeze(1).unsqueeze(0).expand(b, 4, 1)
    single_o = Variable(torch.ones(1)).unsqueeze(1).unsqueeze(0).expand(b, 1, 1)

    if theta.is_cuda:
        x = x.cuda()
        y = y.cuda()
        z = z.cuda()
        o = o.cuda()
        single_o = single_o.cuda()

    A = torch.cat([torch.cat([-x, -y, -o, z, z, z, x * xp, y * xp, xp], 2), torch.cat([z, z, z, -x, -y, -o, x * yp, y * yp, yp], 2)], 1)
    # find homography by assuming h33 = 1 and inverting the linear system
    h = torch.bmm(torch.inverse(A[:, :, :8]), -A[:, :, 8].unsqueeze(2))
    # add h33
    h = torch.cat([h, single_o], 1)

    H = h.squeeze(2)

    return H

def unormalise_and_convert_mapping_to_flow(map, output_channel_first=True):

    if not isinstance(map, np.ndarray):
        #torch tensor
        if len(map.shape) == 4:
            if map.shape[1] != 2:
                # size is BxHxWx2
                map = map.permute(0, 3, 1, 2)

            # channel first, here map is normalised to -1;1
            # we put it back to 0,W-1, then convert it to flow
            B, C, H, W = map.size()
            mapping = torch.zeros_like(map)
            # mesh grid
            mapping[:, 0, :, :] = (map[:, 0, :, :].float().clone() + 1) * (W - 1) / 2.0  # unormalise
            mapping[:, 1, :, :] = (map[:, 1, :, :].float().clone() + 1) * (H - 1) / 2.0  # unormalise

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).float()

            if mapping.is_cuda:
                grid = grid.cuda()
            flow = mapping - grid # here also channel first
            if not output_channel_first:
                flow = flow.permute(0,2,3,1)
        else:
            if map.shape[0] != 2:
                # size is HxWx2
                map = map.permute(2, 0, 1)

            # channel first, here map is normalised to -1;1
            # we put it back to 0,W-1, then convert it to flow
            C, H, W = map.size()
            mapping = torch.zeros_like(map)
            # mesh grid
            mapping[0, :, :] = (map[0, :, :].float().clone() + 1) * (W - 1) / 2.0  # unormalise
            mapping[1, :, :] = (map[1, :, :].float().clone() + 1) * (H - 1) / 2.0  # unormalise

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, H, W)
            yy = yy.view(1, H, W)
            grid = torch.cat((xx, yy), 0).float() # attention, concat axis=0 here

            if mapping.is_cuda:
                grid = grid.cuda()
            flow = mapping - grid # here also channel first
            if not output_channel_first:
                flow = flow.permute(1,2,0).float()
        return flow.float()
    else:
        # here numpy arrays
        flow = np.copy(map)
        if len(map.shape) == 4:
            if map.shape[1] == 2:
                # size is Bx2xHxWx
                map = map.permute(0, 2, 3, 1)

            #BxHxWx2
            b, h_scale, w_scale = map.shape[:3]
            mapping = np.zeros_like(map)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))
            mapping[:,:,:,0] = (map[:,:,:,0] + 1) * (w_scale - 1) / 2
            mapping[:,:,:,1] = (map[:,:,:,1] + 1) * (h_scale - 1) / 2
            for i in range(b):
                flow[i, :, :, 0] = mapping[i, :, :, 0] - X
                flow[i, :, :, 1] = mapping[i, :, :, 1] - Y
            if output_channel_first:
                flow = flow.transpose(0,3,1,2)
        else:
            if map.shape[0] == 2:
                # size is 2xHxW
                map = map.permute(1,2,0)

            # HxWx2
            h_scale, w_scale = map.shape[:2]
            mapping = np.zeros_like(map)
            mapping[:,:,0] = (map[:,:,0] + 1) * (w_scale - 1) / 2
            mapping[:,:,1] = (map[:,:,1] + 1) * (h_scale - 1) / 2
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))

            flow[:,:,0] = mapping[:,:,0]-X
            flow[:,:,1] = mapping[:,:,1]-Y
            if output_channel_first:
                flow = flow.transpose(2, 0, 1).float()
        return flow.astype(np.float32)

def expand_dim(tensor,dim,desired_dim_len):
    sz = list(tensor.size())
    sz[dim]=desired_dim_len
    return tensor.expand(tuple(sz))

class HomoAffTps_Dataset(Dataset):
    """
    Extracted from https://github.com/AaltoVision/DGC-Net/blob/master/data/dataset.py and modified
    Main dataset for generating the training/validation the proposed approach.
    It can handle affine, TPS, and Homography transformations.
    Args:
        image_path: filepath to the dataset
        csv_file: csv file with ground-truth transformation parameters and name of original images
        transforms: image transformations for the uav image (data preprocessing)
        transforms_target: image transformations for the sat image (data preprocessing), if different than that of
        the uav image
        get_flow: bool, whether to get flow or normalized mapping
        pyramid_param: spatial resolution of the feature maps at each level
            of the feature pyramid (list)
        output_size: size (tuple) of the output images
    Output:
        if get_flow:
            uav_image: uav image, shape 3xHxWx
            sat_image: sat image, shape 3xHxWx
            flow_map: corresponding ground-truth flow field, shape 2xHxW
            correspondence_mask: mask of valid flow, shape HxW
        else:
            uav_image: uav image, shape 3xHxWx
            sat_image: sat image, shape 3xHxWx
            correspondence_map: correspondence_map, normalized to [-1,1], shape HxWx2,
                                should correspond to correspondence_map_pyro[-1]
            correspondence_map_pyro: pixel correspondence map for each feature pyramid level
            mask_x: X component of the mask (valid/invalid correspondences)
            mask_y: Y component of the mask (valid/invalid correspondences)
            correspondence_mask: mask of valid flow, shape HxW, equal to mask_x and mask_y
    """

    def __init__(self,
                 image_path,
                 csv_file,
                 transforms,
                 transforms_target=None,
                 get_flow=False,
                 pyramid_param=[256],
                 output_size=(256,256)):
        super().__init__()
        # self.img_path = image_path
        # if not os.path.isdir(self.img_path):
        #     raise ValueError("The image path that you indicated does not exist!")

        self.transform_dict = {0: 'aff', 1: 'tps', 2: 'homo'}
        self.transforms_uav = transforms
        if transforms_target is None:
            self.transforms_target = transforms
        else:
            self.transforms_target = transforms_target
        self.pyramid_param = pyramid_param
        if os.path.exists(csv_file):
            self.df = pd.read_csv(csv_file)
            if len(self.df) == 0:
                raise ValueError("The csv file that you indicated is empty !")
        else:
            raise ValueError("The path to the csv file that you indicated does not exist !")
        self.get_flow = get_flow
        self.H_OUT, self.W_OUT = output_size

        # changed compared to version from DGC-Net
        self.ratio_cropping = 1
        # this is a scaling to apply to the homographies, usually applied to get 240x240 images
        self.ratio_TPS = 1
        self.ratio_homography = 1

        self.H_AFF_TPS, self.W_AFF_TPS = (256,256)
        self.H_HOMO, self.W_HOMO = (256,256)

        self.THETA_IDENTITY = \
            torch.Tensor(np.expand_dims(np.array([[1, 0, 0],
                                                  [0, 1, 0]]),
                                        0).astype(np.float32))
        self.gridGen = TpsGridGen(self.H_OUT, self.W_OUT)
        self.GenTPSGrid = TPS(self.H_OUT, self.W_OUT)
        self.image_path = "../dataset/whu/images/"


    def transform_image(self,
                        image,
                        out_h,
                        out_w,
                        padding_factor=1.0,
                        crop_factor=1.0,
                        theta=None):
        sampling_grid = self.generate_grid(out_h, out_w, theta)
        # rescale grid according to crop_factor and padding_factor
        sampling_grid.data = sampling_grid.data * padding_factor * crop_factor
        # sample transformed image
        warped_image_patch = F.grid_sample(image, sampling_grid)
        # cv2.imwrite("tmp/test/warp_uav_patch.jpg", F.grid_sample(warped_image_patch,
        #                                                          self.generate_grid(out_h, out_w, theta,
        #                                                                             reverse=True)).data.cpu().numpy()[
        #     0].transpose(1, 2, 0))
        return warped_image_patch

    def generate_grid(self, out_h, out_w, theta=None,reverse=False):
        out_size = torch.Size((1, 3, out_h, out_w))
        if theta is None:
            theta = self.THETA_IDENTITY
            theta = theta.expand(1, 2, 3).contiguous()
            return F.affine_grid(theta, out_size)
        elif (theta.shape[1] == 2):
            return F.affine_grid(theta, out_size)
        else:
            return self.GenTPSGrid(theta,reverse)
            # return self.gridGen(theta)

    def my_get_grid(self, theta,w,h):
        grid_X, grid_Y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
        grid_X = torch.FloatTensor(grid_X).unsqueeze(0).unsqueeze(3)
        grid_Y = torch.FloatTensor(grid_Y).unsqueeze(0).unsqueeze(3)
        grid_X = Variable(grid_X, requires_grad=False)
        grid_Y = Variable(grid_Y, requires_grad=False)


        theta = torch.Tensor(theta)#.unsqueeze(0)
        b = theta.size(0)
        if theta.size(1) == 9:
            H = theta
        h0 = H[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h1 = H[:, 1].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h2 = H[:, 2].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h3 = H[:, 3].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h4 = H[:, 4].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h5 = H[:, 5].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h6 = H[:, 6].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h7 = H[:, 7].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h8 = H[:, 8].unsqueeze(1).unsqueeze(2).unsqueeze(3)

        grid_X = expand_dim(grid_X, 0, b)
        grid_Y = expand_dim(grid_Y, 0, b)

        grid_Xp = grid_X * h0 + grid_Y * h1 + h2
        grid_Yp = grid_X * h3 + grid_Y * h4 + h5
        k = grid_X * h6 + grid_Y * h7 + h8

        grid_Xp /= k
        grid_Yp /= k

        return torch.cat((grid_Xp, grid_Yp), 3)

    def get_grid(self, H, ccrop):
        # top-left corner of the central crop
        X_CCROP, Y_CCROP = ccrop[0], ccrop[1]

        W_FULL, H_FULL = (self.W_HOMO, self.H_HOMO)
        W_SCALE, H_SCALE = (self.W_OUT, self.H_OUT)

        # inverse homography matrix
        Hinv = np.linalg.inv(H)
        Hscale = np.eye(3)
        Hscale[0,0] = Hscale[1,1] = self.ratio_homography
        Hinv = Hscale @ Hinv @ np.linalg.inv(Hscale)

        # estimate the grid for the whole image
        X, Y = np.meshgrid(np.linspace(0, W_FULL - 1, W_FULL),
                           np.linspace(0, H_FULL - 1, H_FULL))
        X_, Y_ = X, Y
        X, Y = X.flatten(), Y.flatten()

        # create matrix representation
        XYhom = np.stack([X, Y, np.ones_like(X)], axis=1).T

        # multiply Hinv to XYhom to find the warped grid
        XYwarpHom = np.dot(Hinv, XYhom)

        # vector representation
        XwarpHom = torch.from_numpy(XYwarpHom[0, :]).float()
        YwarpHom = torch.from_numpy(XYwarpHom[1, :]).float()
        ZwarpHom = torch.from_numpy(XYwarpHom[2, :]).float()

        X_grid_pivot = (XwarpHom / (ZwarpHom + 1e-8)).view(H_FULL, W_FULL)
        Y_grid_pivot = (YwarpHom / (ZwarpHom + 1e-8)).view(H_FULL, W_FULL)

        # normalize XwarpHom and YwarpHom and cast to [-1, 1] range
        Xwarp = (2 * X_grid_pivot / (W_FULL - 1) - 1)
        Ywarp = (2 * Y_grid_pivot / (H_FULL - 1) - 1)
        grid_full = torch.stack([Xwarp, Ywarp], dim=-1)

        # getting the central patch from the pivot
        Xwarp_crop = X_grid_pivot[Y_CCROP:Y_CCROP + H_SCALE,
                                  X_CCROP:X_CCROP + W_SCALE]
        Ywarp_crop = Y_grid_pivot[Y_CCROP:Y_CCROP + H_SCALE,
                                  X_CCROP:X_CCROP + W_SCALE]
        X_crop = X_[Y_CCROP:Y_CCROP + H_SCALE,
                    X_CCROP:X_CCROP + W_SCALE]
        Y_crop = Y_[Y_CCROP:Y_CCROP + H_SCALE,
                    X_CCROP:X_CCROP + W_SCALE]

        # crop grid
        Xwarp_crop_range = \
            2 * (Xwarp_crop - X_crop.min()) / (X_crop.max() - X_crop.min()) - 1
        Ywarp_crop_range = \
            2 * (Ywarp_crop - Y_crop.min()) / (Y_crop.max() - Y_crop.min()) - 1
        grid_crop = torch.stack([Xwarp_crop_range,
                                 Ywarp_crop_range], dim=-1)
        return grid_full.unsqueeze(0), grid_crop.unsqueeze(0)

    @staticmethod
    def symmetric_image_pad(image_batch, padding_factor):
        """
        Pad an input image mini-batch symmetrically
        Args:
            image_batch: an input image mini-batch to be pre-processed
            padding_factor: padding factor
        Output:
            image_batch: padded image mini-batch
        """
        b, c, h, w = image_batch.size()
        pad_h, pad_w = int(h * padding_factor), int(w * padding_factor)
        idx_pad_left = torch.LongTensor(range(pad_w - 1, -1, -1))
        idx_pad_right = torch.LongTensor(range(w - 1, w - pad_w - 1, -1))
        idx_pad_top = torch.LongTensor(range(pad_h - 1, -1, -1))
        idx_pad_bottom = torch.LongTensor(range(h - 1, h - pad_h - 1, -1))

        image_batch = torch.cat((image_batch.index_select(3, idx_pad_left),
                                 image_batch,
                                 image_batch.index_select(3, idx_pad_right)),
                                3)
        image_batch = torch.cat((image_batch.index_select(2, idx_pad_top),
                                 image_batch,
                                 image_batch.index_select(2, idx_pad_bottom)),
                                2)
        return image_batch

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        # get the transformation type flag
        transform_type = data['aff/tps/homo'].astype('uint8')
        count =data['count']
        # sat_img_name = data['sat_image_name']
        # uav_img_name = data['uav_image_name']
        # print(sat_img_name,uav_img_name)
        # sat_lbl_name = sat_img_name.replace("image", "label")
        # uav_lbl_name = uav_img_name.replace("image", "label")
        # if not os.path.exists(uav_img_name):
        #     raise ValueError("The path to one of the original image {} does not exist, check your image path "
        #                      "and your csv file !".format(uav_img_name))
        # read image
        sat_img = cv2.cvtColor(cv2.imread(os.path.join(self.image_path,"image_" + str(count) + "_img_sat.jpg")),cv2.COLOR_BGR2RGB)
        uav_img = cv2.cvtColor(cv2.imread(os.path.join(self.image_path,"image_" + str(count) + "_img_uav.jpg")),cv2.COLOR_BGR2RGB)
        sat_lbl = cv2.imread(os.path.join(self.image_path,"image_" + str(count) + "_lbl_sat.jpg"))
        uav_lbl = cv2.imread(os.path.join(self.image_path,"image_" + str(count) + "_lbl_uav.jpg"))
        change_lbl = cv2.imread(os.path.join(self.image_path,"image_" + str(count) + "_change_label.jpg"))
        # aff/tps transformations
        if transform_type == 0 or transform_type == 1:
            # cropping dimention of the image first if it is too big, would occur to big resizing after
            if uav_img.shape[0] > self.H_AFF_TPS*self.ratio_cropping or \
               uav_img.shape[1] > self.W_AFF_TPS*self.ratio_cropping:
                uav_img, x, y = center_crop(uav_img, (int(self.W_AFF_TPS*self.ratio_cropping),
                                                            int(self.H_AFF_TPS*self.ratio_cropping)))

            if transform_type == 0:
                theta = data.iloc[3:9].values.astype('float').reshape(2, 3)
                theta = torch.Tensor(theta.astype(np.float32)).expand(1, 2, 3)
            else:
                theta = data.iloc[3:].values.astype('float')
                theta = np.expand_dims(np.expand_dims(theta, 1), 2)
                theta = torch.Tensor(theta.astype(np.float32))
                theta = theta.expand(1, 18, 1, 1)

            # make arrays float tensor for subsequent processing
            uav_image = torch.Tensor(uav_img.astype(np.float32))
            sat_image = torch.Tensor(sat_img.astype(np.float32))
            uav_label = torch.Tensor(uav_lbl.astype(np.float32))
            sat_label = torch.Tensor(sat_lbl.astype(np.float32))
            change_label = torch.Tensor(change_lbl.astype(np.float32))

            uav_image = uav_image.transpose(1, 2).transpose(0, 1)
            sat_image = sat_image.transpose(1, 2).transpose(0, 1)
            uav_label = uav_label.transpose(1, 2).transpose(0, 1)
            sat_label = sat_label.transpose(1, 2).transpose(0, 1)
            change_label = change_label.transpose(1, 2).transpose(0, 1)

            # Resize image using bilinear sampling with identity affine
            sat_image = self.transform_image(sat_image.unsqueeze(0), self.H_AFF_TPS, self.W_AFF_TPS).squeeze().numpy()
            uav_image = self.transform_image(uav_image.unsqueeze(0), self.H_AFF_TPS, self.W_AFF_TPS,theta=theta).squeeze().numpy()
            sat_label_image = self.transform_image(sat_label.unsqueeze(0), self.H_AFF_TPS, self.W_AFF_TPS).squeeze(0).numpy()
            uav_label_image = self.transform_image(uav_label.unsqueeze(0), self.H_AFF_TPS, self.W_AFF_TPS,theta=theta).squeeze(0).numpy()
            sat_change_label_image = self.transform_image(change_label.unsqueeze(0), self.H_AFF_TPS, self.W_AFF_TPS).squeeze(0).numpy()
            uav_change_label_image = self.transform_image(change_label.unsqueeze(0), self.H_AFF_TPS, self.W_AFF_TPS,theta=theta).squeeze(0).numpy()

            # # convert to [H, W, C] convention (for np arrays)
            # uav_image = uav_image.transpose((1, 2, 0))
            # sat_image = sat_image.transpose((1, 2, 0))
            # sat_label = sat_label_image.transpose((1, 2, 0))
            # sat2uav_label = uav_label_image.transpose((1, 2, 0))

        # Homography transformation
        elif transform_type == 2:
            theta = data.iloc[3:12].values.astype('double').reshape(3,3)
            # cropping dimention of the image first if it is too big, would occur to big resizing after
            if uav_img.shape[0] > self.H_HOMO * self.ratio_cropping \
                    or uav_img.shape[1] > self.W_HOMO*self.ratio_cropping:
                uav_img, x, y = center_crop(uav_img, (int(self.W_HOMO*self.ratio_cropping),
                                                            int(self.H_HOMO*self.ratio_cropping)))

            # resize to value stated at the beginning
            uav_image = cv2.resize(uav_img, dsize=(self.W_HOMO, self.H_HOMO),
                                      interpolation=cv2.INTER_LINEAR) # cv2.resize, W is giving first
            sat_image = cv2.resize(sat_img, dsize=(self.W_HOMO, self.H_HOMO),
                                      interpolation=cv2.INTER_LINEAR)
            sat_label_image = cv2.resize(sat_lbl, dsize=(self.W_HOMO, self.H_HOMO),
                                      interpolation=cv2.INTER_LINEAR)
            uav_label_image = cv2.resize(uav_lbl, dsize=(self.W_HOMO, self.H_HOMO),
                                      interpolation=cv2.INTER_LINEAR)
            sat_change_label_image = cv2.resize(change_lbl, dsize=(self.W_HOMO, self.H_HOMO),
                                      interpolation=cv2.INTER_LINEAR)

            sat2uav_grid = self.my_get_grid(theta.reshape(1,9),self.W_OUT,self.H_OUT)
            inv_theta = np.linalg.inv(theta)
            uav2sat_grid = self.my_get_grid(inv_theta.reshape(1,9),self.W_OUT,self.H_OUT)

            # warp the fullsize original uav image
            uav_image = torch.Tensor(uav_image.astype(np.float32))
            uav_image = uav_image.permute(2, 0, 1)
            sat_image = torch.Tensor(sat_image.astype(np.float32))
            sat_image = sat_image.permute(2, 0, 1)

            uav_image = F.grid_sample(uav_image.unsqueeze(0),sat2uav_grid).squeeze(0)
            
            # sat_label_image = np.expand_dims(sat_label_image, axis=2)
            sat_label_image = torch.Tensor(sat_label_image.astype(np.float32))
            sat_label_image = sat_label_image.permute(2, 0, 1)
            uav_label_image = torch.Tensor(uav_label_image.astype(np.float32))
            uav_label_image = uav_label_image.permute(2, 0, 1)
            sat_change_label_image = torch.Tensor(sat_change_label_image.astype(np.float32))
            sat_change_label_image = sat_change_label_image.permute(2, 0, 1)
            uav_label_image = F.grid_sample(uav_label_image.unsqueeze(0),sat2uav_grid)
            uav_change_label_image = F.grid_sample(sat_change_label_image.unsqueeze(0),sat2uav_grid)

            uav2sat_label_image = F.grid_sample(uav_label_image, uav2sat_grid).squeeze(0)
            uav_label_image = uav_label_image.squeeze(0)
            uav_change_label_image = uav_change_label_image.squeeze(0)

        else:
            print('Error: transformation type')

        # construct a pyramid with a corresponding grid on each layer
        sat2uav_grid_pyramid = []
        sat2uav_mask_x = []
        sat2uav_mask_y = []
        
        uav2sat_grid_pyramid = []
        uav2sat_mask_x = []
        uav2sat_mask_y = []
        
        if transform_type == 0:
            for layer_size in self.pyramid_param:
                # get layer size or change it so that it corresponds to PWCNet
                sat2uav_grid = self.generate_grid(layer_size,
                                          layer_size,
                                          theta).squeeze(0)
                sat2uav_mask = sat2uav_grid.ge(-1) & sat2uav_grid.le(1)
                sat2uav_grid_pyramid.append(sat2uav_grid)
                sat2uav_mask_x.append(sat2uav_mask[:, :, 0])
                sat2uav_mask_y.append(sat2uav_mask[:, :, 1])

                inv_theta = np.linalg.inv(np.vstack((theta.data.cpu().numpy()[0], np.array([0,0,1]))))
                uav2sat_grid = self.generate_grid(layer_size,
                                          layer_size,
                                          torch.Tensor(inv_theta[:2,:]).unsqueeze(0)).squeeze(0)
                uav2sat_mask = uav2sat_grid.ge(-1) & uav2sat_grid.le(1)
                uav2sat_grid_pyramid.append(uav2sat_grid)
                uav2sat_mask_x.append(uav2sat_mask[:, :, 0])
                uav2sat_mask_y.append(uav2sat_mask[:, :, 1])
                
        elif transform_type == 1:
            sat2uav_grid = self.generate_grid(self.H_OUT,
                                      self.W_OUT,
                                      theta).squeeze(0)
            uav2sat_grid = self.generate_grid(self.H_OUT,
                                      self.W_OUT,
                                      theta,reverse=True).squeeze(0)
            for layer_size in self.pyramid_param:
                sat2uav_grid_m = torch.from_numpy(cv2.resize(sat2uav_grid.numpy(),
                                                     (layer_size, layer_size)))
                sat2uav_mask = sat2uav_grid_m.ge(-1) & sat2uav_grid_m.le(1)
                sat2uav_grid_pyramid.append(sat2uav_grid_m)
                sat2uav_mask_x.append(sat2uav_mask[:, :, 0])
                sat2uav_mask_y.append(sat2uav_mask[:, :, 1])
                
                uav2sat_grid_m = torch.from_numpy(cv2.resize(uav2sat_grid.numpy(),
                                                     (layer_size, layer_size)))
                uav2sat_mask = uav2sat_grid_m.ge(-1) & uav2sat_grid_m.le(1)
                uav2sat_grid_pyramid.append(uav2sat_grid_m)
                uav2sat_mask_x.append(uav2sat_mask[:, :, 0])
                uav2sat_mask_y.append(uav2sat_mask[:, :, 1])
                
        elif transform_type == 2:
            sat2uav_grid = sat2uav_grid.squeeze(0)
            uav2sat_grid = uav2sat_grid.squeeze(0)
            for layer_size in self.pyramid_param:
                sat2uav_grid_m = torch.from_numpy(cv2.resize(sat2uav_grid.numpy(),
                                                    (layer_size, layer_size)))
                mask = sat2uav_grid_m.ge(-1) & sat2uav_grid_m.le(1)

                sat2uav_grid_pyramid.append(sat2uav_grid_m)
                sat2uav_mask_x.append(mask[:, :, 0])
                sat2uav_mask_y.append(mask[:, :, 1])
                
                uav2sat_grid_m = torch.from_numpy(cv2.resize(uav2sat_grid.numpy(),
                                                    (layer_size, layer_size)))
                mask = uav2sat_grid_m.ge(-1) & uav2sat_grid_m.le(1)

                uav2sat_grid_pyramid.append(uav2sat_grid_m)
                uav2sat_mask_x.append(mask[:, :, 0])
                uav2sat_mask_y.append(mask[:, :, 1])

        if self.get_flow:
            # ATTENTION, here we just get the flow of the highest resolution asked, not the pyramid of flows !
            sat2uav_flow = unormalise_and_convert_mapping_to_flow(sat2uav_grid_pyramid[-1], output_channel_first=True)
            uav2sat_flow = unormalise_and_convert_mapping_to_flow(uav2sat_grid_pyramid[-1], output_channel_first=True)
            return {'uav_image': uav_image, # shape is 1x3xHxW
                    'sat_image': sat_image, # shape is 1x3xHxW
                    'uav_label': uav_label_image,# shape is 1x1xHxW
                    'sat_label': sat_label_image,# shape is 1x1xHxW
                    'uav_change_label': uav_change_label_image,  # shape is 1x1xHxW
                    'sat_change_label': sat_change_label_image,  # shape is 1x1xHxW
                    'sat2uav_flow_map': sat2uav_flow, # here flow map is 2 x h x w
                    'uav2sat_flow_map': uav2sat_flow, # here flow map is 2 x h x w
                    'sat2uav_correspondence_mask': np.logical_and(sat2uav_mask_x[-1].detach().numpy(),
                                                                  sat2uav_mask_y[-1].detach().numpy()).astype(np.uint8),
                    'uav2sat_correspondence_mask': np.logical_and(uav2sat_mask_x[-1].detach().numpy(),
                                                                  uav2sat_mask_y[-1].detach().numpy()).astype(np.uint8)}
        else:
            # here we get both the pyramid of mappings and the last mapping (at the highest resolution)
            return {'uav_image': uav_image,
                    'sat_image': sat_image,
                    'correspondence_map': sat2uav_grid_pyramid[-1], #torch tensor,  h x w x 2
                    'correspondence_map_pyro': sat2uav_grid_pyramid,
                    'mask_x': sat2uav_mask_x,
                    'mask_y': sat2uav_mask_y}


class TpsGridGen(nn.Module):
    """
    Adopted version of synthetically transformed pairs dataset by I.Rocco
    https://github.com/ignacio-rocco/cnngeometric_pytorch
    """

    def __init__(self,
                 out_h=240,
                 out_w=240,
                 use_regular_grid=True,
                 grid_size=3,
                 reg_factor=0,
                 use_cuda=False):
        super(TpsGridGen, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.reg_factor = reg_factor
        self.use_cuda = use_cuda

        # create grid in numpy
        self.grid = np.zeros([self.out_h, self.out_w, 3], dtype=np.float32)
        # sampling grid with dim-0 coords (Y)
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, out_w),
                                               np.linspace(-1, 1, out_h))
        # grid_X,grid_Y: size [1,H,W,1,1]
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)
        if use_cuda:
            self.grid_X = self.grid_X.cuda()
            self.grid_Y = self.grid_Y.cuda()

        # initialize regular grid for control points P_i
        if use_regular_grid:
            axis_coords = np.linspace(-1, 1, grid_size)
            self.N = grid_size * grid_size
            P_Y, P_X = np.meshgrid(axis_coords, axis_coords)
            P_X = np.reshape(P_X, (-1, 1))  # size (N,1)
            P_Y = np.reshape(P_Y, (-1, 1))  # size (N,1)
            P_X = torch.FloatTensor(P_X)
            P_Y = torch.FloatTensor(P_Y)
            self.Li = self.compute_L_inverse(P_X, P_Y).unsqueeze(0)
            self.P_X = \
                P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
            self.P_Y = \
                P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
            if use_cuda:
                self.P_X = self.P_X.cuda()
                self.P_Y = self.P_Y.cuda()

    def forward(self, theta):
        warped_grid = self.apply_transformation(theta,
                                                torch.cat((self.grid_X,
                                                           self.grid_Y), 3))
        return warped_grid

    def compute_L_inverse(self, X, Y):
        # num of points (along dim 0)
        N = X.size()[0]

        # construct matrix K
        Xmat = X.expand(N, N)
        Ymat = Y.expand(N, N)
        P_dist_squared = \
            torch.pow(Xmat - Xmat.transpose(0, 1), 2) + \
            torch.pow(Ymat - Ymat.transpose(0, 1), 2)

        # make diagonal 1 to avoid NaN in log computation
        P_dist_squared[P_dist_squared == 0] = 1
        K = torch.mul(P_dist_squared, torch.log(P_dist_squared))

        # construct matrix L
        OO = torch.FloatTensor(N, 1).fill_(1)
        Z = torch.FloatTensor(3, 3).fill_(0)
        P = torch.cat((OO, X, Y), 1)
        L = torch.cat((torch.cat((K, P), 1),
                       torch.cat((P.transpose(0, 1), Z), 1)), 0)
        Li = torch.inverse(L)
        if self.use_cuda:
            Li = Li.cuda()
        return Li

    def apply_transformation(self, theta, points):
        if theta.dim() == 2:
            theta = theta.unsqueeze(2).unsqueeze(3)
        '''
        points should be in the [B,H,W,2] format,
        where points[:,:,:,0] are the X coords
        and points[:,:,:,1] are the Y coords
        '''

        # input are the corresponding control points P_i
        batch_size = theta.size()[0]
        # split theta into point coordinates
        Q_X = theta[:, :self.N, :, :].squeeze(3)
        Q_Y = theta[:, self.N:, :, :].squeeze(3)

        # get spatial dimensions of points
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]

        '''
        repeat pre-defined control points along
        spatial dimensions of points to be transformed
        '''
        P_X = self.P_X.expand((1, points_h, points_w, 1, self.N))
        P_Y = self.P_Y.expand((1, points_h, points_w, 1, self.N))

        # compute weigths for non-linear part
        W_X = \
            torch.bmm(self.Li[:, :self.N, :self.N].expand((batch_size,
                                                           self.N,
                                                           self.N)), Q_X)
        W_Y = \
            torch.bmm(self.Li[:, :self.N, :self.N].expand((batch_size,
                                                           self.N,
                                                           self.N)), Q_Y)
        '''
        reshape
        W_X,W,Y: size [B,H,W,1,N]
        '''
        W_X = \
            W_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1,
                                                                 points_h,
                                                                 points_w,
                                                                 1,
                                                                 1)
        W_Y = \
            W_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1,
                                                                 points_h,
                                                                 points_w,
                                                                 1,
                                                                 1)
        # compute weights for affine part
        A_X = \
            torch.bmm(self.Li[:, self.N:, :self.N].expand((batch_size,
                                                           3,
                                                           self.N)), Q_X)
        A_Y = \
            torch.bmm(self.Li[:, self.N:, :self.N].expand((batch_size,
                                                           3,
                                                           self.N)), Q_Y)
        '''
        reshape
        A_X,A,Y: size [B,H,W,1,3]
        '''
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1,
                                                                   points_h,
                                                                   points_w,
                                                                   1,
                                                                   1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1,
                                                                   points_h,
                                                                   points_w,
                                                                   1,
                                                                   1)
        '''
        compute distance P_i - (grid_X,grid_Y)
        grid is expanded in point dim 4, but not in batch dim 0,
        as points P_X,P_Y are fixed for all batch
        '''
        sz_x = points[:, :, :, 0].size()
        sz_y = points[:, :, :, 1].size()
        p_X_for_summation = points[:, :, :, 0].unsqueeze(3).unsqueeze(4)
        p_X_for_summation = p_X_for_summation.expand(sz_x + (1, self.N))
        p_Y_for_summation = points[:, :, :, 1].unsqueeze(3).unsqueeze(4)
        p_Y_for_summation = p_Y_for_summation.expand(sz_y + (1, self.N))

        if points_b == 1:
            delta_X = p_X_for_summation - P_X
            delta_Y = p_Y_for_summation - P_Y
        else:
            # use expanded P_X,P_Y in batch dimension
            delta_X = p_X_for_summation - P_X.expand_as(p_X_for_summation)
            delta_Y = p_Y_for_summation - P_Y.expand_as(p_Y_for_summation)

        dist_squared = torch.pow(delta_X, 2) + torch.pow(delta_Y, 2)
        '''
        U: size [1,H,W,1,N]
        avoid NaN in log computation
        '''
        dist_squared[dist_squared == 0] = 1
        U = torch.mul(dist_squared, torch.log(dist_squared))

        # expand grid in batch dimension if necessary
        points_X_batch = points[:, :, :, 0].unsqueeze(3)
        points_Y_batch = points[:, :, :, 1].unsqueeze(3)
        if points_b == 1:
            points_X_batch = points_X_batch.expand((batch_size,) +
                                                   points_X_batch.size()[1:])
            points_Y_batch = points_Y_batch.expand((batch_size,) +
                                                   points_Y_batch.size()[1:])

        points_X_prime = \
            A_X[:, :, :, :, 0] + \
            torch.mul(A_X[:, :, :, :, 1], points_X_batch) + \
            torch.mul(A_X[:, :, :, :, 2], points_Y_batch) + \
            torch.sum(torch.mul(W_X, U.expand_as(W_X)), 4)

        points_Y_prime = \
            A_Y[:, :, :, :, 0] + \
            torch.mul(A_Y[:, :, :, :, 1], points_X_batch) + \
            torch.mul(A_Y[:, :, :, :, 2], points_Y_batch) + \
            torch.sum(torch.mul(W_Y, U.expand_as(W_Y)), 4)
        return torch.cat((points_X_prime, points_Y_prime), 3)


class TPS(nn.Module):
    def __init__(self,
                 out_h=256,
                 out_w=256,
                 use_regular_grid=True,
                 grid_size=3,
                 reg_factor=0,
                 use_cuda=False):
        super(TPS, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.reg_factor = reg_factor
        self.use_cuda = use_cuda

        # create grid in numpy
        self.grid = np.zeros([self.out_h, self.out_w, 3], dtype=np.float32)
        # sampling grid with dim-0 coords (Y)
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, out_w),
                                               np.linspace(-1, 1, out_h))
        # grid_X,grid_Y: size [1,H,W,1,1]
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)
        if use_cuda:
            self.grid_X = self.grid_X.cuda()
            self.grid_Y = self.grid_Y.cuda()

    def make_T(self,X):
        # X: [K x 2] input control points
        # T: [(K+3) x (K+3)] magic matrix
        K = X.shape[0]
        T = np.zeros((K + 3, K + 3))
        T[:K, 0] = 1
        T[:K, 1:3] = X
        T[K, 3:] = 1
        T[K + 1:, 3:] = X.T
        # compute every point pair of points
        R = pdist(X, metric='sqeuclidean')
        R = squareform(R)
        R[R == 0] = 1  # a trick to make R ln(R) 0
        R = R * np.log(R)
        T[:K, 3:] = R
        return T

    def compute_params(self,X, Y):
        # X and Y: [K x 2] control points
        # params: [(K+3) x 2]
        T = self.make_T(X)
        Y = np.vstack([Y, np.zeros((3, 2))])
        params = np.linalg.solve(T, Y)
        return params

    def make_input_matrix(self,X, p):
        # X: [K x 2] input control points
        # p: [N x 2] input points
        # B: [N x (K+3)]
        K = X.shape[0]
        N = p.shape[0]
        B = np.zeros((N, K + 3))
        B[:, 0] = 1
        B[:, 1:3] = p
        R = cdist(p, X, 'sqeuclidean')
        R[R == 0] = 1
        R = R * np.log(R)
        B[:, 3:] = R
        return B

    def tps_map(self,X, Y, p):
        # X and Y: [K x 2] control points
        # p: [N x 2] input points
        # op: [N x 2] output points
        W = self.compute_params(X, Y)  # [(K+3) x 2]
        B = self.make_input_matrix(X, p)  # [N x (k+3)]
        op = B.dot(W)
        return op

    def apply_transformation(self, theta,grid_size=3,reverse=False):
        theta_x = theta[:, :grid_size*grid_size, :, :].squeeze().data.numpy()
        theta_y = theta[:, grid_size*grid_size:, :, :].squeeze().data.numpy()
        ir_cp3 = np.vstack((theta_x, theta_y)).T
        a, b = np.meshgrid(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3))
        re_cp3 = np.stack((b, a)).reshape(2, 9).T

        g1, g2 = np.meshgrid(np.linspace(-1, 1, self.out_w), np.linspace(-1, 1, self.out_h))
        re_cp_grid = np.stack((g1, g2)).reshape(2, -1).T
        if not reverse:
            ir_cp_grid = self.tps_map(re_cp3, ir_cp3, re_cp_grid)
            ir_cp_grid = ir_cp_grid.reshape(self.out_h, self.out_w, 2)
            ir_cp_grid = torch.as_tensor(ir_cp_grid).float().unsqueeze(0)

            return ir_cp_grid
        else:
            # ir_cp_grid = self.tps_map(re_cp3, ir_cp3, re_cp_grid)
            # ir_cp_grid = ir_cp_grid.reshape(self.out_h, self.out_w, 2)
            # ir_cp_grid = torch.as_tensor(ir_cp_grid).float().unsqueeze(0)
            #
            # g1, g2 = np.meshgrid(np.linspace(-1, 1, self.out_w), np.linspace(-1, 1, self.out_h))
            # grid = np.stack((g1, g2)).reshape(2, -1).T
            re_cp_grid = self.tps_map(ir_cp3,re_cp3, re_cp_grid)
            re_cp_grid = re_cp_grid.reshape(self.out_h, self.out_w, 2)
            re_cp_grid = torch.as_tensor(re_cp_grid).float().unsqueeze(0)

            # re_cp_grid = self.tps_map(ir_cp3,re_cp3,ir_cp_grid)
            # re_cp_grid = re_cp_grid.reshape(self.out_h, self.out_w, 2)
            # re_cp_grid = torch.as_tensor(re_cp_grid).float().unsqueeze(0)

            return re_cp_grid


    def forward(self, theta, reverse=False):
        ir_cp_grid = self.apply_transformation(theta,grid_size=3,reverse=reverse)
        return ir_cp_grid


