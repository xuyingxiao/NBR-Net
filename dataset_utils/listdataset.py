import torch.utils.data as data
import os
import os.path
from imageio import imread
import numpy as np
from dataset_utils.util import load_flo


def get_gt_correspondence_mask(flows):
    masks = []
    for flow in flows:
        # convert flow to mapping
        h,w = flow.shape[:2]
        X, Y = np.meshgrid(np.linspace(0, w - 1, w),
                           np.linspace(0, h - 1, h))
        map_x = (flow[:,:,0]+X).astype(np.float32)
        map_y = (flow[:,:,1]+Y).astype(np.float32)

        mask_x = np.logical_and(map_x>0, map_x< w)
        mask_y = np.logical_and(map_y>0, map_y< h)
        mask = np.logical_and(mask_x, mask_y).astype(np.uint8)
        masks.append(mask)
    return masks


def default_loader(root, path_imgs, path_flows):
    imgs = [os.path.join(root,path) for path in path_imgs]
    flows = [os.path.join(root,path_flo) for path_flo in path_flows]
    return_imgs = []
    for img in imgs:
        shape = imread(img).shape
        #image
        if len(shape) == 3:
            return_imgs.append(imread(img).astype(np.uint8))
        #label
        else:
            label = imread(img)
            label[label<128] = 0
            label[label>=128] = 255
            label = np.expand_dims(label,axis=2)
            return_imgs.append(label.astype(np.uint8))

    return return_imgs, [load_flo(flow) for flow in flows]


class ListDataset(data.Dataset):
    def __init__(self, root, path_list, sat_image=None,uav_image=None,sat_label=None,uav_label=None,
                   sat2uav_flow=None,uav2sat_flow=None, loader=default_loader, mask=False, size=False):
        """

        :param root: directory containing the dataset images
        :param path_list: list containing the name of images and corresponding ground-truth flow files
        :param sat_image: transforms to apply to sat images
        :param uav_image: transforms to apply to uav images
        :param sat_label: transforms to apply to sat labels
        :param uav_label: transforms to apply to uav labels
        :param sat2uav_flow: transforms to apply to sat2uav flow field
        :param sat2uav_flow: transforms to apply to uav2sat flow field
        :param loader: loader function for the images and the flow
        :param mask: bool indicating is a mask of valid pixels needs to be loaded as well from root
        :param size: size of the original source image
        outputs:
            - sat_image
            - uav_image
            - sat2uav_flow
            - uav2sat_flow
            - sat2uav correspondence_mask
            - uav2sat correspondence_mask
            - source_image_size
        """

        self.root = root
        self.path_list = path_list
        self.sat_image = sat_image
        self.uav_image = uav_image
        self.sat_label = sat_label
        self.uav_label = uav_label
        self.sat2uav_flow = sat2uav_flow
        self.uav2sat_flow = uav2sat_flow
        self.loader = loader
        self.mask = mask
        self.size = size

    def __getitem__(self, index):
        # for all inputs[0] must be the sat and inputs[1] must be the uav
        inputs, gt_flows = self.path_list[index]

        if not self.mask:
            inputs, gt_flows = self.loader(self.root, inputs, gt_flows)
            source_size = inputs[0].shape
            sat2uav_mask, uav2sat_mask = get_gt_correspondence_mask(gt_flows)
        else:
            inputs, gt_flows = self.loader(self.root, inputs, gt_flows)
            source_size = inputs[0].shape

        # here gt_flow has shape HxWx2

        # after co transform that could be reshapping the target
        # transforms here will always contain conversion to tensor (then channel is before)
        if self.sat_image is not None:
            inputs[0] = self.sat_image(inputs[0])
        if self.uav_image is not None:
            inputs[1] = self.uav_image(inputs[1])
        if self.sat_label is not None:
            inputs[2] = self.sat_label(inputs[2])
        if self.uav_label is not None:
            inputs[3] = self.uav_label(inputs[3])
        if self.sat2uav_flow is not None:
            gt_flows[0] = self.sat2uav_flow(gt_flows[0])
        if self.uav2sat_flow is not None:
            gt_flows[1] = self.uav2sat_flow(gt_flows[1])

        return {'sat_image': inputs[0],
                'uav_image': inputs[1],
                'sat_label': inputs[2],
                'uav_label': inputs[3],
                'sat2uav_flow': gt_flows[0],
                'uav2sat_flow': gt_flows[1],
                'sat2uav_mask': sat2uav_mask.astype(np.uint8),
                'uav2sat_mask': uav2sat_mask.astype(np.uint8),
                'source_image_size': source_size,
                'index': index
                }

    def __len__(self):
        return len(self.path_list)
