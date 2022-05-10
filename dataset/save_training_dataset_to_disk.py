import os
import numpy as np
import argparse
import random
from matplotlib import pyplot as plt
from tqdm import tqdm
import imageio
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset_utils.training_dataset import HomoAffTps_Dataset
from dataset_utils.dataset_io import boolean_string,writeFlow
from dataset_utils.pixel_wise_mapping import remap_using_flow_fields
from dataset_utils.image_transforms import ArrayToTensor

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='remote sensing image train script')
    parser.add_argument('--image_data_path', type=str,default="../../GLUnet/sat_dataset/sat_data/training-virigina/",
                        help='path to directory containing the original images and label.')
    # parser.add_argument('--csv_path', type=str, default="csv_files/tps_train.csv",
    # parser.add_argument('--csv_path', type=str, default="csv_files/aff_train.csv",
    parser.add_argument('--csv_path', type=str, default="csv_files/aff_homo_test.csv",
                        help='path to the CSV files')
    parser.add_argument('--save_dir', type=str,default="../dataset/aff_homo/test/",
                        help='path directory to save the image pairs and corresponding ground-truth flows')
    parser.add_argument('--plot', default=False, type=boolean_string,
                        help='plot as examples the first 4 pairs ? default is False')
    parser.add_argument('--seed', type=int, default=1981,
                        help='Pseudo-RNG seed')

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    plot = args.plot
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    image_dir=os.path.join(save_dir, 'images')
    flow_dir = os.path.join(save_dir, 'flow')

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(flow_dir):
        os.makedirs(flow_dir)

    # datasets
    source_img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])
    target_img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])
    pyramid_param = [256]

    # training dataset
    train_dataset = HomoAffTps_Dataset(image_path=args.image_data_path,
                                       csv_file=args.csv_path,
                                       transforms=source_img_transforms,
                                       transforms_target=target_img_transforms,
                                       pyramid_param=pyramid_param,
                                       get_flow=True,
                                       output_size=(256, 256))

    test_dataloader = DataLoader(train_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1)

    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    for i, minibatch in pbar:
        uav_image = minibatch['uav_image'] # shape is 1x3xHxW
        sat_image = minibatch['sat_image'] # shape is 1x3xHxW
        uav_label = minibatch['uav_label'] # shape is 1x1xHxW
        sat_label = minibatch['sat_label'] # shape is 1x1xHxW
        print("sat_image shape:",sat_image.shape)
        print("uav_image shape:",uav_image.shape)
        print("sat_label shape:",sat_label.shape)
        print("uav_label shape:",uav_label.shape)
        if uav_image.shape[1] == 3:
            uav_image = uav_image.permute(0, 2, 3, 1)[0].numpy().astype(np.uint8)
        else:
            uav_image = uav_image[0].numpy().astype(np.uint8)

        if sat_image.shape[1] == 3:
            sat_image = sat_image.permute(0, 2, 3, 1)[0].numpy().astype(np.uint8)
        else:
            sat_image = sat_image[0].numpy().astype(np.uint8)

        if sat_label.shape[1] == 1:
            sat_label = sat_label.permute(0, 2, 3, 1)[0].numpy().astype(np.uint8)
        else:
            sat_label = sat_label[0].numpy().astype(np.uint8)

        if uav_label.shape[1] == 1:
            uav_label = uav_label.permute(0, 2, 3, 1)[0].numpy().astype(np.uint8)
        else:
            uav_label = uav_label[0].numpy().astype(np.uint8)

        sat2uav_flow_gt = minibatch['sat2uav_flow_map'][0].permute(1,2,0).numpy() # now shape is HxWx2
        uav2sat_flow_gt = minibatch['uav2sat_flow_map'][0].permute(1,2,0).numpy() # now shape is HxWx2

        # save the flow file and the images files
        base_name = 'image_{}'.format(i)
        writeFlow(sat2uav_flow_gt, base_name + '_sat2uav_flow.flo', flow_dir)
        writeFlow(uav2sat_flow_gt, base_name + '_uav2sat_flow.flo', flow_dir)
        imageio.imwrite(os.path.join(save_dir, 'images/', base_name + '_img_uav.jpg'), uav_image)
        imageio.imwrite(os.path.join(save_dir, 'images/', base_name + '_img_sat.jpg'), sat_image)
        imageio.imwrite(os.path.join(save_dir, 'images/', base_name + '_lbl_uav.jpg'), uav_label)
        imageio.imwrite(os.path.join(save_dir, 'images/', base_name + '_lbl_sat.jpg'), sat_label)

        # # plotting to make sure that eevrything is working
        # if plot and i < 4:
        #     # just for now
        #     fig, axis = plt.subplots(1, 3, figsize=(20, 20))
        #     axis[0].imshow(uav_image)
        #     axis[0].set_title("Image source")
        #     axis[1].imshow(sat_image)
        #     axis[1].set_title("Image target")
        #     remapped_gt = remap_using_flow_fields(uav_image, sat2uav_flow_gt[:,:,0], sat2uav_flow_gt[:,:,1])
        #
        #     axis[2].imshow(remapped_gt)
        #     axis[2].set_title("Warped source image according to ground truth flow")
        #     fig.savefig(os.path.join(save_dir, 'synthetic_pair_{}'.format(i)), bbox_inches='tight')
        #     plt.close(fig)