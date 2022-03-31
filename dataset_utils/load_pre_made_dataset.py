import os.path
import glob
from dataset_utils.listdataset import ListDataset
from dataset_utils.util import split2list


def make_dataset(dir, split=None, dataset_name=None):
    '''Will search for triplets that go by the pattern '[name]_img1.ppm  [name]_img2.ppm  in folder images and
      [name]_flow.flo' in folder flow '''
    images = []
    '''
    if get_mapping:
        flow_dir = 'mapping'
        # flow_dir is actually mapping dir in that case, it is always normalised to [-1,1]
    '''
    flow_dir = 'flow'
    image_dir = 'images'

    # Make sure that the folders exist
    if not os.path.isdir(dir):
        raise ValueError("the training directory path that you indicated does not exist ! ")
    if not os.path.isdir(os.path.join(dir, flow_dir)):
        raise ValueError("the training directory path that you indicated does not contain the flow folder ! Check your directories.")
    if not os.path.isdir(os.path.join(dir, image_dir)):
        raise ValueError("the training directory path that you indicated does not contain the images folder ! Check your directories.")

    for flow_map in sorted(glob.glob(os.path.join(dir, flow_dir, '*_sat2uav_flow.flo'))):
        sat2uav_flow_map = os.path.join(flow_dir, os.path.basename(flow_map))
        uav2sat_flow_map = os.path.join(flow_dir, os.path.basename(flow_map)[:-17] + "_uav2sat_flow.flo")
        root_filename = os.path.basename(flow_map)[:-17]
        sat_img = os.path.join(image_dir, root_filename + '_img_sat.jpg') # sat image
        uav_img = os.path.join(image_dir, root_filename + '_img_uav.jpg') # uav image
        sat_label = os.path.join(image_dir, root_filename + '_lbl_sat.jpg') # sat label
        uav_label = os.path.join(image_dir, root_filename + '_lbl_uav.jpg') # uav label
        if not (os.path.isfile(os.path.join(dir,sat_img)) or os.path.isfile(os.path.join(dir,sat_img))):
            continue
        if dataset_name is not None:
            images.append([[os.path.join(dataset_name, sat_img),
                            os.path.join(dataset_name, uav_img),
                            os.path.join(dataset_name, sat_label),
                            os.path.join(dataset_name, uav_label),
                            ],
                           [os.path.join(dataset_name, sat2uav_flow_map),
                           os.path.join(dataset_name, uav2sat_flow_map)]])
        else:
            images.append([[sat_img, uav_img, sat_label, uav_label], [sat2uav_flow_map,uav2sat_flow_map]])
    return split2list(images, split, default_split=0.97)


def PreMadeDataset(root, sat_image=None,uav_image=None,sat_label=None,uav_label=None,
                   sat2uav_flow=None,uav2sat_flow=None, split=None):
    # that is only reading and loading the data and applying transformations to both datasets
    if isinstance(root, list):
        train_list=[]
        test_list=[]
        for sub_root in root:
            _, dataset_name = os.path.split(sub_root)
            sub_train_list, sub_test_list = make_dataset(sub_root, split, dataset_name=dataset_name)
            train_list.extend(sub_train_list)
            test_list.extend(sub_test_list)
        root = os.path.dirname(sub_root)
    else:
        train_list, test_list = make_dataset(root, split)
    print('Loading dataset at {}'.format(root))

    train_dataset = ListDataset(root, train_list,
                                sat_image=sat_image,uav_image=uav_image,
                                sat_label=sat_label,uav_label=uav_label,
                                sat2uav_flow=sat2uav_flow,uav2sat_flow=uav2sat_flow)
    test_dataset = ListDataset(root, test_list,
                               sat_image=sat_image, uav_image=uav_image,
                               sat_label=sat_label, uav_label=uav_label,
                               sat2uav_flow=sat2uav_flow, uav2sat_flow=uav2sat_flow)

    return train_dataset, test_dataset
