import numpy as np
import os,cv2,random,csv
import pandas as pd
from pathlib import Path
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import test_aff_tps_homo

def read_images(dataset_path):
    p = Path(dataset_path)
    FileList = list(p.glob(("**/*.jpg"))) + list(p.glob(("**/*.png")))
    return FileList


def generate_theta(sat_image_name, uav_image_name, corner,random_t_tps = 0.4):
    # transform_dict = {0: 'aff', 1: 'tps', 2: 'homo'}  #aff+homo+tps
    transform_dict = {0: 'aff', 2: 'homo'}#aff+homo
    # create training theta with affine/homography/tps/affine+tps
    transform_type = random.randint(0, 1) * 2
    print(transform_dict[transform_type])
    # transform_type = 1
    # affine
    if transform_type == 0:
        rot_angle = (np.random.rand(1) - 0.5) * 2 * np.pi / 12  # between -np.pi/12 and np.pi/12
        sh_angle = (np.random.rand(1) - 0.5) * 2 * np.pi / 6  # between -np.pi/6 and np.pi/6
        lambda_1 = 1 + (2 * np.random.rand(1) - 1) * 0.25  # between 0.75 and 1.25
        lambda_2 = 1 + (2 * np.random.rand(1) - 1) * 0.25  # between 0.75 and 1.25
        tx = (2 * np.random.rand(1) - 1) * 0.25  # between -0.25 and 0.25
        ty = (2 * np.random.rand(1) - 1) * 0.25

        R_sh = np.array([[np.cos(sh_angle[0]), -np.sin(sh_angle[0])],
                         [np.sin(sh_angle[0]), np.cos(sh_angle[0])]])
        R_alpha = np.array([[np.cos(rot_angle[0]), -np.sin(rot_angle[0])],
                            [np.sin(rot_angle[0]), np.cos(rot_angle[0])]])

        D = np.diag([lambda_1[0], lambda_2[0]])

        A = R_alpha @ R_sh.transpose() @ D @ R_sh

        theta_aff = np.array([A[0, 0], A[0, 1], tx, A[1, 0], A[1, 1], ty])
    # homography
    elif transform_type == 2:
        theta_hom = np.array([-1, -1, 1, 1, -1, 1, -1, 1])
        theta_hom = theta_hom + (np.random.rand(8) * 0.4 - 0.2) * 2 * random_t_tps
        theta_hom = homography_mat_from_4_pts(Variable(torch.Tensor(theta_hom.reshape(1,-1)))).data.cpu().numpy()
    elif transform_type == 1:
        theta_tps = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1])
        theta_tps = theta_tps + (np.random.rand(18) * 0.4 - 0.2) * 2 * random_t_tps

    if transform_type == 0:
        theta = theta_aff
    elif transform_type == 1:
        theta = theta_tps
    elif transform_type == 2:
        theta = theta_hom
    # elif transform_type == 3:
    #     theta = np.concatenate((theta_aff, theta_tps))
    return transform_type, theta



def generate_csv(dataset_path, patch_size,train_path, csv_path):
    """

    Args:
        dataset_path: 原始影像
        patch_size: 256
        train_path:切分好的影像存储路径
        csv_path: csv的存储路径

    Returns:

    """
    uav_file_list = read_images(dataset_path + "uav/")
    sat_file_list = read_images(dataset_path + "sat/")
    new_records = []
    count = 0
    for i in range(len(sat_file_list)):
        sat_img = cv2.cvtColor(cv2.imread(str(sat_file_list[i])), cv2.COLOR_BGR2RGB)
        # sat_img = cv2.cvtColor(cv2.imread(os.fspath(i)), cv2.COLOR_BGR2RGB)
        # 每张5120*5120的大图切成200个小图
        # 每张12800*10240的大图切成400个小图
        for j in range(400):
            # random cropped the original image into 256*256
            left_top_x = random.randint(0, sat_img.shape[0] - patch_size)
            left_top_y = random.randint(0, sat_img.shape[0] - patch_size)#train
            # left_top_y = random.randint(sat_img.shape[0] - patch_size,sat_img.shape[1] - patch_size)#test

            #
            corner = left_top_x,left_top_x + patch_size, left_top_y,left_top_y + patch_size
            sat_patch = sat_img[left_top_x:left_top_x + patch_size, left_top_y:left_top_y + patch_size, :]

            # patch_name = train_path + os.fspath(i).split('/')[-1][:-4] + "_" + str(j)+os.fspath(i)[-4:]
            # cv2.imwrite(patch_name,img_patch)
            sat_image_name = str(sat_file_list[i])
            uav_image_name = os.fspath(uav_file_list[random.randint(0,len(uav_file_list)-1)])
            uav_img = cv2.cvtColor(cv2.imread(uav_image_name), cv2.COLOR_BGR2RGB)
            uav_patch = uav_img[left_top_x:left_top_x + patch_size, left_top_y:left_top_y + patch_size, :]
            theta_save = [0 for i in range(18)]
            new_record = [sat_image_name, uav_image_name, corner]
            transform_type_i,theta_i = generate_theta(sat_image_name, uav_image_name, corner)
            count +=1
            theta_i = theta_i.reshape(1,-1)
            new_record.append(transform_type_i)
            for a in range(theta_i.shape[1]):
                theta_save[a] = theta_i[0,a]
            for b in range(18):
                new_record.append(theta_save[b])
            # new_record.append(name)
            print(new_record)
            # warped_image_patch = test_transform(sat_image_name, uav_image_name, corner, transform_type_i, theta_save)
            # cv2.imwrite("tmp/uav_patch_" + str(i*4+j) + ".jpg",uav_patch)
            # cv2.imwrite("tmp/sat_patch_" + str(i*4+j) + ".jpg",sat_patch)
            # cv2.imwrite("tmp/warp_patch_" + str(i*4+j) + ".jpg",warped_image_patch.data.cpu().numpy()[0].transpose(1,2,0))
            new_records.append(new_record)

    with open(csv_path,"w+",newline='') as f:
        csv_file = csv.writer(f)
        csv_file.writerow(["sat_image_name","uav_image_name", "corner(ltx,rtx,lty,rty)","aff/tps/homo",
                           "theta0","theta1","theta2","theta3","theta4","theta5","theta6","theta7","theta8",
                           "theta9","theta10","theta11","theta12","theta13","theta14","theta15","theta16","theta17"])
        csv_file.writerows(new_records)


def homography_mat_from_4_pts(theta):
    b = theta.size(0)
    if not theta.size() == (b, 8):
        theta = theta.view(b, 8)
        theta = theta.contiguous()

    xp = theta[:, :4].unsqueeze(2);
    yp = theta[:, 4:].unsqueeze(2)

    x = Variable(torch.FloatTensor([-1, -1, 1, 1])).unsqueeze(1).unsqueeze(0).expand(b, 4, 1)
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

    A = torch.cat([torch.cat([-x, -y, -o, z, z, z, x * xp, y * xp, xp], 2),
                   torch.cat([z, z, z, -x, -y, -o, x * yp, y * yp, yp], 2)], 1)
    # find homography by assuming h33 = 1 and inverting the linear system
    h = torch.bmm(torch.inverse(A[:, :, :8]), -A[:, :, 8].unsqueeze(2))
    # add h33
    h = torch.cat([h, single_o], 1)

    H = h.squeeze(2)

    return H

def test_transform(sat_image_name, uav_image_name,corner,transform_type,theta):
    left_top_x = int(corner[0])
    right_top_x = int(corner[1])
    left_top_y = int(corner[2])
    right_top_y = int(corner[3])
    # sat_img = torch.Tensor(cv2.imread(sat_image_name)[left_top_x:right_top_x, left_top_y:right_top_y, :]).transpose(1,2).transpose(0,1)
    uav_img = torch.Tensor(cv2.imread(uav_image_name)[left_top_x:right_top_x, left_top_y:right_top_y, :]).transpose(1,2).transpose(0,1)
    out_h = 256
    out_w = 256
    if transform_type==0:
        theta = theta[:6]
        gridGen = test_aff_tps_homo.AffineGridGen(out_h, out_w, use_cuda=False)
    elif transform_type == 2:
        # theta = torch.Tensor(theta.astype(np.float32))
        # theta = homography_mat_from_4_pts(Variable(theta.unsqueeze(0))).squeeze(0).data
        theta = theta[:9]
        gridGen = test_aff_tps_homo.HomographyGridGen(out_h, out_w, use_cuda=False)
    elif transform_type == 1:
        gridGen = test_aff_tps_homo.TpsGridGen(out_h, out_w, use_cuda=False)

    sampling_grid = gridGen(theta)


    # sample transformed image
    warped_image_patch = F.grid_sample(uav_img.unsqueeze(0), sampling_grid)

    return warped_image_patch



def load_csv(csv_path):
    rows_csv = pd.read_csv(csv_path)
    for i in rows_csv.index:
        row = rows_csv.loc[i]
        sat_image_name = row[0]
        uav_image_name = row[1]
        corner = row[2][1:-1].split(',')
        left_top_x = corner[0]
        right_top_x = corner[1]
        left_top_y = corner[2]
        right_top_y = corner[3]
        transform_type = row[3]
        theta = row.values[4:]
        if transform_type == 0:
            theta_aff = theta[:6].astype('float').reshape(2, 3)
            theta = theta_aff
        elif transform_type == 1:
            theta_tps = theta[:18].astype('float')
            theta = theta_tps
        elif transform_type == 2:
            theta_homo = theta[:9].astype('double')
            theta = theta_homo
        warped_image_patch = test_transform(sat_image_name, uav_image_name,corner,transform_type,theta)
        # return warped_image_patch


if __name__ == '__main__':
    dataset_path = "../../GLUnet/sat_dataset/sat_data/training-virigina/"
    patch_size = 256    #每个patch的大小
    train_path = "../dataset/train/"
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    # csv_path = "csv_files/aff_homo_tps_train.csv"
    # csv_path = "csv_files/aff_homo_tps_test.csv"
    # csv_path = "csv_files/aff_homo_test.csv"
    csv_path = "csv_files/aff_homo_train.csv"
    generate_csv(dataset_path,patch_size,train_path, csv_path)

    # _ = load_csv(csv_path)
