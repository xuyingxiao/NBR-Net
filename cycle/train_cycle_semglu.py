import numpy as np
import argparse
import time
import random
import os,shutil,sys
sys.path.append("../")
from os import path as osp
from termcolor import colored
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler
from dataset_utils.image_transforms import ArrayToTensor
from dataset_utils.load_pre_made_dataset import PreMadeDataset
from cycle.cycle_all_loss_optimize import train_epoch, validate_epoch
from cycle.semglu import SemanticGLUNet_model
from models.utils.utils_CNN import load_checkpoint, save_checkpoint, boolean_string
from tensorboardX import SummaryWriter

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='cycle semglu train script')
    # Paths
    parser.add_argument('--name_exp', type=str,
                        default=time.strftime('%Y_%m_%d_%H_%M'),
                        help='name of the experiment to save')
    parser.add_argument('--training_data_dir', default="../dataset/train/", type=str,
                        help='path to directory containing original images for training if --pre_loaded_training_'
                             'dataset is False or containing the synthetic pairs of training images and their '
                             'corresponding flow fields if --pre_loaded_training_dataset is True')
    parser.add_argument('--evaluation_data_dir', default="../dataset/test/", type=str,
                        help='path to directory containing original images for validation if --pre_loaded_training_'
                             'dataset is False or containing the synthetic pairs of validation images and their '
                             'corresponding flow fields if --pre_loaded_training_dataset is True')
    parser.add_argument('--snapshots', type=str, default='../snapshots/cycle/')
    parser.add_argument('--pretrained', dest='pretrained', default=None,
                        help='path to pre-trained model')
    # Optimization parameters
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--momentum', type=float,
                        default=4e-4, help='momentum constant')
    parser.add_argument('--start_epoch', type=int, default=-1,
                        help='start epoch')
    parser.add_argument('--n_epoch', type=int, default=300,
                        help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size')
    parser.add_argument('--n_threads', type=int, default=4,
                        help='number of parallel threads for dataloaders')
    parser.add_argument('--weight-decay', type=float, default=4e-4,
                        help='weight decay constant')
    parser.add_argument('--div_flow', type=float, default=1.0,
                        help='div flow')
    parser.add_argument('--seed', type=int, default=1986,
                        help='Pseudo-RNG seed')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # datasets, pre-processing of the images is done within the network function !
    sat_img = transforms.Compose([ArrayToTensor(get_float=False)])
    uav_img = transforms.Compose([ArrayToTensor(get_float=False)])
    sat_lbl = transforms.Compose([ArrayToTensor(get_float=False)])
    uav_lbl = transforms.Compose([ArrayToTensor(get_float=False)])

    #loaded dataset
    sat2uav_flow = transforms.Compose([ArrayToTensor()])  # just put channels first and put it to float
    uav2sat_flow = transforms.Compose([ArrayToTensor()])  # just put channels first and put it to float
    train_dataset, _ = PreMadeDataset(root=args.training_data_dir,
                                      sat_image=sat_img,
                                      uav_image=uav_img,
                                      sat_label=sat_lbl,
                                      uav_label=uav_lbl,
                                      sat2uav_flow=sat2uav_flow,
                                      uav2sat_flow=uav2sat_flow,
                                      split=1)  # only training

    _, val_dataset = PreMadeDataset(root=args.evaluation_data_dir,
                                    sat_image=sat_img,
                                    uav_image=uav_img,
                                    sat_label=sat_lbl,
                                    uav_label=uav_lbl,
                                    sat2uav_flow=sat2uav_flow,
                                    uav2sat_flow=uav2sat_flow,
                                    split=0)  # only validation

    # Dataloader
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.n_threads)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.n_threads)

    model = SemanticGLUNet_model(batch_norm=True, pyramid_type='VGG',
                                 div=args.div_flow, evaluation=False,
                                 cyclic_consistency=False, consensus_network=True)

    print(colored('==> ', 'blue') + 'cycle supervised semantic ori GLU-Net created.')

    # Optimizer
    optimizer = \
        optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                   lr=args.lr,
                   weight_decay=args.weight_decay)

    scheduler = lr_scheduler.MultiStepLR(optimizer,
                                         milestones=[65, 100, 150],
                                         gamma=0.5)
    weights_loss_coeffs = [0.32, 0.08, 0.02, 0.01]
    # weights_loss_coeffs = [0.2, 0.2, 0.2, 0.4]

    if args.pretrained:
        # reload from pre_trained_model
        model, optimizer, scheduler, start_epoch, best_val = load_checkpoint(model, optimizer, scheduler,
                                                                             filename=args.pretrained)
        # now individually transfer the optimizer parts...
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        cur_snapshot = os.path.basename(os.path.dirname(args.pretrained))
    else:
        if not os.path.isdir(args.snapshots):
            os.makedirs(args.snapshots)

        cur_snapshot = args.name_exp
        if not osp.isdir(osp.join(args.snapshots, cur_snapshot)):
            os.makedirs(osp.join(args.snapshots, cur_snapshot))

        with open(osp.join(args.snapshots, cur_snapshot, 'args.pkl'), 'wb') as f:
            pickle.dump(args, f)

        best_val = float("inf")
        start_epoch = 0

    # create summary writer
    save_path = osp.join(args.snapshots, cur_snapshot)
    save_code_path = os.path.join(save_path, 'code')
    if not os.path.exists(save_code_path):
        os.makedirs(save_code_path)

    shutil.copy2("train_cycle_semglu.py",os.path.join(save_code_path, "train_cycle_semglu.py"))
    shutil.copy2("cycle_all_loss_optimize.py",os.path.join(save_code_path, "cycle_all_loss_optimize.py"))
    shutil.copy2("semglu.py",os.path.join(save_code_path, "semglu.py"))
    shutil.copy2("loss.py",os.path.join(save_code_path, "loss.py"))

    train_writer = SummaryWriter(os.path.join(save_path, 'train'))
    test_writer = SummaryWriter(os.path.join(save_path, 'test'))

    # model = nn.DataParallel(model)
    model = nn.DataParallel(model)
    model = model.to(device)

    train_started = time.time()
    epoch_save_list = []
    for epoch in range(start_epoch, args.n_epoch):
        print('starting epoch {}:  learning rate is {}'.format(epoch, scheduler.get_lr()[0]))

        # Training one epoch
        train_loss,train_epe_loss,train_iou_loss = train_epoch(model,
                                 optimizer,
                                 train_dataloader,
                                 device,
                                 epoch,
                                 train_writer,
                                 div_flow=args.div_flow,
                                 save_path=os.path.join(save_path, 'train'),
                                 loss_grid_weights=weights_loss_coeffs)
        train_writer.add_scalar('train loss', train_loss, epoch)
        train_writer.add_scalar('train epe loss', train_epe_loss, epoch)
        train_writer.add_scalar('train iou loss', train_iou_loss, epoch)
        train_writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], epoch)
        print(colored('==> ', 'green') + 'Train average loss:', train_loss)

        # Validation
        val_loss_grid, val_epe_loss, val_iou_loss, \
        val_sat2uav_step1_epe, val_sat2uav_step1_epe_H_8, val_sat2uav_step1_epe_32, val_sat2uav_step1_epe_16,\
        val_uav2sat_step2_epe, val_uav2sat_step2_epe_H_8, val_uav2sat_step2_epe_32, val_uav2sat_step2_epe_16= \
            validate_epoch(model, val_dataloader, device, epoch=epoch,test_writer=test_writer,
                           save_path=os.path.join(save_path, 'test'),
                           div_flow=args.div_flow,
                           loss_grid_weights=weights_loss_coeffs)

        print(colored('==> ', 'blue') + 'Val average grid loss :',
              val_loss_grid)
        print('step1:')
        print('mean EPE is {}'.format(val_sat2uav_step1_epe))
        print('mean EPE from reso H/8 is {}'.format(val_sat2uav_step1_epe_H_8))
        print('mean EPE from reso 32 is {}'.format(val_sat2uav_step1_epe_32))
        print('mean EPE from reso 16 is {}'.format(val_sat2uav_step1_epe_16))
        print('step2:')
        print('mean EPE is {}'.format(val_uav2sat_step2_epe))
        print('mean EPE from reso H/8 is {}'.format(val_uav2sat_step2_epe_H_8))
        print('mean EPE from reso 32 is {}'.format(val_uav2sat_step2_epe_32))
        print('mean EPE from reso 16 is {}'.format(val_uav2sat_step2_epe_16))
        print(colored('==> ', 'blue') + 'finished epoch :', epoch + 1)

        val_mean_epe = val_sat2uav_step1_epe
        # save checkpoint for each epoch and a fine called best_model so far
        is_best = val_mean_epe < best_val
        best_val = min(val_mean_epe, best_val)
        if epoch % 10 ==0 or is_best:
            save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.module.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(),
                         'best_loss': best_val},
                        is_best, save_path, 'epoch_{}.pth'.format(epoch + 1))
            epoch_save_list.append(epoch+1)

        print(epoch_save_list)
        scheduler.step()
        if len(epoch_save_list)>5:
            print(save_path + "/epoch_" + str(epoch_save_list[0]) + ".pth")
            os.remove(save_path + "/epoch_" + str(epoch_save_list[0]) + ".pth")
            del epoch_save_list[0]
        print(epoch_save_list)

    print(args.seed, 'Training took:', time.time() - train_started, 'seconds')