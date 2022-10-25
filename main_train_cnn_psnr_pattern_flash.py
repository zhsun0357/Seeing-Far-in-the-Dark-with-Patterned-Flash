import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model
import pdb

import os
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
from tqdm import tqdm

'''
# --------------------------------------------
# training code for MSRResNet
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/KAIR
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


def main(json_path='options/train_msrresnet_psnr.json'):
    
    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()
    
    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if ('pretrained' not in key) and ("restore" not in key)))
    ## copy scripts
    if not os.path.exists(os.path.join(opt['path']['log'], 'scripts')):
        os.mkdir(os.path.join(opt['path']['log'], 'scripts'))
    # pdb.set_trace()
    os.system("cp -r /home/szh/cnn_models_fad/ {}".format(os.path.join(opt['path']['log'], 'scripts')))
    
    # ----------------------------------------
    # update opt, automatically restart from the last checkpoint
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    # pdb.set_trace()
    if opt['restore']:
        init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
        init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
        # pdb.set_trace()
        opt['path']['pretrained_netG'] = init_path_G
        opt['path']['pretrained_netE'] = init_path_E
        init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
        opt['path']['pretrained_optimizerG'] = init_path_optimizerG
        current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)
    else:
        current_step = 0
    
    border = opt['scale']
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''
    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''
    
    # pdb.set_trace()
    model = define_Model(opt)
    model.init_train()
    if opt['rank'] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''

    for epoch in range(1000000):  # keep running
        for i, train_data in tqdm(enumerate(train_loader)):
            # pdb.set_trace()
            current_step += 1

            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            # pdb.set_trace()
            model.feed_data(train_data)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            # -------------------------------
            # 6) testing
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:

                avg_psnr = 0.0
                idx = 0

                for test_data in test_loader:
                    idx += 1
                    image_name_ext = os.path.basename(test_data['L_path'][0])
                    img_name, ext = os.path.splitext(image_name_ext)

                    img_dir = os.path.join(opt['path']['images'], img_name)
                    util.mkdir(img_dir)

                    # pdb.set_trace()
                    model.feed_data(test_data)
                    model.test()

                    visuals = model.current_visuals()
                    # E_img = util.tensor2uint(visuals['E'])
                    # H_img = util.tensor2uint(visuals['H'])

                    # -----------------------
                    # save estimated image E
                    # -----------------------
                    # save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
                    # util.imsave(E_img, save_img_path)
                    if ((current_step % (opt['train']['checkpoint_test']*10)) == 0) or (current_step <= opt['train']['checkpoint_test']*5):
                        np.save(os.path.join(img_dir, \
                        'restore_{:s}_{:d}.npy'.format(img_name, current_step)), visuals['E'].detach().cpu().numpy())
                        np.save(os.path.join(img_dir, \
                        'gt_{:s}_{:d}.npy'.format(img_name, current_step)), visuals['H'].detach().cpu().numpy())
                        np.save(os.path.join(img_dir, \
                        'inp_{:s}_{:d}.npy'.format(img_name, current_step)), test_data['L'].detach().cpu().numpy())  
                        if 'P'in test_data.keys():
                            np.save(os.path.join(img_dir, \
                            'pattern_{:s}_{:d}.npy'.format(img_name, current_step)), test_data['P'].detach().cpu().numpy())
                        if 'S'in test_data.keys():
                            np.save(os.path.join(img_dir, \
                            'scale_{:s}_{:d}.npy'.format(img_name, current_step)), test_data['S'].detach().cpu().numpy())
                        if 'N' in test_data.keys():
                            np.save(os.path.join(img_dir, \
                            'noise_{:s}_{:d}.npy'.format(img_name, current_step)), test_data['N'])
                        if 'F' in test_data.keys():
                            np.save(os.path.join(img_dir, \
                            'flash_{:s}_{:d}.npy'.format(img_name, current_step)), test_data['F'].detach().cpu().numpy()) 
                        if 'D' in test_data.keys():
                            np.save(os.path.join(img_dir, \
                            'depth_{:s}_{:d}.npy'.format(img_name, current_step)), test_data['D'].detach().cpu().numpy()) 
                        if 'K' in test_data.keys():
                            np.save(os.path.join(img_dir, \
                            'kernel_{:s}_{:d}.npy'.format(img_name, current_step)), test_data['K'].detach().cpu().numpy()) 
                    
                    # -----------------------
                    # calculate PSNR
                    # -----------------------
                    # current_psnr = util.calculate_psnr(E_img, H_img, border=border)

                    # logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name_ext, current_psnr))

                    # avg_psnr += current_psnr

                avg_psnr = avg_psnr / idx

                # testing log
                logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB\n'.format(epoch, current_step, avg_psnr))

if __name__ == '__main__':
    main()
