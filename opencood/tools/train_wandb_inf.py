# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
from math import e
import os
import re
import statistics
import subprocess
import wandb
import torch
import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
import yaml

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.tools import multi_gpu_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.tools.inference_fun import inference_once

def run_inference_and_log_map(model_dir, epoch):
    cmd = [
        "xvfb-run", "-a",
        "python",
        "/home/ecoprt/cooperative_pers/OpenCOOD/opencood/tools/inference.py",
        "--model_dir", model_dir,
        "--fusion_method", "intermediate",
        "--show_sequence"
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output, _ = process.communicate()

    print(f'Inference output: {output}')


    match = re.search(r'IOU 0.3 is (\d\.\d+).*IOU 0.5 is (\d\.\d+).*IOU 0.7 is (\d\.\d+)', output)
    if match:
        iou_03, iou_05, iou_07 = [float(x) for x in match.groups()]
        print(f"mAP@0.3: {iou_03}, mAP@0.5: {iou_05}, mAP@0.7: {iou_07}")
        wandb.log({
            "mAP@0.3": iou_03,
            "mAP@0.5": iou_05,
            "mAP@0.7": iou_07,
            "epoch": epoch
        })


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help='data generation yaml file needed ')
    # parser.add_argument('--model_dir', default='',
    #                     help='Continued training path')
    parser.add_argument("--half", action='store_true',
                        help="whether train with half precision.")
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    opt, unknown = parser.parse_known_args()
    return opt, unknown


def update_yaml_with_args(hypes, params_dict):
    for key, value in params_dict.items():
        keys = key.split('.')
        d = hypes
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        if isinstance(d.get(keys[-1]), dict) and isinstance(value, dict):
            d[keys[-1]].update(value)
        else:
            d[keys[-1]] = value
    return hypes


def param_to_dict(param:list):
    param_dict = {}
    for p in param:
        key, value = p.split('=', 1)
        value_dict = yaml.safe_load(value)
        param_dict[key] = value_dict
    return param_dict

def main():
    opt, unknown_params = train_parser()
    unknown_params_dict = param_to_dict(unknown_params)

    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    # print(f'Loaded hypes: {hypes}')
    wandb.init(project="rinne-former_half", config=unknown_params_dict)
    config = wandb.config
    # print(config)
    # Update hypes with command line arguments
    hypes = update_yaml_with_args(hypes, config)
    print(f'Updated hypes: {hypes}')


    multi_gpu_utils.init_distributed_mode(opt)

    print('-----------------Dataset Building------------------')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes, visualize=False, train=False)

    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset)
        sampler_val = DistributedSampler(opencood_validate_dataset,
                                         shuffle=False)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes['train_params']['batch_size'], drop_last=True)

        train_loader = DataLoader(opencood_train_dataset,
                                  batch_sampler=batch_sampler_train,
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train)
        val_loader = DataLoader(opencood_validate_dataset,
                                sampler=sampler_val,
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                drop_last=False)
    else:
        train_loader = DataLoader(opencood_train_dataset,
                                  batch_size=hypes['train_params']['batch_size'],
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train,
                                  shuffle=True,
                                  pin_memory=False,
                                  drop_last=True)
        val_loader = DataLoader(opencood_validate_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                shuffle=False,
                                pin_memory=False,
                                drop_last=True)

    print('---------------Creating Model------------------')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if we want to train from last checkpoint.
    # if opt.model_dir:
    #     saved_path = opt.model_dir
    #     init_epoch, model = train_utils.load_saved_model(saved_path,
    #                                                      model)

    # else:
    #     init_epoch = 0
    #     # if we train the model from scratch, we need to create a folder
    #     # to save the model,
    #     saved_path = train_utils.setup_train(hypes)

    init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
    saved_path = train_utils.setup_train(hypes)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
    model_without_ddp = model

    if opt.distributed:
        model = \
            torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[opt.gpu],
                                                      find_unused_parameters=True)
        model_without_ddp = model.module

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    # lr scheduler setup
    num_steps = len(train_loader)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps)

    # record training
    writer = SummaryWriter(saved_path)

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    print('Training start')
    epoches = hypes['train_params']['epoches']
    # used to help schedule learning rate

    initial_val_loss = 10.0
    wandb.log({"val_loss": initial_val_loss, "epoch": 0})
    
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
            scheduler.step(epoch)
        if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
            scheduler.step_update(epoch * num_steps + 0)
        for param_group in optimizer.param_groups:
            print('learning rate %.7f' % param_group["lr"])
        
        current_lr = optimizer.param_groups[0]["lr"]
        wandb.log({"learning_rate": current_lr, "epoch": epoch})

        if opt.distributed:
            sampler_train.set_epoch(epoch)

        pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)

        for i, batch_data in enumerate(train_loader):
            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            batch_data = train_utils.to_device(batch_data, device)

            # case1 : late fusion train --> only ego needed,
            # and ego is random selected
            # case2 : early fusion train --> all data projected to ego
            # case3 : intermediate fusion --> ['ego']['processed_lidar']
            # becomes a list, which containing all data from other cavs
            # as well
            if not opt.half:
                ouput_dict = model(batch_data['ego'])
                # first argument is always your output dictionary,
                # second argument is always your label dictionary.
                final_loss = criterion(ouput_dict,
                                       batch_data['ego']['label_dict'])
            else:
                with torch.cuda.amp.autocast():
                    ouput_dict = model(batch_data['ego'])
                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                    
            reg_loss = criterion.loss_dict['reg_loss'].item()
            conf_loss = criterion.loss_dict['conf_loss'].item()
            total_loss = criterion.loss_dict['total_loss'].item()
            wandb.log({
                "Confidence_loss": conf_loss,
                "Regression_loss": reg_loss,
                "Total_loss": total_loss,
                "epoch": epoch
            })
            criterion.logging(epoch, i, len(train_loader), writer, pbar=pbar2)
            pbar2.update(1)

            if not opt.half:
                final_loss.backward()
                optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
                scheduler.step_update(epoch * num_steps + i)

            # Log metrics to wandb
            wandb.log({"train_loss": final_loss.item(), "epoch": epoch})

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model_without_ddp.state_dict(),
                os.path.join(saved_path, 'net_epoch%d.pth' % (epoch + 1)))

        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    ouput_dict = model(batch_data['ego'])

                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                    valid_ave_loss.append(final_loss.item())
            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                              valid_ave_loss))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)
            wandb.log({"val_loss": valid_ave_loss, "epoch": epoch})

            if epoch >= 5:
                ap30, ap50, ap70 = inference_once(saved_path, fusion_method='intermediate', show_sequence=False)

                wandb.log({
                    "mAP@0.3": ap30,
                    "mAP@0.5": ap50,
                    "mAP@0.7": ap70,
                    "epoch": epoch
                })



    print('Training Finished, checkpoints saved to %s' % saved_path)
    wandb.finish()

if __name__ == '__main__':
    main()
