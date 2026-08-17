from __future__ import print_function

import argparse
import datetime
import logging
import os
import torch
import yaml

import torch.distributed as dist

from torch.distributed.elastic.multiprocessing.errors import record

from local.me.utils.executor import Executor
from wenet.utils.config import override_config
from local.me.utils.init_model import init_model
from local.me.utils.init_tokenizer import init_tokenizer
from local.me.utils.train_utils import (
    add_model_args, add_dataset_args, add_ddp_args, add_deepspeed_args,
    add_trace_args, init_distributed,
    check_modify_and_save_config, init_optimizer_and_scheduler,
    trace_and_print_model, wrap_cuda_model, init_summarywriter, save_model,
    log_per_epoch, init_dataset_and_dataloader)


def get_save_model_flag(loss_dict, catch_dict, save_model_num, epoch):
    partens = ['acc', 'f_score', 'p_score', 'r_score']
    save_flag = False
    for parten in partens:
        flag = False
        catch_scores = catch_dict[parten + '_arr']
        this_score = loss_dict[parten]
        # if isinstance(this_score, torch.Tensor):
        #     this_score = this_score.item()
        if len(catch_scores) < save_model_num:
            flag = True
            save_flag = True
            catch_scores.append((epoch, this_score))
            catch_scores = sorted(catch_scores, key=lambda x:x[1], reverse=True)
        else:
            min_score = catch_scores[-1][-1]
            if this_score > min_score:
                flag = True
                save_flag = True
                catch_scores[-1] = (epoch, this_score)
                catch_scores = sorted(catch_scores, key=lambda x:x[1], reverse=True)
        if flag:
            catch_dict[parten + '_arr'] = catch_scores
    return save_flag, catch_dict


def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--train_engine',
                        default='torch_ddp',
                        choices=['torch_ddp', 'deepspeed'],
                        help='Engine for paralleled training')
    parser.add_argument('--save_model_num',
                        default=20,
                        help='save how many models')
    parser = add_model_args(parser)
    parser = add_dataset_args(parser)
    parser = add_ddp_args(parser)
    parser = add_deepspeed_args(parser)
    parser = add_trace_args(parser)
    args = parser.parse_args()
    if args.train_engine == "deepspeed":
        args.deepspeed = True
        assert args.deepspeed_config is not None
    return args


# NOTE(xcsong): On worker errors, this recod tool will summarize the
#   details of the error (e.g. time, rank, host, pid, traceback, etc).
@record
def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')

    # Set random seed
    torch.manual_seed(777)

    # Read config
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)

    # init tokenizer
    tokenizer = init_tokenizer(configs)

    # Init env for ddp OR deepspeed
    _, _, rank = init_distributed(args)

    # Get dataset & dataloader
    train_dataset, cv_dataset, train_data_loader, cv_data_loader = \
        init_dataset_and_dataloader(args, configs, tokenizer)

    # Do some sanity checks and save config to arsg.model_dir
    configs = check_modify_and_save_config(args, configs,
                                           tokenizer.symbol_table)

    # Init asr model from configs
    model, configs = init_model(args, configs)

    # Check model is jitable & print model archtectures
    trace_and_print_model(args, model)

    # Tensorboard summary
    writer = init_summarywriter(args)

    # Dispatch model from cpu to gpu
    model, device = wrap_cuda_model(args, model)

    # Get optimizer & scheduler
    model, optimizer, scheduler = init_optimizer_and_scheduler(
        args, configs, model)

    # Save checkpoints
    # save_model(model,
    #            info_dict={
    #                "save_time":datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
    #                "tag":"init",
    #                **configs
    #            })

    # Get executor
    tag = configs["init_infos"].get("tag", "init")
    executor = Executor(global_step=configs["init_infos"].get('step', -1) + int("step_" in tag))

    # Init scaler, used for pytorch amp mixed precision training
    scaler = None
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    # Start training loop
    start_epoch = configs["init_infos"].get('epoch', 0) + int("epoch_" in tag)
    # if save_interval in configs, steps mode else epoch mode
    end_epoch = configs.get('max_epoch',
                            100) if "save_interval" not in configs else 1
    assert start_epoch <= end_epoch
    configs.pop("init_infos", None)
    final_epoch = None

    #me
    save_model_num = args.save_model_num
    catch_dict = dict(
                acc_arr = [],
                f_score_arr = [],
                p_score_arr = [],
                r_score_arr = [])

    for epoch in range(start_epoch, end_epoch):
        configs['epoch'] = epoch

        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {} rank {}'.format(
            epoch, lr, rank))

        dist.barrier(
        )  # NOTE(xcsong): Ensure all ranks start Train at the same time.
        # NOTE(xcsong): Why we need a new group? see `train_utils.py::wenet_join`
        group_join = dist.new_group(
            backend="gloo", timeout=datetime.timedelta(seconds=args.timeout))
        executor.train(model, optimizer, scheduler, train_data_loader,
                       cv_data_loader, writer, configs, scaler, group_join)
        dist.destroy_process_group(group_join)

        dist.barrier(
        )  # NOTE(xcsong): Ensure all ranks start CV at the same time.
        loss_dict = executor.cv(model, cv_data_loader, configs)

        lr = optimizer.param_groups[0]['lr']
        try:
            logging.info('Epoch {} CV info lr {} cv_loss {} rank {} acc {} P {} R {} F {}'.format(
                epoch, lr, loss_dict["loss"], rank, loss_dict["acc"], loss_dict["p_score"], loss_dict["r_score"],loss_dict["f_score"]))
        except:
            logging.info('Epoch {} CV info lr {} cv_loss {} rank {} acc {}'.format(
                epoch, lr, loss_dict["loss"], rank, loss_dict["acc"]))
        info_dict = {
            'epoch': epoch,
            'lr': lr,
            'step': executor.step,
            'save_time': datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
            'tag': "epoch_{}".format(epoch),
            'loss_dict': loss_dict,
            **configs
        }
        log_per_epoch(writer, info_dict=info_dict)
        save_flag, catch_dict = get_save_model_flag(loss_dict, catch_dict, save_model_num, epoch)
        if save_flag:
            logging.info('save model this epoch!')
            save_model(model, info_dict=info_dict)

        final_epoch = epoch

    if final_epoch is not None and rank == 0:
        final_model_path = os.path.join(args.model_dir, 'final.pt')
        os.remove(final_model_path) if os.path.exists(
            final_model_path) else None
        os.symlink('{}.pt'.format(final_epoch), final_model_path)
        writer.close()

        logging.info('save model info:')
        for key, value in catch_dict.items():
            for i, info in enumerate(value):
                value[i] = f'{info[0]}: {info[1]}'
            logging.info(f'{key}: \n{", ".join(value)}')

if __name__ == '__main__':
    main()
