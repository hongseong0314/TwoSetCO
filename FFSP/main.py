import os
import sys

import logging
from utils.utils import create_logger, copy_all_src
from train import FFSPTrainer

DEBUG_MODE = False
USE_CUDA = True #not DEBUG_MODE
CUDA_DEVICE_NUM = 0

env_params = {
    'stage_cnt': 3,
    'machine_cnt_list': [5, 5, 5],
    'machine_cnt': 5,
    'job_cnt': 30,
    'process_time_params': {
        'time_low': 2,
        'time_high': 20,
    },
    'pomo_size': 24  # assuming 4 machines at each stage! 4*3*2*1
}

model_params = {
    'stage_cnt': env_params['stage_cnt'],
    'machine_cnt_list': env_params['machine_cnt_list'],
    'I':env_params['job_cnt'],
    'J':env_params['machine_cnt'],
    'embedding_dim': 256,
    'sqrt_embedding_dim': 256**(1/2),
    'encoder_layer_num': 3,
    'qkv_dim': 16,
    'sqrt_qkv_dim': 16**(1/2),
    'head_num': 16,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'ms_hidden_dim': 12,
    'ms_layer1_init': (1/2)**(1/2),
    'ms_layer2_init': (1/12)**(1/2),
    'sqrt_ms_dim': 12**(1/2),
    'problem_size': 2,
    'eval_type': 'argmax',
    'one_hot_seed_cnt': env_params['machine_cnt'],  # must be >= machine_cnt
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6, 
    },
    'scheduler': {
        'milestones': [101, 151],
        'gamma': 0.1
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 100,
    'train_episodes': 1*1000,
    'train_batch_size': 4,
    'logging': {
        'model_save_interval': 100,
        'img_save_interval': 100,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss.json'
        },
    },
    'model_load': {
        'enable': False,  # enable loading pre-trained model
        # 'path': './result/saved_ffsp_model',  # directory path of pre-trained model and log files saved.
        # 'epoch': 100,  # epoch version of pre-trained model to load.
    }
}

logger_params = {
    'log_file': {
        'desc': 'matnet_train',
        'filename': 'log.txt'
    }
}


##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    trainer = FFSPTrainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)

    copy_all_src(trainer.result_folder)

    trainer.run()

    if DEBUG_MODE:
        # Print Scehdule for last batch problem
        # env.print_schedule()
        pass


def _set_debug_mode():
    global env_params
    env_params['job_cnt'] = 5

    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 4
    trainer_params['train_batch_size'] = 2
    trainer_params['validate_episodes'] = 4
    trainer_params['validate_batch_size'] = 2

def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]

##########################################################################################
if __name__ == "__main__":
    main()