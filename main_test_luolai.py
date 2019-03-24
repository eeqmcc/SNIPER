# --------------------------------------------------------------
# SNIPER: Efficient Multi-Scale Training
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Inference Module
# by Mahyar Najibi and Bharat Singh
# --------------------------------------------------------------
import init
import matplotlib
import time
import shutil
matplotlib.use('Agg')
from symbols.faster import *
from configs.faster.default_configs import config, update_config, update_config_from_list
from data_utils.load_data import load_proposal_roidb
import mxnet as mx
import argparse
from train_utils.utils import create_logger, load_param
from inference import imdb_detection_wrapper
from inference import imdb_proposal_extraction_wrapper
import os

def parser():
    arg_parser = argparse.ArgumentParser('SNIPER test module')
    arg_parser.add_argument('--cfg', dest='cfg', help='Path to the config file',
    							default='configs/faster/sniper_res101_e2e_cloth_rescale.yml',type=str)
    arg_parser.add_argument('--save_prefix', dest='save_prefix', help='Prefix used for snapshotting the network',
                            default='SNIPER', type=str)
    arg_parser.add_argument('--vis', dest='vis', help='Whether to visualize the detections',
                            action='store_true')
    arg_parser.add_argument('--set', dest='set_cfg_list', help='Set the configuration fields from command line',
                            default=None, nargs=argparse.REMAINDER)
    return arg_parser.parse_args()



def main():
    args = parser()
    update_config(args.cfg)
    if args.set_cfg_list:
        update_config_from_list(args.set_cfg_list)

    context = [mx.gpu(int(gpu)) for gpu in config.gpus.split(',')]

    if not os.path.isdir(config.output_path):
        os.mkdir(config.output_path)

    if os.path.exists(config.dataset.root_path):
        shutil.rmtree(config.dataset.root_path)
        time.sleep(3)
    os.makedirs(os.path.join(config.dataset.root_path, 'cache'))

    # Create roidb
    roidb, imdb = load_proposal_roidb(config.dataset.dataset, config.dataset.test_image_set, config.dataset.root_path,
                                      config.dataset.dataset_path,
                                      proposal=config.dataset.proposal, only_gt=True, flip=False,
                                      result_path=config.output_path,
                                      proposal_path=config.proposal_path, get_imdb=True)

    # Creating the Logger
    logger, output_path = create_logger(config.output_path, args.cfg, config.dataset.image_set)
    print('output_path:', output_path)
    model_prefix = os.path.join(output_path, args.save_prefix)
    result_txt_path = "./output/sniper_res101_bn_rescale/results"
    if os.path.exists(result_txt_path):
        shutil.rmtree(result_txt_path)
        time.sleep(3)
    os.makedirs(result_txt_path)

    class_list = ["podong", "shajie", 'baijiao', 'chousha', 'louyin', 'cusha']
    class_lenth = len(class_list)
    old_file = []
    for classname in class_list:
        filename = "comp4_det_test_" + classname + ".txt"
        old_file.append(os.path.join(imdb.res_file_folder_ll, filename))
    
    result_file_name = os.path.join(result_txt_path, 'results.txt')
    if os.path.exists(result_file_name):
        os.remove(result_file_name)
    
    imdb.result_file_name = result_file_name 
    for i in range(33, 34):
        f = open(imdb.result_file_name, 'a+')
        f.write('model number {0} map result:\n'.format(i))
        f.close()
        arg_params, aux_params = load_param(model_prefix, i, convert=True, process=True)

        sym_inst = eval('{}.{}'.format(config.symbol, config.symbol))
        if config.TEST.EXTRACT_PROPOSALS:
            imdb_proposal_extraction_wrapper(sym_inst, config, imdb, roidb, context, arg_params, aux_params, args.vis)
        else:
            imdb_detection_wrapper(sym_inst, config, imdb, roidb, context, arg_params, aux_params, args.vis)
        for j in range(class_lenth):
            (d, n) = os.path.split(old_file[j])
            name = os.path.splitext(n)[0]
            new_file = os.path.join(result_txt_path, name + "_" + str(i) + ".txt")
            print(old_file[j], new_file)
            os.rename(old_file[j], new_file)

if __name__ == '__main__':
    main()



