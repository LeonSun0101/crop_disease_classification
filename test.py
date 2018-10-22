# coding=utf-8
import argparse
import textwrap
import time
import os, sys
from torch.autograd import Variable

sys.path.append(os.path.dirname(__file__))
from utils.config import process_config, check_config_dict
from utils.logger import ExampleLogger
from trainers.example_model import ExampleModel
from trainers.example_trainer import ExampleTrainer
from data_loader.dataset import get_data_loader
import sys
import utils.global_variable as global_value
import torch
import json
from nets.net_interface import NetModule

config = process_config(os.path.join(os.path.dirname(__file__), 'configs', 'config.json'))

class ImageClassificationPytorch:
    def __init__(self, config):
        gpu_id = config['gpu_id']
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        print(config)
        #check_config_dict(config)
        self.config = config
        self.init()


    def init(self):
        # create net
        self.model = ExampleModel(self.config)
        # load
        self.model.load()
        # create your data generator
        self.train_loader, self.test_loader = get_data_loader(self.config)
        # create logger
        self.logger = ExampleLogger(self.config)
        # create trainer and path all previous components to it
        self.trainer = ExampleTrainer(self.model, self.train_loader, self.test_loader, self.config, self.logger)


    def run(self):
        # here you train your model
        self.trainer.train()


    def close(self):
        # close
        self.logger.close()


def main():
    global_value._init()
    reload(sys)
    sys.setdefaultencoding("utf-8")
    state_dict=torch.load('/home1/sas/crop_disease_detect-master/val_eval_best.pth')
    interface = NetModule(config['model_module_name'], config['model_net_name'])
    net = interface.create_model(num_classes=config['num_classes'])
    if torch.cuda.is_available():
        net.cuda()
    net.load_state_dict(state_dict)
    net.eval()
    #print(net)
    train_loader, test_loader = get_data_loader(config)
    num=0
    f=open("/home1/sas/datasets/ai_challenger_pdr2018_validationset_20180905/AgriculturalDisease_validationset/AgriculturalDisease_validation_annotations.json",'r+')
    f_results=open("/home1/sas/datasets/ai_challenger_pdr2018_testA_20180905/AgriculturalDisease_testA/AgriculturalDisease_val_results_val_best_annotations.json",'w')
    json_data=json.load(f)
    print(len(json_data))
    for _, (image, label) in enumerate(test_loader):
        image=image.cuda()
        image=Variable(image)
        labels=net(image)
        label=torch.max(labels,1)[1].data.squeeze()
        label=int(label)
        json_data[_]['disease_class']=label
        #print(json_data[_])
        num=num+1
    dict_json=json.dumps(json_data, ensure_ascii=True, indent=2)
    f_results.write(dict_json)
    f_results.close()
    f.close()

if __name__ == '__main__':

    now = time.strftime('%Y-%m-%d | %H:%M:%S', time.localtime(time.time()))

    print('----------------------------------------------------------------------')
    print('Time: ' + now)
    print('----------------------------------------------------------------------')
    print('                    Now start ...')
    print('----------------------------------------------------------------------')

    main()

    print('----------------------------------------------------------------------')
    print('                      All Done!')
    print('----------------------------------------------------------------------')
    print('Start time: ' + now)
    print('Now time: ' + time.strftime('%Y-%m-%d | %H:%M:%S', time.localtime(time.time())))
    print('----------------------------------------------------------------------')