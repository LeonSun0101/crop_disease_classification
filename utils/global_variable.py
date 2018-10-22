# -*- coding: utf-8 -*-

def _init():#初始化
    global _train_eval_list, _val_eval_list
    _train_eval_list = []
    _val_eval_list = []


def _set_train_eval_value(value):
    """ 定义一个全局变量 """
    _train_eval_list.append(value)
    print("_train_eval_list",_train_eval_list)

def _set_val_eval_value(value):
    """ 定义一个全局变量 """
    _val_eval_list.append(value)
    print("_val_eval_list",_val_eval_list)


def _get_train_max():
    return max(_train_eval_list)


def _get_val_max():
    return max(_val_eval_list)

def _get_train_eval_value(epoch):
    return _train_eval_list[epoch]

def _get_val_eval_value(epoch):
    return _val_eval_list[epoch]

