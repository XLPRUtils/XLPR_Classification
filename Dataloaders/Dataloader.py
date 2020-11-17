# -*- coding:utf-8 -*-
from .GestureDataSet import GestureDataSet

def get_data_set(type, split):
    if type == 'gesture':
        return GestureDataSet(split=split)