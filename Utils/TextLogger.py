# -*- coding:utf-8 -*-
import time
import torch

class TextLogger():
    def __init__(self, title, save_path, append=False):
        print(save_path)
        file_state = 'wb'
        if append:
            file_state = 'ab'
        self.file = open(save_path, file_state, 0)
        self.log(title)

    def log(self, strdata):
        outstr = strdata + '\n'
        outstr = outstr.encode("utf-8")
        self.file.write(outstr)

    def __del__(self):
        self.file.close()

if __name__ == '__main__':
    train_logger = TextLogger('Train loss', 'train_loss.log')
    for ix in range(30):
        print(ix)
        train_logger.log('%s, %s' % (str(torch.rand(1)[0]), str(torch.rand(1)[0])))
        time.sleep(1)