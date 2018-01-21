import numpy as np


class GenerateBatch:
    def __init__(self, dataset, label, batch_size, epoch_num=None):
        self.dataset = dataset
        self.label = label
        self.batch_size = batch_size
        self.start = 0
        self.epoch_num = epoch_num

    def generate_next_batch(self):
        if self.epoch_num is not None:
            for i in range(self.epoch_num):
                while self.start < len(self.dataset):
                    cur_image_batch = self.dataset[self.start: self.start + self.batch_size]
                    cur_label_batch = self.label[self.start: self.start + self.batch_size]
                    self.start += self.batch_size
                    cur_image_batch = np.asarray(cur_image_batch, np.float32) / 255.0
                    yield cur_image_batch, cur_label_batch
        else:
            while True:
                cur_image_batch = self.dataset[self.start: self.start + self.batch_size]
                cur_label_batch = self.label[self.start: self.start + self.batch_size]
                self.start = (self.start + self.batch_size) % len(self.dataset)
                cur_image_batch = np.asarray(cur_image_batch, np.float32) / 255.0
                yield cur_image_batch, cur_label_batch