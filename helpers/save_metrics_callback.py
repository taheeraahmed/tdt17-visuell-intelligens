from fastai.callback.core import Callback
import csv
import os

class SaveMetricsCallback(Callback):
    def __init__(self, unique_id, model_arg, augmentation, path, logger):
        self.unique_id = unique_id
        self.model_arg = model_arg
        self.augmentaion = augmentation
        self.path = path
        self.logger = logger
        self.fname = f"{path}/metrics.csv"
        # Open the file in write mode to create it and write headers
        with open(self.fname, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['epoch', 'train_loss', 'valid_loss', 'multi_dice_score', 'time'])

        self.logger.info(f'Logging metrics in path: {self.fname}')

    def after_epoch(self):
        # Append metrics to the CSV file
        with open(self.fname, mode='a', newline='') as file:
            writer = csv.writer(file)
            metrics = [self.epoch, self.learn.recorder.values[-1]]
            writer.writerow(metrics)
        self.logger.info(f'Logged metrics to: {self.fname}')