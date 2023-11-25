from fastMONAI.vision_all import med_img_reader, MedDataset, MedMask, MedMaskBlock, MedImage, RandomSplitter, ColReader, ImageBlock, MedDataBlock, CustomLoss, multi_dice_score, ranger, Learner, store_variables
from helpers.create_dir import create_directory_if_not_exists
from helpers.get_transforms import get_transforms
from sklearn.model_selection import train_test_split
from monai.apps import DecathlonDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from monai.losses import DiceCELoss
from monai.networks.nets import (UNet, UNETR)

def liver_segmentation(logger, model_arg, user, unique_id=0, augmentation="baseline"):
  bs=2 # batch size
  size=[224,224,288]
  epochs = 1
  logger.info(f'batch size: {bs}, size: {size}, epochs: {epochs}')
  path = f'/cluster/home/{user}/runs/output/{unique_id}'
  create_directory_if_not_exists(path)
  task = 'Task03_Liver'

  logger.info(f'Augmentation {augmentation}')
  logger.info('Loading data..')
  data_path = '/cluster/projects/vc/data/mic/open/MSD'                               #IDUN
  training_data = DecathlonDataset(root_dir=data_path, task=task, section="training", download=False, cache_num=0, num_workers=3)

  task = task.lower()
  df = pd.DataFrame(training_data.data)
  train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
  codes = np.unique(med_img_reader(train_df.label.tolist()[0]))
  n_classes = len(codes)

  pd.set_option('display.max_columns', None)
  logger.info('MedData stuff..')
  med_dataset = MedDataset(img_list=train_df.label.tolist(), dtype=MedMask, max_workers=12)
  logger.info(f"{med_dataset.df.iloc[0,:]}")
  summary_df = med_dataset.summary()
  logger.info(f"{summary_df.head()}")

  resample, reorder = med_dataset.suggestion()

  item_tfms = get_transforms(logger, augmentation=augmentation, size=size)

  dblock = MedDataBlock(blocks=(ImageBlock(cls=MedImage), MedMaskBlock), splitter=RandomSplitter(seed=42), get_x=ColReader('image'), get_y=ColReader('label'), item_tfms=item_tfms,reorder=reorder,resample=resample)
  dls = dblock.dataloaders(train_df, bs=bs)
  logger.info(f'len traing:  {len(dls.train_ds.items)}   len val: {len(dls.valid_ds.items)}')
  dls.show_batch(anatomical_plane=0)
  plt.savefig(f'{path}/{task}-batch.png')
  plt.close()
  logger.info('Done with MedData stuff..')

  if model_arg == 'unetr_liver':
    model = UNETR(spatial_dims=3, in_channels=1, out_channels=n_classes, img_size=size)
  elif model_arg=="unet_liver":
    model = UNet(spatial_dims=3, in_channels=1, out_channels=n_classes, channels=(16, 32, 64, 128, 256),strides=(2, 2, 2, 2), num_res_units=2)

  loss_func = CustomLoss(loss_func=DiceCELoss(to_onehot_y=True, include_background=True, softmax=True))

  learn = Learner(dls, model, loss_func=loss_func, opt_func=ranger, metrics=multi_dice_score)#.to_fp16()
  lr = learn.lr_find()
  plt.savefig(f'{path}/{task}-lr-find.png')
  plt.close()
  logger.info("Done with find")

  learn.fit_flat_cos(epochs ,lr)
  learn.save(f'{path}/{task}-liver-model')
  learn.show_results(anatomical_plane=0, ds_idx=1)
  plt.savefig(f"{path}/{task}-fit.png")
  plt.close()
  logger.info("Training stuff done, now inference")

  learn.load(f'{path}/{task}-liver-model')
  test_dl = learn.dls.test_dl(test_df[:10],with_labels=True)
  test_dl.show_batch(anatomical_plane=0, figsize=(10,10))

  pred_acts, labels = learn.get_preds(dl=test_dl)
  logger.info(f'Predicted shape: {pred_acts.shape}, Label shape: {labels.shape}')
  dice_score = multi_dice_score(pred_acts, labels)
  logger.info(f'Dice score: {dice_score}')
  learn.show_results(anatomical_plane=0, dl=test_dl)
  plt.savefig(f"{path}/{task}-results.png")
  plt.close()

  logger.info("Done")