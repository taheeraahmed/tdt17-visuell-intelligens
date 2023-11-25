from fastMONAI.vision_all import *

from sklearn.model_selection import train_test_split
from monai.apps import DecathlonDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from monai.losses import DiceCELoss
from monai.networks.nets import (UNet, UNETR)

def unet_liver(logger, version):
  v = "04" if version == "no_augmentaion" else "05" if version == "RandomAffine" else "06" if version == "RandomGamma" else "07" if version == "RandomNoise" else "UNETR"
  logger.info(f'Running version {v}')

  data_path = '/cluster/projects/vc/data/mic/open/MSD'

  logger.info('Loading data..')
  training_data = DecathlonDataset(root_dir=data_path, task="Task03_Liver", section="training", cache_num=0, num_workers=3)

  logger.info('Done loading data!')

  df = pd.DataFrame(training_data.data)
  train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

  codes = np.unique(med_img_reader(train_df.label.tolist()[0]))
  n_classes = len(codes)

  pd.set_option('display.max_columns', None)

  logger.info('MedData stuff..')
  bs=2
  size=[224,224,288] #change to something that fits the liver images (changed from 224,224,128 -> 256,256,320)
  # The numbers must be a 64-step, or maybe 32?
  med_dataset = MedDataset(img_list=train_df.label.tolist(), dtype=MedMask, max_workers=12)
  logger.info(med_dataset.df.iloc[0,:])
  summary_df = med_dataset.summary()
  logger.info(summary_df.head())

  resample, reorder = med_dataset.suggestion()

  item_tfms = []
  if (version == 'no_augmentaion'):
    item_tfms = [ZNormalization(), PadOrCrop(size)] #Data augmentation
  elif (version=='UNETR'):
    item_tfms = [ZNormalization(), PadOrCrop(size)] #Data augmentation
  elif (version == 'RandomAffine'):
    item_tfms = [ZNormalization(), PadOrCrop(size), RandomAffine(degrees=15, translation=15) ] #Data augmentation
  elif (version == 'RandomGamma'):
    item_tfms = [ZNormalization(), PadOrCrop(size), RandomGamma()] #Data augmentation
  elif (version == 'RandomNoise'):
    item_tfms = [ZNormalization(), PadOrCrop(size), RandomNoise()] #Data augmentation

  #item_tfms = [ZNormalization(), PadOrCrop(size)] #, RandomAffine(degrees=15, translation=15) , RandomGamma() , RandomNoise()
  dblock = MedDataBlock(blocks=(ImageBlock(cls=MedImage), MedMaskBlock), splitter=RandomSplitter(seed=42), get_x=ColReader('image'), get_y=ColReader('label'), item_tfms=item_tfms,reorder=reorder,resample=resample)
  dls = dblock.dataloaders(train_df, bs=bs)
  logger.info(f'len traing:  {len(dls.train_ds.items)}   len val: {len(dls.valid_ds.items)}')
  dls.show_batch(anatomical_plane=0)
  plt.savefig(f"liver/{v}_batch.png")
  plt.close()
  logger.info('Done with MedData stuff..')

  if (version == 'UNETR'):
    model = UNETR(spatial_dims=3, in_channels=1, out_channels=n_classes, img_size=(224,224,288))
  else:
    model = UNet(spatial_dims=3, in_channels=1, out_channels=n_classes, channels=(16, 32, 64, 128, 256),strides=(2, 2, 2, 2), num_res_units=2)

  #model = UNet(spatial_dims=3, in_channels=1, out_channels=n_classes, channels=(16, 32, 64, 128, 256),strides=(2, 2, 2, 2), num_res_units=2)
  #model = UNETR(spatial_dims=3, in_channels=1, out_channels=n_classes, img_size=(224,224,288))

  loss_func = CustomLoss(loss_func=DiceCELoss(to_onehot_y=True, include_background=True, softmax=True))

  learn = Learner(dls, model, loss_func=loss_func, opt_func=ranger, metrics=multi_dice_score)#.to_fp16()
  lr = learn.lr_find()
  plt.savefig(f"liver/{v}_lr_find.png")
  plt.close()
  logger.info("done with find")

  epochs = 20
  learn.fit_flat_cos(epochs ,lr)
  learn.save('liver-model')
  learn.show_results(anatomical_plane=0, ds_idx=1)
  plt.savefig(f"liver/{v}_fit.png")
  plt.close()
  logger.info("training stuff done, now inference")

  learn.load('liver-model')
  test_dl = learn.dls.test_dl(test_df[:10],with_labels=True)
  test_dl.show_batch(anatomical_plane=0, figsize=(10,10))

  pred_acts, labels = learn.get_preds(dl=test_dl)
  print('predicted shape: ', pred_acts.shape, 'Label shape', labels.shape)
  dice_score = multi_dice_score(pred_acts, labels)
  print('Dice score: ',multi_dice_score(pred_acts, labels))
  logger.info(f'Dice score: {dice_score}')
  learn.show_results(anatomical_plane=0, dl=test_dl)
  plt.savefig(f"liver/{v}_results.png")
  plt.close()

  logger.info("done")