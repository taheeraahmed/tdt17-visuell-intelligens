from fastMONAI.vision_all import *

from sklearn.model_selection import train_test_split
from monai.apps import DecathlonDataset
import pandas as pd
import matplotlib.pyplot as plt


data_path = '/cluster/projects/vc/data/mic/open/MSD'

def get_batch_images(task, size):
  training_data = DecathlonDataset(root_dir=data_path, task=task, section="training", cache_num=0, num_workers=3)
  df = pd.DataFrame(training_data.data)
  train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

  bs=4
  med_dataset = MedDataset(img_list=train_df.label.tolist(), dtype=MedMask, max_workers=12)

  resample, reorder = med_dataset.suggestion()

  item_tfms = [PadOrCrop(size)]

  dblock = MedDataBlock(blocks=(ImageBlock(cls=MedImage), MedMaskBlock), splitter=RandomSplitter(seed=42), get_x=ColReader('image'), get_y=ColReader('label'), item_tfms=item_tfms,reorder=reorder,resample=resample)
  dls = dblock.dataloaders(train_df, bs=bs)
  print(f'len trainig:  {len(dls.train_ds.items)}   len val: {len(dls.valid_ds.items)}')
  dls.show_batch(anatomical_plane=0)
  
  plt.savefig(f"batch_images/{task}_batch3.png")
  plt.close()
  print(f"Done with batch images for {task}")

get_batch_images("Task03_Liver", [224,224,288])
get_batch_images("Task07_Pancreas", [224,224,128])
get_batch_images("Task09_Spleen", [512,512,128])