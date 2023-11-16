from fastMONAI.vision_all import *

from sklearn.model_selection import train_test_split
from monai.apps import DecathlonDataset
from pandas import DataFrame
import numpy as np

from monai.losses import DiceCELoss
from monai.networks.nets import UNet


def unet_spleen(logger):

    logger.info('Running UNET spleen')
    data_path = '/cluster/projects/vc/data/mic/open/MSD'

    logger.info('Loading data..')
    training_data = DecathlonDataset(root_dir=data_path, task="Task03_Liver", section="training", download=False, cache_num=0, num_workers=3)
    logger.info('Done loading data!')

    df = DataFrame(training_data.data)
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

    codes = np.unique(med_img_reader(train_df.label.tolist()[0]))
    n_classes = len(codes)

    logger.info('MedData stuff..')
    bs=4
    size=[224,224,128]
    med_dataset = MedDataset(img_list=train_df.label.tolist(), dtype=MedMask, max_workers=12)
    resample, reorder = med_dataset.suggestion()
    item_tfms = [ZNormalization(), PadOrCrop(size), RandomAffine(scales=0, degrees=5, isotropic=True)]
    dblock = MedDataBlock(blocks=(ImageBlock(cls=MedImage), MedMaskBlock), splitter=RandomSplitter(seed=42), get_x=ColReader('image'), get_y=ColReader('label'), item_tfms=item_tfms,reorder=reorder,resample=resample)
    dls = dblock.dataloaders(train_df, bs=bs)
    logger.info('Done with MedData stuff..')

    model = UNet(spatial_dims=3, in_channels=1, out_channels=n_classes, channels=(16, 32, 64, 128, 256),strides=(2, 2, 2, 2), num_res_units=2)
    loss_func = CustomLoss(loss_func=DiceCELoss(to_onehot_y=True, include_background=True, softmax=True))

    learn = Learner(dls, model, loss_func=loss_func, opt_func=ranger, metrics=multi_dice_score)#.to_fp16()
    learn.lr_find()




