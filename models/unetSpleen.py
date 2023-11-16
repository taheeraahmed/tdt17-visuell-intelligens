from fastMONAI.vision_all import med_img_reader, MedDataset, MedMask, PadOrCrop, RandomAffine, MedMaskBlock, MedImage, RandomSplitter, ColReader, ImageBlock, ZNormalization, MedDataBlock, CustomLoss, multi_dice_score, ranger, Learner, store_variables
from sklearn.model_selection import train_test_split
from monai.apps import DecathlonDataset
from pandas import DataFrame
import numpy as np

from monai.losses import DiceCELoss
from monai.networks.nets import UNet

def unet_spleen(logger):
    task = 'Task09_Spleen'
    logger.info('Running UNET spleen')
    data_path = '/cluster/projects/vc/data/mic/open/MSD'

    logger.info('Loading data..')
    training_data = DecathlonDataset(root_dir=data_path, task=task, section="training", download=False, cache_num=0, num_workers=3)
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

    lr = 1e-1

    logger.info('Learn-fit-flat')
    learn.fit_flat_cos(20 ,lr)

    learn.save('checkpoints/task09')
    learn.show_results(anatomical_plane=0, ds_idx=1)
    logger.info('Saved checkpoints to: checkpoints/task09')
    learn.load('checkpoints/task09');
    test_dl = learn.dls.test_dl(test_df[:10],with_labels=True)
    test_dl.show_batch(anatomical_plane=0, figsize=(10,10))

    logger.info('Predicting')
    pred_acts, labels = learn.get_preds(dl=test_dl)
    pred_acts.shape, labels.shape
    multi_dice_score(pred_acts, labels)
    learn.show_results(anatomical_plane=0, dl=test_dl)
    store_variables(pkl_fn='vars.pkl', size=size, reorder=reorder,  resample=resample)
    learn.export('checkpoints/task09/model.pkl')
    logger.info('Exported pickle file: checkpoints/task09/model.pkl')

