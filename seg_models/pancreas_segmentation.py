from fastMONAI.vision_all import med_img_reader, MedDataset, MedMask, MedMaskBlock, MedImage, RandomSplitter, ColReader, ImageBlock, MedDataBlock, CustomLoss, multi_dice_score, ranger, Learner, store_variables
from sklearn.model_selection import train_test_split
from monai.apps import DecathlonDataset
from monai.transforms import RandAffined
from pandas import DataFrame
import numpy as np
from helpers.create_dir import create_directory_if_not_exists
from helpers.get_transforms import get_transforms
from monai.losses import DiceCELoss
from monai.networks.nets import (
    UNet,
    UNETR
)
import matplotlib.pyplot as plt

def pancreas_segmentation(logger, model_arg, user, unique_id=0, augmentation="baseline"):
    bs=4 # batch size
    size=[224,224,128] # make every image this size
    epochs = 1
    logger.info(f'batch size: {bs}, size: {size}, epochs: {epochs}')
    path = f'/cluster/home/{user}/runs/output/{unique_id}'
    create_directory_if_not_exists(path)
    task = 'Task07_Pancreas'

    logger.info(f'Augmentation {augmentation}')
    logger.info('Loading data..')
    data_path = '/cluster/projects/vc/data/mic/open/MSD'                               #IDUN
    training_data = DecathlonDataset(root_dir=data_path, task=task, section="training", download=False, cache_num=0, num_workers=3)

    task = task.lower()
    df = DataFrame(training_data.data)
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    codes = np.unique(med_img_reader(train_df.label.tolist()[0]))
    n_classes = len(codes)

    med_dataset = MedDataset(img_list=train_df.label.tolist(), dtype=MedMask, max_workers=12)
    logger.info('Information of one training image')
    logger.info(f"{med_dataset.df.head()}")

    codes = np.unique(med_img_reader(train_df.label.tolist()[0]))
    n_classes = len(codes)

    logger.info('Cacheing it and data augmentation')
    
    med_dataset = MedDataset(img_list=train_df.label.tolist(), dtype=MedMask, max_workers=12)
    summary_df = med_dataset.summary()
    logger.info(f"{summary_df.head()}")
    resample, reorder = med_dataset.suggestion() # this returns Voxel value that appears most often in dim_0, dim_1 and dim_2, and whether the data should be reoriented
    print("Voxel value for dim_0, dim_1 and dim_2: ", resample)
    print("Should reorder data: ", reorder)

    item_tfms = get_transforms(logger, augmentation=augmentation, size=size)

    dblock = MedDataBlock(blocks=(ImageBlock(cls=MedImage), MedMaskBlock), splitter=RandomSplitter(seed=42), get_x=ColReader('image'), get_y=ColReader('label'),reorder=reorder,resample=resample, item_tfms=item_tfms) # item_tfms=item_tfms
    dls = dblock.dataloaders(train_df, bs=bs)
    logger.info(f'Length of traning data: {len(dls.train_ds.items)}, length of validation data:  {len(dls.valid_ds.items)}')
    dls.show_batch(anatomical_plane=0, slice_index=20)
    plt.savefig(f'{path}/{task}-examples_of_data_pancreas_slice.png')
    plt.close()
    dls.show_batch(anatomical_plane=0)
    plt.savefig(f'{path}/{task}-examples_of_data_pancreas.png')
    plt.close()
    dls.show_batch(anatomical_plane=1)
    plt.savefig(f'{path}/{task}-examples_of_data_pancreas_ap1.png')
    plt.close()
    logger.info('Done with cache and data augmentation..')

    if model_arg == 'unetr_pancreas':
        model = UNETR(spatial_dims=3, in_channels=1, out_channels=n_classes, img_size=size)
    elif model_arg=="unet_pancreas":
        model = UNet(spatial_dims=3, in_channels=1, out_channels=n_classes, channels=(16, 32, 64, 128, 256),strides=(2, 2, 2, 2), num_res_units=2)
    loss_func = CustomLoss(loss_func=DiceCELoss(to_onehot_y=True, include_background=True, softmax=True))

    logger.info('Learning rate')
    learn = Learner(dls, model, loss_func=loss_func, opt_func=ranger, metrics=multi_dice_score)#.to_fp16()
    lr = learn.lr_find() #set learning rate?
    logger.info(f'Learning rate: {lr}')
    plt.savefig(f'{path}/{task}-lr-find.png')
    plt.close()

    logger.info('Training')
    learn.fit_flat_cos(epochs, lr)
    learn.save(f'{model_arg}')

    logger.info('Results from training')
    learn.show_results(anatomical_plane=0, ds_idx=1)
    plt.savefig(f'{path}/{task}-result_training_pancreas.png')
    plt.close()

    logger.info('Testing')
    learn.load(f'{path}/{task}-trained-pancreas-model')
    test_dl = learn.dls.test_dl(test_df[:10],with_labels=True)
    test_dl.show_batch(anatomical_plane=0, figsize=(10,10))
    plt.savefig(f'{path}/{task}-show-results.png')
    plt.close()


    logger.info('result from test')
    pred_acts, labels = learn.get_preds(dl=test_dl)
    print('predicted shape: ', pred_acts.shape, 'Label shape', labels.shape)
    print('Dice score for label_1 and label 2: ',multi_dice_score(pred_acts, labels))
    learn.show_results(anatomical_plane=0, dl=test_dl)
    plt.savefig(f'{path}/{task}-result_training.png')
    plt.close()

    logger.info('Export')
    learn.export(f'{path}/{task}-pancreas_model.pkl')
