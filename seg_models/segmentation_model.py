import matplotlib.pyplot as plt
from fastMONAI.vision_all import med_img_reader, MedDataset, MedMask, MedMaskBlock, MedImage, RandomSplitter, ColReader, ImageBlock, MedDataBlock, CustomLoss, multi_dice_score, ranger, Learner, store_variables
from sklearn.model_selection import train_test_split
from monai.apps import DecathlonDataset
from pandas import DataFrame
import numpy as np
from helpers.create_dir import create_directory_if_not_exists
from helpers.get_transforms import get_transforms
from monai.losses import DiceCELoss
from monai.networks.nets import (
    UNet,
    UNETR
)
import sys

def segmentation_model(logger, model_arg, user, task, unique_id=0, augmentation="baseline"):
    bs = 1
    size=[512, 512, 128]
    epochs = 100
    logger.info(f'batch size: {bs}, size: {size}, epochs: {epochs}')
    path = f'/cluster/home/{user}/runs/output/{unique_id}'
    create_directory_if_not_exists(path)

    logger.info(f'Augmentation {augmentation}')
    logger.info('Loading data..')
    data_path = '/cluster/projects/vc/data/mic/open/MSD'
    training_data = DecathlonDataset(root_dir=data_path, task=task, section="training", download=False, cache_num=0, num_workers=3)

    df = DataFrame(training_data.data)
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    codes = np.unique(med_img_reader(train_df.label.tolist()[0]))
    n_classes = len(codes)

    if model_arg == 'unetr':
        model = UNETR(spatial_dims=3, in_channels=1, out_channels=n_classes, img_size=size)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total number of parameters in UNETR: {total_params}")
        
    elif model_arg=="unet":
        model = UNet(spatial_dims=3, in_channels=1, out_channels=n_classes, channels=(16, 32, 64, 128, 256),strides=(2, 2, 2, 2), num_res_units=2)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total number of parameters in U-Net: {total_params}")
    else:
        logger.error('Invalid model')
        sys.exit(1)

    med_dataset = MedDataset(img_list=train_df.label.tolist(), dtype=MedMask, max_workers=12)
    resample, reorder = med_dataset.suggestion()
    item_tfms = get_transforms(logger, augmentation=augmentation, size=size)
    logger.info(f'Added these augmentations: {item_tfms}')

    dblock = MedDataBlock(
        blocks=(ImageBlock(cls=MedImage), MedMaskBlock), 
        splitter=RandomSplitter(seed=42), 
        get_x=ColReader('image'), 
        get_y=ColReader('label'), 
        item_tfms=item_tfms,
        reorder=reorder,
        resample=resample
    )
    dls = dblock.dataloaders(train_df, bs=bs)
    dls.show_batch(anatomical_plane=0)
    plt.savefig(f'{path}/{task}-dls.png') 
    logger.info(f'Show batches figure has been stored at path: {path}/task09-dls.png')
    logger.info('Data is loaded..')

    loss_func = CustomLoss(loss_func=DiceCELoss(to_onehot_y=True, include_background=True, softmax=True))

    logger.info('Running learner and lr_find')
    learn = Learner(dls, model, loss_func=loss_func, opt_func=ranger, metrics=multi_dice_score)#.to_fp16()
    lr = learn.lr_find()
    plt.savefig(f'{path}/{task}-lr-find.png')
    logger.info(f'Learning rate figure has been stored at path: {path}/{task}-lr-find.png')
    
    logger.info('Learn-fit-flat')
    learn.fit_flat_cos(epochs, lr)
    learn.save(f'{model_arg}')
    
    logger.info(f'Model has been stored at path: {path}/{model_arg}.pth')
    learn.show_results(anatomical_plane=0, ds_idx=1)
    plt.savefig(f'{path}/{task}-show-results.png')  # Replace with your desired file path and name
    logger.info(f'Results figure has been stored at path: {path}/{task}-show-results.png')


    learn.load(f'{model_arg}');
    logger.info(f'Model has been loaded at path: {path}/{model_arg}.pth')
    test_dl = learn.dls.test_dl(test_df[:10],with_labels=True)
    test_dl.show_batch(anatomical_plane=0, figsize=(10,10))
    plt.savefig(f'{path}/{task}-show-batch.png')
    logger.info(f'Test batch figures has been stored at path: {path}/{task}-show-batch.png')

    logger.info('Predicting')
    pred_acts, labels = learn.get_preds(dl=test_dl)
    pred_acts.shape, labels.shape
    logger.info(f'{multi_dice_score(pred_acts, labels)}')
    learn.show_results(anatomical_plane=0, dl=test_dl)
    plt.savefig(f'{path}/{task}-test-results.png')  # Replace with your desired file path and name
    logger.info(f'Test results figure has been stored at path: {path}/{task}-test-results.png')

    store_variables(pkl_fn=f'{path}/vars.pkl', size=size, reorder=reorder, resample=resample)
    logger.info(f'Exported vars file: {path}/vars.pkl')
    learn.export(f'{path}/model.pkl')
    logger.info(f'Exported model: {path}/model.pkl')
