import matplotlib.pyplot as plt
from fastMONAI.vision_all import med_img_reader, MedDataset, MedMask, PadOrCrop, RandomAffine, RandomNoise, RandomGamma, MedMaskBlock, MedImage, RandomSplitter, ColReader, ImageBlock, ZNormalization, MedDataBlock, CustomLoss, multi_dice_score, ranger, Learner, store_variables
from sklearn.model_selection import train_test_split
from monai.apps import DecathlonDataset
from pandas import DataFrame
import numpy as np
from helpers.create_dir import create_directory_if_not_exists
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
import sys

def unet_spleen(logger,  model_arg, unique_id=0, augmentation=None):
    path = f'/cluster/home/taheeraa/runs/output/{unique_id}'
    create_directory_if_not_exists(path)
    task = 'Task09_Spleen'

    logger.info(f'Augmentation {augmentation}')
    logger.info('Loading data..')
    data_path = '/cluster/projects/vc/data/mic/open/MSD'                               #IDUN
    #data_path = 'C:\\Users\\Taheera Ahmed\\code\\tdt17-visuell-intelligens\\dataset'    #Locally
    training_data = DecathlonDataset(root_dir=data_path, task=task, section="training", download=False, cache_num=0, num_workers=3)
    logger.info('Done loading data!')

    df = DataFrame(training_data.data)
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    codes = np.unique(med_img_reader(train_df.label.tolist()[0]))
    n_classes = len(codes)

    logger.info('MedData stuff..')
    bs=2
    size=[512,512,128]
    med_dataset = MedDataset(img_list=train_df.label.tolist(), dtype=MedMask, max_workers=12)
    resample, reorder = med_dataset.suggestion()

    logger.info('Adding augmentations')
    random_affine = RandomAffine(degrees=15, translation=15)
    random_gamma = RandomGamma()
    random_noise = RandomNoise()
    item_tfms = [ZNormalization(), PadOrCrop(size)]

    if augmentation == 'rand_affine':
        item_tfms.append(random_affine)
    elif augmentation == 'rand_gamma':
        item_tfms.append(random_gamma)
    elif augmentation == 'rand_noise':
        item_tfms.append(random_noise)
    else:
        pass
    logger.info(f'{item_tfms}')

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
    plt.savefig(path+'/task09-dls.png') 
    logger.info(f'Figure has been stored at path: {path}')
    logger.info('Done with MedData stuff..')

    if (model_arg == 'unetr_spleen'):
        model = UNETR(spatial_dims=3, in_channels=1, out_channels=n_classes, img_size=size)
    elif model=="unet_spleen":
        model = UNet(spatial_dims=3, in_channels=1, out_channels=n_classes, channels=(16, 32, 64, 128, 256),strides=(2, 2, 2, 2), num_res_units=2)
    
    loss_func = CustomLoss(loss_func=DiceCELoss(to_onehot_y=True, include_background=True, softmax=True))

    logger.info('Running learner and lr_find')
    learn = Learner(dls, model, loss_func=loss_func, opt_func=ranger, metrics=multi_dice_score)#.to_fp16()
    lr = learn.lr_find()
    plt.savefig(path+'/task09-spleen-lr-find.png')
    logger.info(f'Figure has been stored at path: {path}/task09-spleen-lr-find.png')

    logger.info('Learn-fit-flat')
    learn.fit_flat_cos(50 ,lr)

    learn.save('spleen-model')
    logger.info(f'Model has been stored at path: {path}/spleen-model.pth')
    learn.show_results(anatomical_plane=0, ds_idx=1)
    plt.savefig(path +'/task09-show-results.png')  # Replace with your desired file path and name
    logger.info(f'Figure has been stored at path: {path}/task09-show-results.png')


    learn.load('spleen-model');
    logger.info(f'Model has been loaded at path: {path}/spleen-model.pth')
    test_dl = learn.dls.test_dl(test_df[:10],with_labels=True)
    test_dl.show_batch(anatomical_plane=0, figsize=(10,10))
    plt.savefig(path + '/task09-show-batch.png')
    logger.info(f'Figure has been stored at path: {path}/task09-show-batch.png')

    logger.info('Predicting')
    pred_acts, labels = learn.get_preds(dl=test_dl)
    pred_acts.shape, labels.shape
    multi_dice_score(pred_acts, labels)
    learn.show_results(anatomical_plane=0, dl=test_dl)
    plt.savefig(path + '/task09-show-results.png')  # Replace with your desired file path and name
    logger.info(f'Figure has been stored at path: {path}/task09-show-results.png')

    store_variables(pkl_fn='vars.pkl', size=size, reorder=reorder,  resample=resample)
    learn.export(path + '/model.pkl')

    logger.info(f'Exported pickle file: {path}/model.pkl')