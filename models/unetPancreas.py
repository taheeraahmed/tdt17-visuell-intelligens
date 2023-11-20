from fastMONAI.vision_all import *

from sklearn.model_selection import train_test_split
from monai.apps import DecathlonDataset
from pandas import DataFrame
import numpy as np
import pyfiglet

from monai.losses import DiceCELoss
from monai.networks.nets import UNet

result = pyfiglet.figlet_format("VI babes", font = "slant"  ) 
print(result) 

data_path = '/cluster/projects/vc/data/mic/open/MSD'

print('Loading data..')
training_data = DecathlonDataset(root_dir=data_path, task="Task07_Pancreas", section="training", download=False, cache_num=0, num_workers=3)
print('Done loading data!')

df = DataFrame(training_data.data)
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
print('Analysing training data and test data')
print('Shape of train df')
print(train_df.shape)
print('Shape of test df')
print(test_df.shape)
med_dataset = MedDataset(img_list=train_df.label.tolist(), dtype=MedMask, max_workers=12)
print('Training data')
med_dataset.df.head()

codes = np.unique(med_img_reader(train_df.label.tolist()[0]))
n_classes = len(codes)

print('MedData stuff.. I think this is cacheing it and data augmentation')
bs=4 # batch size
size=[224,224,128] # make every image this size
med_dataset = MedDataset(img_list=train_df.label.tolist(), dtype=MedMask, max_workers=12)
resample, reorder = med_dataset.suggestion() # this returns Voxel value that appears most often in dim_0, dim_1 and dim_2, and whether the data should be reoriented
print("Voxel value for dim_0, dim_1 and dim_2: ", resample, "Shoud reorient: ", reorder)
item_tfms = [ZNormalization(), PadOrCrop(size), RandomAffine(scales=0, degrees=5, isotropic=True)] #Data augmentation
dblock = MedDataBlock(blocks=(ImageBlock(cls=MedImage), MedMaskBlock), splitter=RandomSplitter(seed=42), get_x=ColReader('image'), get_y=ColReader('label'), item_tfms=item_tfms,reorder=reorder,resample=resample)
dls = dblock.dataloaders(train_df, bs=bs)
print('len traing: ',len(dls.train_ds.items), 'len val: ', len(dls.valid_ds.items))
dls.show_batch(anatomical_plane=0)
print('Done with MedData stuff..')

model = UNet(spatial_dims=3, in_channels=1, out_channels=n_classes, channels=(16, 32, 64, 128, 256),strides=(2, 2, 2, 2), num_res_units=2)
loss_func = CustomLoss(loss_func=DiceCELoss(to_onehot_y=True, include_background=True, softmax=True))

learn = Learner(dls, model, loss_func=loss_func, opt_func=ranger, metrics=multi_dice_score)#.to_fp16()
lr = learn.lr_find() #set learning rate?
epochs=20
print('Now trainign!!')
learn.fit_flat_cos(epochs, lr)
learn.save('trained.Pancreas-model')

print('Results from training')
learn.show_results(anatomical_plane=0, ds_idx=1)

print('Testing')
learn.load('trained.Pancreas-model')
test_dl = learn.dls.test_dl(test_df[:10],with_labels=True)
test_dl.show_batch(anatomical_plane=0, figsize=(10,10))


print('result from test')
pred_acts, labels = learn.get_preds(dl=test_dl)
print('predicted shape: ', pred_acts.shape, 'Label shape', labels.shape)
print('Dice score for label_1 and label 2: ',multi_dice_score(pred_acts, labels))
learn.show_results(anatomical_plane=0, dl=test_dl)

print('Export')
learn.export('pancreas_model.pkl')

