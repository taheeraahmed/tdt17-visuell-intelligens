from fastMONAI.vision_all import *

from sklearn.model_selection import train_test_split
from monai.apps import DecathlonDataset
from pandas import DataFrame
import numpy as np
import pyfiglet

from monai.losses import DiceCELoss
from monai.networks.nets import UNet
pd.set_option('display.max_columns', None)

result = pyfiglet.figlet_format("VI babes", font = "slant"  ) 
print(result) 

data_path = '/cluster/projects/vc/data/mic/open/MSD'

print('Loading data..')
training_data = DecathlonDataset(root_dir=data_path, task="Task07_Pancreas", section="training", download=False, cache_num=0, num_workers=3)
print('Done loading data!')

df = DataFrame(training_data.data)
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
print('Analysing training data and test data')
print('Shape of train df: ',train_df.shape)
print('Shape of test df: ', test_df.shape)

med_dataset = MedDataset(img_list=train_df.label.tolist(), dtype=MedMask, max_workers=12)
print('Information of one training image')
print(med_dataset.df.head())

codes = np.unique(med_img_reader(train_df.label.tolist()[0]))
n_classes = len(codes)

print('Cacheing it and data augmentation')
bs=4 # batch size
size=[225,225,68] # make every image this size
med_dataset = MedDataset(img_list=train_df.label.tolist(), dtype=MedMask, max_workers=12)
summary_df = med_dataset.summary()
print(summary_df.head())
resample, reorder = med_dataset.suggestion() # this returns Voxel value that appears most often in dim_0, dim_1 and dim_2, and whether the data should be reoriented
print("Voxel value for dim_0, dim_1 and dim_2: ", resample, "Shoud reorient: ", reorder)
item_tfms = [PadOrCrop(size)] #Data augmentation
# item_tfms = [ZNormalization(), PadOrCrop(size), RandomAffine(scales=0, degrees=5, isotropic=True)] #Data augmentation
dblock = MedDataBlock(blocks=(ImageBlock(cls=MedImage), MedMaskBlock), splitter=RandomSplitter(seed=42), get_x=ColReader('image'), get_y=ColReader('label'),reorder=reorder,resample=resample, item_tfms=item_tfms) # item_tfms=item_tfms
dls = dblock.dataloaders(train_df, bs=bs)
print('Length of traning data: ',len(dls.train_ds.items), 'Length of validation data: ', len(dls.valid_ds.items))
dls.show_batch(anatomical_plane=0)
plt.title('Image/label')
plt.savefig('Pancreas/v1/images/Examples_of_data_pancreas.png')
plt.close()
print('Done with cache and data augmentation..')

model = UNet(spatial_dims=3, in_channels=1, out_channels=n_classes, channels=(16, 32, 64, 128, 256),strides=(2, 2, 2, 2), num_res_units=2)
loss_func = CustomLoss(loss_func=DiceCELoss(to_onehot_y=True, include_background=True, softmax=True))

print('Learning rate')
learn = Learner(dls, model, loss_func=loss_func, opt_func=ranger, metrics=multi_dice_score)#.to_fp16()
lr = learn.lr_find() #set learning rate?
print('learning rate', lr)
plt.savefig('Pancreas/v1/images/learning.png')
plt.close()
epochs=40
print('Training')
learn.fit_flat_cos(epochs, lr)
learn.save('Pancreas/v1/models/trained.Pancreas-model')

print('Results from training')
learn.show_results(anatomical_plane=0, ds_idx=1)
plt.savefig('Pancreas/v1/images/result_training_pancreas.png')
plt.close()

print('Testing')
learn.load('Pancreas/v1/models/trained.Pancreas-model')
test_dl = learn.dls.test_dl(test_df[:10],with_labels=True)
test_dl.show_batch(anatomical_plane=0, figsize=(10,10))
plt.savefig('Pancreas/v1/images/result_testing_pancreas.png')
plt.close()


print('result from test')
pred_acts, labels = learn.get_preds(dl=test_dl)
print('predicted shape: ', pred_acts.shape, 'Label shape', labels.shape)
print('Dice score for label_1 and label 2: ',multi_dice_score(pred_acts, labels))
learn.show_results(anatomical_plane=0, dl=test_dl)
plt.savefig('Pancreas/v1/images/result_training.png')
plt.close()

print('Export')
learn.export('Pancreas/v1/models/pancreas_model.pkl')

