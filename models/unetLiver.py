from fastMONAI.vision_all import *

from sklearn.model_selection import train_test_split
from monai.apps import DecathlonDataset
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt

from monai.losses import DiceCELoss
from monai.networks.nets import UNet

print('Running UNET liver')

data_path = '/cluster/projects/vc/data/mic/open/MSD'

# Below is code with fastMonai 

print('Loading data..')
training_data = DecathlonDataset(root_dir=data_path, task="Task03_Liver", section="training", cache_num=0, num_workers=3)

print('Done loading data!')

df = DataFrame(training_data.data)
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

codes = np.unique(med_img_reader(train_df.label.tolist()[0]))
n_classes = len(codes)

print('MedData stuff..')
bs=4
size=[224,224,128]
med_dataset = MedDataset(img_list=train_df.label.tolist(), dtype=MedMask, max_workers=12)
print(med_dataset.df.iloc[0,:])

resample, reorder = med_dataset.suggestion()
item_tfms = [ZNormalization(), PadOrCrop(size), RandomAffine(scales=0, degrees=5, isotropic=True)]
dblock = MedDataBlock(blocks=(ImageBlock(cls=MedImage), MedMaskBlock), splitter=RandomSplitter(seed=42), get_x=ColReader('image'), get_y=ColReader('label'), item_tfms=item_tfms,reorder=reorder,resample=resample)
dls = dblock.dataloaders(train_df, bs=bs)
print('len traing: ',len(dls.train_ds.items), 'len val: ', len(dls.valid_ds.items))
dls.show_batch(anatomical_plane=0)
plt.savefig("figures/batch_liver.png")
print('Done with MedData stuff..')

model = UNet(spatial_dims=3, in_channels=1, out_channels=n_classes, channels=(16, 32, 64, 128, 256),strides=(2, 2, 2, 2), num_res_units=2)
loss_func = CustomLoss(loss_func=DiceCELoss(to_onehot_y=True, include_background=True, softmax=True))

learn = Learner(dls, model, loss_func=loss_func, opt_func=ranger, metrics=multi_dice_score)#.to_fp16()
learn.lr_find()
plt.savefig("figures/lr_find_liver.png")
print("done with find")

lr = 1e-1
learn.fit_flat_cos(10 ,lr)
learn.save('liver-model')
learn.show_results(anatomical_plane=0, ds_idx=1)
plt.savefig("figures/fit_liver.png")
print("training stuff done, now inference")

learn.load('liver-model')
test_dl = learn.dls.test_dl(test_df[:10],with_labels=True)
test_dl.show_batch(anatomical_plane=0, figsize=(10,10))

pred_acts, labels = learn.get_preds(dl=test_dl)
multi_dice_score(pred_acts, labels)
learn.show_results(anatomical_plane=0, dl=test_dl)
plt.savefig("figures/results_liver.png")

print("done")


# below is based on the spleen notebook

# tasks = [x for x in os.listdir(data_path) if x.startswith('Task')]

# # Check if the path exists
# if os.path.exists(data_path):
#     # List all directories that start with 'Task'
#     tasks = [x for x in os.listdir(data_path) if x.startswith('Task') and os.path.isdir(os.path.join(data_path, x))]
# else:
#     print(f"Path {data_path} does not exist!")

# data_dir = os.path.join(data_path, "Task03_Liver")

# train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
# train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
# data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
# train_files, val_files = data_dicts[:-9], data_dicts[-9:]

# set_determinism(seed=0)

# val_transforms = Compose(
#     [
#         LoadImaged(keys=["image", "label"]),
#         EnsureChannelFirstd(keys=["image", "label"]),
#         ScaleIntensityRanged(
#             keys=["image"],
#             a_min=-57,
#             a_max=164,
#             b_min=0.0,
#             b_max=1.0,
#             clip=True,
#         ),
#         # CropForegroundd(keys=["image", "label"], source_key="image"),
#         # Orientationd(keys=["image", "label"], axcodes="RAS"),
#         # Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
#     ]
# )

# train_transforms = Compose()

# check_ds = Dataset(data=val_files, transform=val_transforms)
# check_loader = DataLoader(check_ds, shuffle=True, batch_size=1)
# check_data = first(check_loader)
# image, label = (check_data["image"][0][0], check_data["label"][0][0])
# print(label)
# print(f"image shape: {image.shape}, label shape: {label.shape}")
# # plot the slice [:, :, 80]
# plt.figure("check", (12, 6))
# plt.subplot(1, 2, 1)
# plt.title("image")
# plt.imshow(image[:,:,80], cmap="gray")
# plt.subplot(1, 2, 2)
# plt.title("label")
# plt.imshow(label[:,:,80])
# plt.savefig("figures/liver.png")


