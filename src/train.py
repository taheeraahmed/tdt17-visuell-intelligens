from datasets import MSDataset
from torch.utils.data import DataLoader
from torchvision import transforms

if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = MSDataset(annotations_file='path_to_annotations.csv', img_dir='path_to_images', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


    for images, labels in dataloader:
        pass


