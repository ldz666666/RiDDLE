import torch
import os
from torch.utils.data import Dataset
import cv2
from PIL import Image
from torchvision import transforms as trans 
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import default_collate
import shutil
import numpy as np

class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=128, starge='train'):
        if starge == 'train':
            #self.length = 8140 # train id
            self.img_path=os.path.join(path,'celeba_train')
        elif starge == 'test':
            self.img_path=os.path.join(path,'celeba_test')
        
        #definition of dataset's length
        self.img_list=os.listdir(os.path.join(self.img_path,'clr'))
        self.length=len(self.img_list)

        print('constructing dataset, now path is',self.img_path)
        
        self.resolution = resolution
        self.transform = transform
        self.data_path=os.path.join(self.img_path,'clr')
        self.msk_path=os.path.join(self.img_path,'msk')

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        id=self.img_list[index]
        
        files1=os.listdir(os.path.join(self.data_path,str(index)))
        files2=os.listdir(os.path.join(self.msk_path,str(index)))
        files=list(set(files1).intersection(set(files2)))
        site = np.random.randint(len(files), size=2)
        image_name1 = files[site[0]]
        image_name2 = files[site[1]]

        img1,msk1=Image.open(os.path.join(self.data_path,str(index),image_name1)),Image.open(os.path.join(self.msk_path,str(index),image_name1))
        img2,msk2=Image.open(os.path.join(self.data_path,str(index),image_name2)),Image.open(os.path.join(self.msk_path,str(index),image_name2))

        img1=self.transform(img1)
        msk1=self.transform(msk1)
        img2=self.transform(img2)
        msk2=self.transform(msk2)

        return img1,msk1,img2,msk2


class SimplePairDataset(Dataset):

    def __init__(self,path1,path2,transform=None):
        #make sure path has only images
        print('Image path1',path1)
        print('Image path2',path2)
        self.file_path1=path1
        self.file_path2=path2
        self.file_list=os.listdir(path1)
        self.file_list.sort()
        #print(self.file_list)
        if transform is not None:
            self.transform=transform
        else:
            self.transform=trans.Compose([trans.ToTensor()])

    def __getitem__(self,index):
        file_name=self.file_list[index]
        #print(index)
        #print(file_name)
        #image=cv2.imread(os.path.join(self.file_path,file_name))
        #image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image1=Image.open(os.path.join(self.file_path1,file_name))
        if self.transform is not None:
            image1=self.transform(image1)
        image2=Image.open(os.path.join(self.file_path2,file_name))
        if self.transform is not None:
            image2=self.transform(image2)
        #print('image shape is',image.shape)
        return image1,image2,file_name
        
    def __len__(self):
        return len(self.file_list)


class SimpleDataset(Dataset):

    def __init__(self,path,transform=None):
        #make sure path has only images
        print('Image path',path)
        self.file_path=path
        self.file_list=os.listdir(path)
        self.file_list.sort()
        #print(self.file_list)
        if transform is not None:
            self.transform=transform
        else:
            self.transform=trans.Compose([trans.ToTensor()])

    def __getitem__(self,index):
        file_name=self.file_list[index]
        #print(index)
        #print(file_name)
        #image=cv2.imread(os.path.join(self.file_path,file_name))
        #image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image=Image.open(os.path.join(self.file_path,file_name)).convert('RGB')
        #image=Image.open(os.path.join(self.file_path,file_name)
        if self.transform is not None:
            image=self.transform(image)
        #print('image shape is',image.shape)
        return image,file_name
        
    def __len__(self):
        return len(self.file_list)


class MyImageFolder(ImageFolder):
    def __getitem__(self,index):
        try:
            return super(MyImageFolder,self).__getitem__(index)
        except:
            return None,None

def My_collate_fn(batch):
    
    if isinstance(batch, list):
        batch = [(image, image_id) for (image, image_id) in batch if image is not None]
    if batch==[]:
        return (None,None)
    return default_collate(batch)

            
def my_collate_fn(batch):
    
    def my_collate_fn(batch):
        batch=list(filter(lambda x:x[0] is not None,batch))
        return default_collate(batch)

#get image pair from different id, dataset format should like imagefolder
class PairDataset():
    def __init__(self, path, transform, resolution=128, starge='train'):
        self.path=path
        self.folder_list=os.listdir(path)

        self.img_list,self.label_list,self.id_list=[],[],[]
        
        for i,id in enumerate(self.folder_list):
            imglist_all= [os.path.join(path,id,file) for file in os.listdir(os.path.join(path,id))]
            #number of the image
            self.img_list+=imglist_all
            self.label_list+=[i]*len(imglist_all)
            #self.id_list+=[os.path.join(path,id)]*(len(imglist_all))

        self.length=len(self.img_list)
        #print('constructing dataset, now path is',self.img_path)
        self.resolution = resolution
        self.transform = transform
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):

        index2=np.random.randint(len(self.img_list))
        #print('index2 is',index2)
        while self.label_list[index2]==self.label_list[index]:
            index=np.random.randint(len(self.img_list))
        
        image_name1 = self.img_list[index]
        image_name2 = self.img_list[index2]

        img1,img2=Image.open(image_name1),Image.open(image_name2)

        img1=self.transform(img1)
        img2=self.transform(img2)

        return img1,image_name1,img2,image_name2


class ID_Folder_Dataset():
    def __init__(self, path, transform, resolution=128, starge='train'):
        self.path=path
        self.folder_list=os.listdir(path)

        self.img_list,self.label_list,self.id_list=[],[],[]
        
        for i,id in enumerate(self.folder_list):
            imglist_all= [os.path.join(path,id,file) for file in os.listdir(os.path.join(path,id))]
            #number of the image
            self.img_list+=imglist_all
            self.label_list+=[i]*len(imglist_all)
            #self.id_list+=[os.path.join(path,id)]*(len(imglist_all))

        self.length=len(self.img_list)
        #print('constructing dataset, now path is',self.img_path)
        self.resolution = resolution
        self.transform = transform
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):

        # index2=np.random.randint(len(self.img_list))
        # #print('index2 is',index2)
        # while self.label_list[index2]==self.label_list[index]:
        #     index=np.random.randint(len(self.img_list))
        
        image_name1 = self.img_list[index]
        # image_name2 = self.img_list[index2]

        img1=Image.open(image_name1)

        img1=self.transform(img1)
        # img2=self.transform(img2)

        return img1,image_name1



def merge_all(source_path,target_path):
    os.makedirs(target_path,exist_ok=True)
    for root,folders,filenames in os.walk(source_path):
        for f in filenames:
            print(f'copying {f}')
            print(f'source {os.path.join(root,f)}, target {os.path.join(target_path,f)}')
            shutil.copy(os.path.join(root,f),os.path.join(target_path,f))
    print('merge finished') 