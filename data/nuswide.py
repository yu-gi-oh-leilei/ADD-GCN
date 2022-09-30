import os, sys
import os.path as osp
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image

class NusWideDataset(Dataset):
    def __init__(self, 
        img_dir,
        anno_path, 
        labels_path,
        transform=None,
        rm_no_label_data=True,
        phase = None
        ) -> None:
        """[summary]
        Args:
            img_dir ([type]): dir of imgs
            anno_path ([type]): list of used imgs
            labels_path ([type]): labels of used imgs
            transform ([type], optional): [description]. Defaults to None.
        """
        super().__init__()

        self.img_dir = img_dir
        self.anno_path = anno_path
        self.labels_path = labels_path
        self.transform = transform
        self.rm_no_label_data = rm_no_label_data
        self.classes = 81

        self.itemlist = self.preprocess() # [(imgpath, label),...]
        print('[dataset] NusWide classification phase={} number of classes={}  number of images={}'.format(phase, self.classes, self.__len__()))
        '''
        [dataset] NusWide classification phase=Train number of classes=81  number of images=125449
        [dataset] NusWide classification phase=Test number of classes=81  number of images=83898
        '''

    def preprocess(self):
        imgnamelist = [line.strip().replace('\\', '/') for line in open(self.anno_path, 'r')]
        labellist = [line.strip() for line in open(self.labels_path, 'r')]
        assert len(imgnamelist) == len(labellist)

        res = []
        for idx, (imgname, labelline) in enumerate(zip(imgnamelist, labellist)):
            imgpath = osp.join(self.img_dir, imgname)
            labels = [int(i) for i in labelline.split(' ')]
            labels = np.array(labels).astype(np.float32)

            if sum(labels) == 0:
                continue
            
            res.append((imgpath, labels))

        return res
    
    def __len__(self) -> int:
        return len(self.itemlist)

    def __getitem__(self, index: int):
        imgpath, labels = self.itemlist[index]

        img = Image.open(imgpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        labels[labels==0] = -1
        labels = torch.from_numpy(labels)

        data = {'image':img, 'name': imgpath[:-19], 'target': labels}
        return data


# /media/data/maleilei/MLICdataset/nus_wide
if __name__ == '__main__':
    # train_label = osp.join(vg_root, 'NUS_WID_Tags','Train_Tags81.txt')
    # test_label = osp.join(vg_root, 'NUS_WID_Tags','Test_Tags81.txt')
    # path = '/media/data1/maleilei/dataset/nuswide/Flickr/actor/0001_2124494179.jpg'
    # print(len('0001_2124494179.jpg'))
    # print(path[:-19])
    ds = NusWideDataset(
        # img_dir = '/media/data/maleilei/MLICdataset/nus_wide/Flickr',
        # img_dir = '/media/data/maleilei/MLICdataset/NUS-WIDE-downloader/image',
        # /media/data/maleilei/MLICdataset/nuswide/slsplit
        img_dir = '/media/data1/maleilei/dataset/nuswide/Flickr',
        anno_path = '/media/data1/maleilei/dataset/nuswide/ImageList/TrainImagelist.txt',
        labels_path = '/media/data1/maleilei/dataset/nuswide/Groundtruth/Labels_Train.txt',
        transform = None,
        phase = 'Train'
    )
    # print(ds[0]['target'].size(-1))
    
    # /media/data/maleilei/MLICdataset/nuswide/slsplit
    # /media/data/maleilei/MLICdataset/nuswide/Groundtruth
    ds_test = NusWideDataset(
        # img_dir = '/media/data/maleilei/MLICdataset/nus_wide/Flickr',
        # img_dir = '/media/data/maleilei/MLICdataset/NUS-WIDE-downloader/image',
        img_dir = '/media/data1/maleilei/dataset/nuswide/Flickr',
        anno_path = '/media/data1/maleilei/dataset/nuswide/ImageList/TestImagelist.txt',
        labels_path = '/media/data1/maleilei/dataset/nuswide/Groundtruth/Labels_Test.txt',
        transform = None,
        phase = 'Test'
    )
    # # Test_label.txt  Test_split.txt  Train_label.txt  Train_split.txt
    # # Labels_Train.txt Labels_Test.txt
    print("len(ds):", len(ds)) 
    print("len(ds_test):", len(ds_test))
    # labels_path = '/media/data/maleilei/MLICdataset/nuswide/Groundtruth/Labels_Train.txt' # 161789
    # labels_path = '/media/data/maleilei/MLICdataset/nuswide/slsplit/Train_label.txt' # 150000

    


    # labellist = [line.strip() for line in open(labels_path, 'r')]
    # print(len(labellist))

    # anno_path = '/media/data/maleilei/MLICdataset/nuswide/oldtxt/train-file-list.txt'
    # imgnamelist = [line.strip().replace('\\', '/') for line in open(anno_path, 'r')]
    # print(len(imgnamelist))


    # ds = NusWideDataset(
    #         img_dir='/data/shilong/data/nus_wide/nuswide/Flickr',
    #         anno_path='/data/shilong/data/nus_wide/nuswide/ImageList/TrainImagelist.txt',
    #         labels_path='/data')