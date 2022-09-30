import torch.utils.data as data
import json
import os, sys
import subprocess
from PIL import Image
import numpy as np
import torch
import pandas as pd
import csv
import torchvision.transforms as transforms

urls = {
    "image_set": "http://press.liacs.nl:8080/mirflickr/mirflickr25k.v3/mirflickr25k.zip",
    "annotation_set": "http://press.liacs.nl:8080/mirflickr/mirflickr25k.v3/mirflickr25k_annotations_v080.zip"}

object_categories = ['animals', 'baby', 'bird', 'car','clouds','dog',
                    'female', 'flower', 'food','indoor','lake','male',
                    'night','people','plant_life','portrait','river','sea',
                    'sky','structures', 'sunset', 'transport','tree','water',
                    ] # length==24

def get_all_annotation_txt(root_dir):
    '''
    get the all annotation txt file name
    :param root:
    :return:
    '''
    txt_name_list = []
    class_name = []
    file_name_list = []
    base_root = ''
    all_txts = []
    all_dirs = []
    for root, all_dirs, all_txts in os.walk(root_dir):
        base_root = root
        for file in all_txts:
            f_str = list(os.path.splitext(file))
            if f_str[-1] == '.txt' and 'r1' not in str(f_str[0]) and "READ" not in str(f_str[0]):
                txt_name_list.append(os.path.join(root, file))
                file_name_list.append(file)
                class_name.append(f_str[0])
    class_name = sorted(class_name)
    file_name_list = sorted(file_name_list)

    return txt_name_list, class_name, file_name_list, base_root


def write_csv(path_name, class_name, file_name, csv_file, root):
    img_label_dict = {}
    for i in range(len(path_name)):
        txt_name = str(file_name[i].split('.')[-2])
        with open(str(root + "/" + file_name[i]), 'r') as f:
            all_lines = f.readlines()
            for item in all_lines:
                info = item.split()[0]
                if str(info) not in img_label_dict.keys():
                    img_label_dict[str(info)] = [-1 for ii in range(len(class_name))]
                img_label_dict[str(info)][class_name.index(txt_name)] = 1

    print('[dataset] write file %s' % csv_file)
    with open(csv_file, 'w') as csvfile:
        fieldnames = ['name']
        fieldnames.extend(class_name)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for (name, labels) in img_label_dict.items():
            example = {'name': name}
            for i in range(len(class_name)):
                example[fieldnames[i + 1]] = int(labels[i])
            writer.writerow(example)
    
    csvfile.close()


def download_mirflickr25k(root, phase):
    if not os.path.exists(root):
        os.makedirs(root)
    tmpdir = os.path.join(root, 'tmp/')
    
    data = os.path.join(root, 'data/')  
    if not os.path.exists(data):
        os.makedirs(data)
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)
    filename = ''
    if phase == 'image_set':
        filename = 'mirflickr25k.zip'
    elif phase == 'annotation_set':
        filename = 'mirflickr25k_annotations_v080.zip'
    cached_file = os.path.join(tmpdir, filename)
    if not os.path.exists(cached_file):
        print('Downloading: "{0}" to {1}\n'.format(urls[phase], cached_file))
        os.chdir(tmpdir)
        subprocess.call('wget ' + urls[phase], shell=True)
        os.chdir(root)

    img_data = os.path.join(data, filename.split('.')[0])
    if not os.path.exists(img_data):
        print('[dataset] Extracting zip file {file} to {path}'.format(file=cached_file, path=data))
        command = 'unzip {0} -d {1}'.format(cached_file, data)
        os.system(command)
    
    if phase == 'image_set':
        print('[mirflickr25k image dataset] Done!')
    if phase == 'annotation_set':
        print('[mirflickr25k annotation dataset] Done!')


def read_object_labels_csv(file, header=True):
    images = []
    num_categories = 0
    print('[dataset] read', file)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        for row in reader:
            if header and rownum == 0:
                header = row
            else:
                if num_categories == 0:
                    num_categories = len(row) - 1
                name = row[1]
                labels = (np.asarray(row[2:num_categories + 1])).astype(np.float32)
                # labels[labels == -1] = 0
                labels = torch.from_numpy(labels)
                item = (name, labels)
                images.append(item)
            rownum += 1
    return images

def pandas_split(path, per=0.4):
    '''
    split one .csv into train.csv and test.csv, use only once, because of the samples are selected randomly
    :param path:
    :param per:
    :return:
    '''
    df = pd.read_csv(path, encoding='utf-8')
    df = df.sample(frac=1.0)
    cut_idx = int(round(per * df.shape[0]))
    df_test, df_train = df.iloc[:cut_idx], df.iloc[cut_idx:]
    # df_test.to_csv('/media/data/maleilei/MLICdataset/mirflickr25k/csv_files/mirflickr25k_test_' + str(per*10) + '.csv')
    # df_train.to_csv('/media/data/maleilei/MLICdataset/mirflickr25k/csv_files/mirflickr25k_train_'+ str((1-per)*10) +'.csv')
    df_test.to_csv('/media/data1/maleilei/dataset/mirflickr25k/csv_files/mirflickr25k_test' + '.csv')
    df_train.to_csv('/media/data1/maleilei/dataset/mirflickr25k/csv_files/mirflickr25k_train' + '.csv')
    print(df.shape, df_test.shape, df_train.shape)


class MirFlickr25kPreProcessing(data.Dataset):
    def __init__(self, root, set, target_transform=None):

        self.root = root
        self.path_images = os.path.join(root, 'data','mirflickr25k')
        self.set = set
        self.transform = target_transform
        # download_mirflickr25k(self.root,'image_set')
        # download_mirflickr25k(self.root, 'annotation_set')

        path_csv = os.path.join(self.root, 'csv_files')  
        file_csv = os.path.join(path_csv, 'mirflickr25k_' + set + '.csv')
        # print(file_csv)
        
        if not os.path.exists(file_csv):
            if not os.path.exists(path_csv):  
                os.makedirs(path_csv)

            file_csv_all = os.path.join(path_csv, 'dataset.csv')
            txt_name_list, class_name, file_name_list, base_root=get_all_annotation_txt(os.path.join(self.root, 'data','mirflickr25k_annotations_v080') )
            write_csv(txt_name_list, class_name, file_name_list, file_csv_all, base_root)
            pandas_split(file_csv_all, per=0.4)

        self.classes = object_categories

        self.images = read_object_labels_csv(file_csv)


        print('[dataset] MirFlickr25k classification set=%s number of classes=%d  number of images=%d' % (
            set, len(self.classes), len(self.images)))

    def __getitem__(self, index):

        path, target = self.images[index]
        img = Image.open(os.path.join(self.path_images,'im'+ path + '.jpg')).convert('RGB') 
        if self.transform:
            img = self.transform(img)

        # target[target==0] = -1
        # target = torch.from_numpy(target)

        data = {'image':img, 'name': 'im'+ path + '.jpg', 'target': target}
        return data

    def __len__(self):
        '''
        return the amount of elements
        :return:
        '''
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)



if __name__ == "__main__":

    data_dir = '/media/data1/maleilei/dataset/mirflickr25k'
    trainset = MirFlickr25kPreProcessing(data_dir, set='train', target_transform=None)
    print(trainset[24])
    testset = MirFlickr25kPreProcessing(data_dir, set='test', target_transform=None)
    print(len(trainset))
    print(len(testset))
    print(len(trainset) / len(testset))
    # print(testset[24])
    # for index, value in enumerate(trainset):
    #     print(trainset[index])
    # print('trainset is ok')
    # for index, value in enumerate(testset):
    #     testset[index]
        # print(testset[index])
    print('testset is ok')
