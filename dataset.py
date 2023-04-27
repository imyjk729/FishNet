import pickle
import numpy as np 
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torchvision.transforms import Normalize, RandomHorizontalFlip


class GetDataset(Dataset):
    def __init__(self, data, label, transform):
        self.data = data
        self.label = label
        self.transform = transform

    def __getitem__(self, index):
        X = torch.tensor(self.data[index], dtype=torch.float32)
        y = torch.tensor(self.label[index], dtype=torch.int64)

        if self.transform:
            X = self.transform(X)

        return X, y
    
    def __len__(self):
        return len(self.data)
    

class GetCIFAR10(object):
    def __init__(self, args):
        self.args = args
        self.train_transform = transforms.Compose([
            # RandomResizedCrop(img_size),
            RandomHorizontalFlip(),
            # Normalize(mean=[0.485, 0.456, 0.406],
            #           std=[0.229, 0.224, 0.225])
        ])

        self.test_transform = transforms.Compose([
            # Normalize(mean=[0.485, 0.456, 0.406],
            #           std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def unpickle(file):
        """
        pickle로 파이썬 객체 load
        """
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
    def load_data(self):
        """
        pickle로 dictionary를 load하여 np.array로 train 데이터 구성
        """
        data = []
        labels = []
        for i in range(1, 6):
            filename = self.args.DATA_PATH + 'data_batch_' + str(i)
            data_dict = GetCIFAR10.unpickle(filename)
            data.append(data_dict[b'data'])
            labels += data_dict[b'labels']
        data = np.concatenate(data, axis=0)
        data = np.reshape(data, (50000, 3, 32, 32))
        data = data.reshape(50000,-1)
        labels = np.array(labels).reshape(-1,1)
        concat_data = np.concatenate((data, labels), axis=1)

        return concat_data
    
    def load_test(self):
        """
        pickle로 dictionary를 load하여 np.array로 test 데이터 구성
        """
        filename = self.args.DATA_PATH  + 'test_batch'
        data_dict = GetCIFAR10.unpickle(filename)
        data = data_dict[b'data']
        labels = data_dict[b'labels']
        data = np.reshape(data, (10000, 3, 32, 32))

        return data, labels
    
    def get_loader(self):
        """
        train dataloader, valid dataloader 구성
        """
        data = self.load_data()
        train_data, valid_data = train_test_split(data, train_size=self.args.ratio, shuffle=True)
        train_dset = GetDataset(data = train_data[:,:-1].reshape(-1, 3, 32, 32),
                                label = train_data[:,-1], transform = self.train_transform)
        valid_dset = GetDataset(data = valid_data[:,:-1].reshape(-1, 3, 32, 32),
                                label = valid_data[:,-1], transform = self.test_transform)
        
        train_loader = DataLoader(train_dset, shuffle=True,
                                  batch_size=self.args.batch_size, num_workers=4)
        valid_loader = DataLoader(valid_dset, shuffle=False,
                                  batch_size=self.args.batch_size, num_workers=4)
        
        return train_loader, valid_loader
    
    def get_test(self):
        """
        test dataloader 구성
        """
        data, labels = self.load_test()
        test_dset = GetDataset(data, labels, self.test_transform)
        test_loader = DataLoader(test_dset, shuffle=False, 
                                 batch_size=self.args.batch_size, num_workers=4)
        
        return test_loader
    