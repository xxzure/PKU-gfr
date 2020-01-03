from torch.utils.data.dataset import Dataset
import os
from PIL import Image
import pandas as pd

class MultiViewDataSet(Dataset):

    def find_gfr(self, dir):
        df=pd.read_csv('gfr.csv')
        data_dict = {}      
        def map_dict(item):
            data_dict[item["name"]] = [item["gfr_left"],item["gfr_right"],item["gfr_all"],item["age"],item["height"],item["weight"],item["depth_left"],item["depth_right"]]
        df.apply(map_dict,axis=1)
        
        return data_dict

    def __init__(self, root, data_type, transform=None, target_transform=None):
        self.x = []
        self.gfr = []
        self.info = []
        self.root = root

        self.data_dict = self.find_gfr(root)

        self.transform = transform
        self.target_transform = target_transform
        # root / <train/test> / <item> / <view>.png
        for item in os.listdir(root + '/' + data_type):
            views = []
            path = root + '/' + data_type + '/' + item
            if os.path.isdir(path):
                for view in os.listdir(root + '/' + data_type + '/' + item):
                    views.append(root + '/' + data_type + '/' + item + '/' + view)
                self.x.append(views)
                self.gfr.append([self.data_dict[item][0],self.data_dict[item][1],self.data_dict[item][2]])
                self.info.append([self.data_dict[item][3],self.data_dict[item][4],self.data_dict[item][5],self.data_dict[item][6]])

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        orginal_views = self.x[index]
        views = []

        for view in orginal_views:
            im = Image.open(view)
            im = im.convert('L')
            if self.transform is not None:
                im = self.transform(im)
            views.append(im)

        return views, self.gfr[index], self.info[index]

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.x)
