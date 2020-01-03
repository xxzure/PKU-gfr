import torch
import torch.nn as nn


__all__ = ['SIMPLENN', 'simplenn']



class SIMPLENN(nn.Module):

    def __init__(self):
        super(SIMPLENN, self).__init__()
        # self.features = nn.Sequential(
        #     nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.Conv2d(64, 192, kernel_size=5, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.Conv2d(192, 384, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(384, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        # )
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # self.classifier = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(256, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(256, 64),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(64, 2),
        # )
        self.rnn = nn.LSTM(3600,256,2)
        self.out = nn.Sequential(
            nn.Linear(3600, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(64, 2),
        )
        self.finalout = nn.Sequential(
            nn.Linear(6,16),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            # nn.Linear(32, 32),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(16, 3),
        )

    def forward(self, x, infos):
        x = x.transpose(0, 1)
        x = x[2:3]
        # print(x.shape)
        view_pool = []
        
        for v in x:
            v = self.features(v)
            # print(v.shape) #[8,16,7,7]
            v = v.view(v.size(0), 16*15*15)
            view_pool.append(v)
        
        view_pool = torch.stack(view_pool)
        # print(view_pool.shape)
        # out,h = self.rnn(view_pool)
        # print(h.shape)
        predict = self.out(view_pool)
        # print(predict.shape)
        predict = torch.squeeze(predict[-1::])

        info_inputs = torch.cat((predict,infos),-1)
        # print(info_inputs.shape)
        predict = self.finalout(info_inputs)
        # pooled_view = view_pool[0]
        # for i in range(1, len(view_pool)):
        #     pooled_view = torch.max(pooled_view, view_pool[i])
        
        # pooled_view = self.classifier(pooled_view)
        # return pooled_view
        return predict


def simplenn(pretrained=False, **kwargs):
    model = SIMPLENN(**kwargs)
    return model