import torch
import torch.nn as nn


__all__ = ['SIMPLENN', 'simplenn']



class SIMPLENN(nn.Module):

    def __init__(self):
        super(SIMPLENN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2),
        )
        self.rnn = nn.LSTM(4096,512,2,bidirectional=True)
        self.out = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        x = x.transpose(0, 1)
        view_pool = []
        
        for v in x:
            # v = self.features(v)
            
            v = v.view(v.size(0), 64*64*1)
            view_pool.append(v)
        
        view_pool = torch.stack(view_pool)
        # print(view_pool.shape)
        out,h = self.rnn(view_pool)
        # print(out.shape)
        # print(h.shape)
        predict = self.out(out)
        predict = torch.squeeze(predict[-1::])
        # pooled_view = view_pool[0]
        # for i in range(1, len(view_pool)):
        #     pooled_view = torch.max(pooled_view, view_pool[i])
        
        # pooled_view = self.classifier(pooled_view)
        # return pooled_view
        return predict


def simplenn(pretrained=False, **kwargs):
    model = SIMPLENN(**kwargs)
    return model