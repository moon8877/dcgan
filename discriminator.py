import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self,inputsize,hiddensize):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(inputsize,hiddensize,4,2,1),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(hiddensize,hiddensize*2,4,2,1),
            nn.BatchNorm2d(hiddensize*2),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(hiddensize*2,hiddensize*4,4,2,1),
            nn.BatchNorm2d(hiddensize*4),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(hiddensize*4,hiddensize*8,4,2,1),
            nn.BatchNorm2d(hiddensize*8),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(hiddensize*8,1,4,1,0),
            nn.Sigmoid()

        )

    def forward(self,input):
        return self.main(input)