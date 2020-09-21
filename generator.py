import torch.nn as nn

class Generator(nn.Module):
    def __init__(self,inputsize,hiddensize,outputsize):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(inputsize,hiddensize*8,4,1,0),
            nn.BatchNorm2d(hiddensize*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(hiddensize*8,hiddensize*4,4,2,1),
            nn.BatchNorm2d(hiddensize*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(hiddensize*4,hiddensize*2,4,2,1),
            nn.BatchNorm2d(hiddensize*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(hiddensize*2,hiddensize,4,2,1),
            nn.BatchNorm2d(hiddensize),
            nn.ReLU(True),

            nn.ConvTranspose2d(hiddensize,outputsize,4,2,1),
            nn.Tanh()
        )

    def forward(self,input):
        return self.main(input)