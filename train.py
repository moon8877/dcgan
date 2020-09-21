import random
import sys
sys.path.append('c:/Users/chou_wen_chi/Desktop/python/dcgan')
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets
import torchvision.transforms as transforms
import torchvision.utils as utils
import numpy as np
import matplotlib.pyplot as plt
import discriminator
import generator

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU state:',device)

manualSeed = 7777
print('Randomseed:',manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)




def train(device=device):
    batch_size = 1024
    image_size = 64
    G_out_D_in = 3
    G_in = 100
    G_hidden = 64
    D_hidden = 64
    epochs = 5
    lr = 0.001
    betal = 0.5

    dataset = torchvision.datasets.ImageFolder(root='./dcgan/pic',transform=transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ]
    ))

    dataloader = torch.utils.data.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)
    netg = generator.Generator(G_in,G_hidden,G_out_D_in).to(device)
    print(netg)

    netd = discriminator.Discriminator(G_out_D_in,D_hidden).to(device)
    print(netd)

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64,G_in,1,1,device=device)

    real_label = 1
    fake_label = 0

    optimizerd = optim.Adam(netd.parameters(),lr=lr,betas=(betal,0.999))
    optimizerg = optim.Adam(netg.parameters(),lr=lr,betas=(betal,0.999))
    img_list = []
    g_losses = []
    d_losses = []
    iters = 0
    print('start')
    for epoch in range(epochs):
        for i,data in enumerate(dataloader,0):
            netd.zero_grad()
            real_cpu = data[0].to(device)
            b_size =real_cpu.size(0)
            label = torch.full((b_size,),real_label,device=device)
            output = netd(real_cpu).view(-1)
            
            errD_real = criterion(output,label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size,G_in,1,1,device=device)
            fake = netg(noise)
            label.fill_(fake_label)
            output = netd(fake.detach()).view(-1)

            errd_fake = criterion(output,label)
            errd_fake.backward()

            d_g_z1 = output.mean().item()
            errd = errD_real+errd_fake
            optimizerd.step()

            netg.zero_grad()
            label.fill_(real_label)
            output = netd(fake).view(-1)
            errg = criterion(output,label)
            errg.backwar()
            d_g_z2 = output.mean().item()
            optimizerg.step()
            if(i%50 == 0):
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (epoch, epochs, i, len(dataloader), errd.item(), errg.item(), D_x, d_g_z1, d_g_z2))
            g_losses.append(errg.item())
            d_losses.append(errd.item())
            if (iters%500 == 0) or((epoch == epochs-1)and(i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netg(fixed_noise).detach().cpu()
                img_list.append(utils.make_grid(fake,padding=2,normalize=True))
            iters = iters+1

    torch.save(netd,'netd.pkl')
    torch.save(netd.state_dict(),'netd_parameters.pkl')
    torch.save(netg,'net.pkl')
    torch.save(netg.state_dict(),'net_parameters.pkl')
    return g_losses,d_losses,img_list
g_losses,d_losses,img_list = train()
def plotImage(G_losses, D_losses,img_list):
    print('Start to plot!!')
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(utils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.show()


