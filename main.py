# Load core modules and classes
import os
import pdb
import argparse
from csv import writer, reader
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
from nibabel import load as nib_imgload

import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary

from logger import Logger
from probabilistic_unet import ProbabilisticUnet
from utils import l2_regularisation
#np.random.seed(10)

# Define handy functions
# For visualization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def quick_volume_view( vol, dim, nStep, fignum, clim=(0,1000)):
    counter=0
    nX, nY, nZ = vol.shape
    if dim==1:   step = np.round(np.linspace(0, nX-1, nStep))
    elif dim==2: step = np.round(np.linspace(0, nY-1, nStep))
    else:        step = np.round(np.linspace(0, nZ-1, nStep))
    plt.figure(fignum,figsize=(18, 2))    
    for idx in step:
        counter += 1
        plt.subplot( 1, nStep,counter)
        if dim==1:   plt.imshow( vol[int(idx),:,:], cmap='gray' )
        elif dim==2: plt.imshow( vol[:,int(idx),:], cmap='gray' )
        else:        plt.imshow( vol[:,:,int(idx)], cmap='gray' )
        plt.title('slice #'+str(int(idx)))
        plt.xticks([])
        plt.yticks([])
        plt.clim(clim)
    plt.show()

# Define reshaping function
def reshape_3d( vol, shape, method='nearest' ):
    f1,f2,f3 = shape[0]/vol.shape[0], shape[1]/vol.shape[1], shape[2]/vol.shape[2]
    vol_reshape = zoom(vol, (f1,f2,f3), mode=method )
    return vol_reshape


class MedicalDataset():
    def __init__(self, mother_dir='/st2/hyewon/dataset/decathlon/', task_dir='Task02_Heart/', mode='train', iteration=0):
        self.mother_dir = mother_dir
        self.task_dir = task_dir
        self.mode = mode
        self.iteration = iteration

        self.rawimg_dir = self.mother_dir + self.task_dir + 'imagesTr/'
        self.labimg_dir = self.mother_dir + self.task_dir + 'labelsTr/'

        self.trainset_fname = []
        self.validset_fname = [] 
        self.testset_fname = []
        
        validsplit = 0.6
        testsplit = 0.7

        count = 0
        for fname in sorted(os.listdir(self.rawimg_dir)):
            if fname[0] != '.':
                count += 1
        
        num_set = count
        valididx = int(num_set*validsplit)
        testidx = int(num_set*testsplit)
        
        count2 = 0
        for fname in sorted(os.listdir(self.rawimg_dir)):
            if fname[0] != '.':
                count2 += 1
                if count2 < valididx:
                    self.trainset_fname.append(fname)
                elif count2 < testidx:
                    self.validset_fname.append(fname)
                else:
                    self.testset_fname.append(fname)    

        print('nTrain: '+ str(len(self.trainset_fname)) + ', nValid: ' + str(len(self.validset_fname)) + ', nTest: '+ str(len(self.testset_fname)))
        
        if self.mode == 'train':
            self.dataset_fname = self.trainset_fname
        elif self.mode == 'valid':
            self.dataset_fname = self.validset_fname
        else:
            self.dataset_fname = self.testset_fname
        
    def __len__(self):
        return len(self.dataset_fname)
    '''
    def __getitem__(self, idx):
        x_batch = []
        y_batch = []
        filename = self.dataset_fname[idx]
        train_nii = nib_imgload(self.rawimg_dir+filename )
        train_img = train_nii.get_data()
        label_nii = nib_imgload(self.labimg_dir+filename )
        label_img = label_nii.get_data()
        
        axis = np.random.choice(3, 1)
        if axis == 0:  # coronal
            for i in range(train_img.shape[0]):
                img = train_img[i,:,:]
                img = img / 4095.
                img = np.expand_dims(img, axis=0)
                x_batch.append(img)
                label = label_img[i,:,:]
                label = np.expand_dims(label, axis=0)
                y_batch.append(label)
        elif axis == 1:  # axial
            for i in range(train_img.shape[1]):
                img = train_img[:,i,:]
                img = img / 4095.
                img = np.expand_dims(img, axis=0)
                x_batch.append(img)
                label = label_img[:,i,:]
                label = np.expand_dims(label, axis=0)
                y_batch.append(label)                
        elif axis == 2:  # sagittal
            for i in range(train_img.shape[2]):
                img = train_img[:,:,i]
                img = img / 4095.
                img = np.expand_dims(img, axis=0)
                x_batch.append(img)
                label = label_img[:,:,i]
                label = np.expand_dims(label, axis=0)
                y_batch.append(label)
        idx = np.random.choice(len(x_batch), 10, False)
        x_batch, y_batch = np.array(x_batch, np.float32)[idx], np.array(y_batch, np.float32)[idx]
        
        return x_batch, y_batch
    '''
    def next(self):
        num_file = int(np.random.choice(len(self.dataset_fname), 1))
        filename = self.dataset_fname[num_file]
        train_nii = nib_imgload(self.rawimg_dir+filename)
        train_img = train_nii.get_data()
        label_nii = nib_imgload(self.labimg_dir+filename)
        label_img = label_nii.get_data()

        #axis = np.random.choice(3, 1)
        axis = 1
        if axis == 0:  # coronal
            i = np.random.randint(train_img.shape[0])
            img = train_img[i,:,:]
            img = img / 4095.
            img = np.expand_dims(img, axis=0)
            label = label_img[i,:,:]
            label = np.expand_dims(label, axis=0)
        elif axis == 1:  # axial
            i = np.random.randint(train_img.shape[1])
            img = train_img[:,i,:]
            img = img / 4095.
            img = np.expand_dims(img, axis=0)
            label = label_img[:,i,:]
            label = np.expand_dims(label, axis=0)               
        elif axis == 2:  # sagittal
            i = np.random.randint(train_img.shape[2])
            img = train_img[:,:,i]
            img = img / 4095.
            img = np.expand_dims(img, axis=0)
            label = label_img[:,:,i]
            label = np.expand_dims(label, axis=0)
        self.iteration += 1

        return img, np.array(label, np.float32)

def train(args):
    num_epoch = args.epoch
    learning_rate = args.learning_rate
    task_dir = args.task
    
    trainset = MedicalDataset(task_dir=task_dir, mode='train' )
    validset = MedicalDataset(task_dir=task_dir, mode='valid')

    model =  ProbabilisticUnet(input_channels=1, num_classes=1, num_filters=[32,64,128,192], latent_dim=2, no_convs_fcomb=4, beta=10.0)
    model.to(device)
    #summary(model, (1,320,320))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    criterion = torch.nn.BCELoss()

    for epoch in range(num_epoch):
        model.train()
        while trainset.iteration < args.iteration:
            x, y = trainset.next()
            x, y = torch.from_numpy(x).unsqueeze(0).cuda(), torch.from_numpy(y).unsqueeze(0).cuda()
            #print(x.size(), y.size())
            #output = torch.nn.Sigmoid()(model(x))
            model.forward(x,y,training=True)
            elbo = model.elbo(y)

            reg_loss = l2_regularisation(model.posterior) + l2_regularisation(model.prior) + l2_regularisation(model.fcomb.layers)
            loss = -elbo + 1e-5 * reg_loss
            #loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        trainset.iteration = 0

        model.eval()
        with torch.no_grad():
            while validset.iteration < args.test_iteration:
                x, y = validset.next()
                x, y = torch.from_numpy(x).unsqueeze(0).cuda(), torch.from_numpy(y).unsqueeze(0).cuda()
                #output = torch.nn.Sigmoid()(model(x, y))
                model.forward(x,y,training=True)
                elbo = model.elbo(y)

                reg_loss = l2_regularisation(model.posterior) + l2_regularisation(model.prior) + l2_regularisation(model.fcomb.layers)
                valid_loss = -elbo + 1e-5 * reg_loss
            validset.iteration = 0
                
        print('Epoch: {}, elbo: {:.4f}, regloss: {:.4f}, loss: {:.4f}, valid loss: {:.4f}'.format(epoch+1, elbo.item(), reg_loss.item(), loss.item(), valid_loss.item()))
        """
        #Logger
         # 1. Log scalar values (scalar summary)
        info = { 'loss': loss.item(), 'accuracy': valid_loss.item() }

        for tag, value in info.items():
            Logger.scalar_summary(tag, value, epoch+1)

        # 2. Log values and gradients of the parameters (histogram summary)
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            Logger.histo_summary(tag, value.data.cpu().numpy(), epoch+1)
            Logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch+1)
        """
    torch.save(model.state_dict(), './save/'+trainset.task_dir+'model.pth')

def visualize(args, path):
    model = ProbabilisticUnet(input_channels=1, num_classes=1, num_filters=[32,64,128,192], latent_dim=2, no_convs_fcomb=4, beta=10.0)
    model.to(device)
    model.load_state_dict(torch.load(path))
    task_dir = args.task
    
    testset = MedicalDataset(task_dir=task_dir, mode='test')
    testloader = data.DataLoader(testset, batch_size=1, shuffle=False)
    
    model.eval()
    with torch.no_grad():
        while testset.iteration < args.test_iteration:
            x, y = testset.next()
            x, y = torch.from_numpy(x).unsqueeze(0).cuda(), torch.from_numpy(y).unsqueeze(0).cuda()
            #output = torch.nn.Sigmoid()(model(x))
            #output = torch.round(output)   
            output = model.forward(x,y,training=True)
            output = torch.round(output)
#elbo = model.elbo(y)

#            reg_loss = l2_regularisation(model.posterior) + l2_regularisation(model.prior) + l2_regularisation(model.fcomb.layers)
#            valid_loss = -elbo + 1e-5 * reg_loss
            print (x.size(), y.size(), output.size())

            grid = torch.cat((x,y,output), dim=0)
            torchvision.utils.save_image(grid, './save/'+testset.task_dir+'prediction'+str(testset.iteration)+'.png', nrow=8, padding=2, pad_value=1)
            #torchvision.utils.save_image(y, './save/'+testset.task_dir+'gt'+str(testset.iteration)+'.png', nrow=8, padding=2)
            #torchvision.utils.save_image(output, './save/'+testset.task_dir+'predict'+str(testset.iteration)+'.png', nrow=8, padding=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['train', 'visualize'], default="train")
    parser.add_argument("--task", default='Task02_Heart/')
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--iteration", type=int, default=100)
    parser.add_argument("--test_iteration", type=int, default=10)
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'visualize':
        visualize(args, './save/'+args.task+'model.pth')
