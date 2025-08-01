#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 12:34:49 2020

@author: apramanik
"""


from torch.utils.data import DataLoader
from torch import optim
import numpy as np
import os, torch, time
import torch.nn as nn
from scipy.io import savemat
from datetime import datetime


from train_val_dataset import cardiacdata
from UNET3D_D4 import UNet3D

'''
Save training model
'''
def save_checkpoint(states,  path, filename='model_best.pth.tar'):
    if not os.path.exists(path):
        os.makedirs(path)
    checkpoint_name = os.path.join(path,  filename)
    torch.save(states, checkpoint_name)


'''
Compute dice coefficients
'''
def dice_comp(pred, gt):
    return (2. * (np.sum(pred.astype(float) * gt.astype(float))) + 1.) / (np.sum(pred.astype(float)) \
        + np.sum(gt.astype(float)) + 1.)


'''
Train the model
'''
def train_unet(depth, learning_rate, Weight_Decay, chunk_size, batch_Size, nImg, epochs, savemodNepoch):
    """
    Train on iowa96.
    """
    print ('*************************************************')
    start_time=time.time()
    saveDir='savedModels/'
    directory=saveDir+datetime.now().strftime("%d%b_%I%M%P_")+str(nImg)+'I_'+  str(epochs)+'E_'+str(batch_Size)+'B'
    net = UNet3D(num_classes=4, in_channels=1, depth=depth, start_filts=32, up_mode="transpose", res=True).cpu()
    params=list(net.parameters())
    
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, \
                        weight_decay=Weight_Decay)

    loss_fn = nn.CrossEntropyLoss()
    best_loss = float("inf")
    epochloss,validLoss,ep=[],[],0
    val_dataset = cardiacdata('validation', 0, chunk_size, nImg)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    for epoch in range(epochs):
        epoch=epoch+1
        tr_dataset = cardiacdata('training', epoch, chunk_size, nImg)
        tr_loader = DataLoader(tr_dataset, batch_size=batch_Size, shuffle=True, num_workers=0)
        net.train()
        ep=ep+1
        totalLoss=[]
        for step, (img, seg_gt) in enumerate(tr_loader, 1):
            img, seg_gt = img.cpu(), seg_gt.cpu()
            pred = net(img)
            loss = loss_fn(pred, seg_gt)
            totalLoss.append(loss.detach().cpu().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epochloss.append(np.mean(totalLoss))
        print("epoch: ", epoch, "tr loss: ", "%.5e" % epochloss[epoch-1])

        with torch.no_grad():
            net.eval()
            vLoss = []
            dice = np.zeros((5,3))
            for step, (img, seg_gt) in enumerate(val_loader, 0):
                img, seg_gt = img.cpu(), seg_gt.cpu()
                seg = net(img)
                loss = loss_fn(seg, seg_gt)
                vLoss.append(loss.detach().cpu().numpy())
                _, pred = torch.max(seg, 1)
                pred = pred.detach().cpu().numpy()
                gt = seg_gt.detach().cpu().numpy()
                for i in range(3):
                    dice[step, i] = dice_comp(pred==i+1, gt==i+1)
            validLoss.append(np.mean(vLoss))

       

        print("epoch: ", epoch, "val loss: ", "%.5e" % validLoss[epoch-1])
        print(["%1.4f" % np.mean(dice[:,i]) for i in range(3)])
        
        if np.remainder(ep,savemodNepoch)==0:
            save_checkpoint(
                {
            'epoch': ep,
            'state_dict': net.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict()
        },
        path=directory,
        filename='model-{}.pth.tar'.format(ep)
                )
    
        if validLoss[epoch-1] < best_loss:    
            best_loss = validLoss[epoch-1]
            save_checkpoint(
        {
            'epoch': ep,
            'state_dict': net.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict()
        },
        path=directory,
        )
    savemat(directory+'/epochloss.mat',mdict={'epochs':epochloss},appendmat=True)
    savemat(directory+'/validloss.mat',mdict={'epochs':validLoss},appendmat=True)
    end_time = time.time()    
    print ('Training completed in minutes ', ((end_time - start_time) / 60))
    print ('training completed at', datetime.now().strftime("%d-%b-%Y %I:%M %P"))
    print ('*************************************************')
    for i in range(len(params)):
        params[i]=params[i].detach().cpu().numpy()
    return epochloss,validLoss,best_loss,params




if __name__ == "__main__":
    
    learning_rate=1e-4
    Weight_Decay=1e-4
    chunk_size=1
    batch_Size=1
    nImg=70
    epochs=10000
    savemodNepoch=1000
    depth=4
    epochloss,validLoss,best_loss,weights=train_unet(depth, learning_rate, Weight_Decay, chunk_size, batch_Size, nImg, epochs, savemodNepoch)
    
        
