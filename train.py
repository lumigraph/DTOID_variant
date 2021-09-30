from network import Network, CorrelationModel, ImageFeatExtract, TemplateFeatExtract
from torch.nn import CrossEntropyLoss, NLLLoss2d, MSELoss, BCELoss
#from dataset import Pacman
from dataset import UIScreen
from torch.utils import data
import torch.optim as optim
import torch
import cv2 as cv
import numpy as np
import os
from tqdm import tqdm

batch_size = 2
epochs = 200
output = 'output'
checkpoint = 'output/checkpoint/model_best.pth'
#checkpoint = ''


def IoU(y, pred):
    pred = pred.permute(0, 2, 3, 1).detach().cpu().numpy()
    y = y.permute(0, 2, 3, 1).detach().cpu().numpy()

    # print(np.sum((y == 1)), np.sum((pred == 1)))
    intersect = np.sum((y == 1) * (pred > 0.5))
    union = np.sum((y == 1) + (pred > 0.5))
    if union == 0:
        return 1.0
    return intersect / union


def train_step(model, data_loader, criterion, optimizer, epoch, best_iou):
    model.train()

    running_loss = 0.0
    iou = 0.0

    for i, data in enumerate(data_loader):
        img, tmp, msk = data
        (img, tmp, msk) = (
            img.cuda(), tmp.cuda(), msk.cuda())

        y = model.forward(img, tmp)

#        y = y.to(torch.float64)
#        msk = msk.to(torch.float64)

#        print(y)
#        print(msk)

        loss = criterion(y, msk)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        iou += IoU(msk, y)

        if i % 5 == 4:
            print('\nEpoch [%d/%d], iter %d: avg loss = %.3f, avg iou = %.3f, best_iou = %.3f' %
                  (epoch + 1, epochs, i + 1, running_loss / 5, iou / 5, best_iou))
            running_loss = 0.0
            iou = 0.0

        if epoch % 20 == 19:
            vis_img = img.detach().cpu().permute(0, 2, 3, 1).numpy()
            vis_img *= [0.229, 0.224, 0.225]
            vis_img += [0.485, 0.456, 0.406]
            vis_img *= 255
            vis = y.permute(0, 2, 3, 1).detach().cpu().numpy()

            for b in range(batch_size):
                vis_mini = vis[b]
                vis_img_mini = vis_img[b]
                vis_mini = cv.cvtColor(vis_mini * 255, cv.COLOR_GRAY2BGR)
                added_image = cv.addWeighted(
                    vis_img_mini[:, :, ::-1], 0.6, vis_mini, 0.4, 0)
                cv.imwrite('%s/pred/pred-%d.png' %
                           (output, i * batch_size + b), added_image)
    return model, optimizer


def train(model, train_data_loader, test_data_loader, criterion, optimizer, epochs=50, best_iou=0):
    model.train()

    best_loss = 1.0
    best_iou = best_iou
    for epoch in range(epochs):
        model, optimizer = train_step(
            model, train_data_loader, criterion, optimizer, epoch, best_iou)

        val_loss, val_iou = val(model, test_data_loader,
                                criterion, epoch % 20 == 19)

        if val_iou > best_iou:
            checkpoint = {'DTOID': model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'val_loss': val_loss,
                          'best_iou': val_iou
                          }
            torch.save(checkpoint, os.path.join(
                output, "checkpoint", 'model_best.pth'))

            print('Best IoU improve from %.5f to %.5f. Save best model ...' %
                  (best_iou, val_iou))
            best_iou = val_iou


def val(model, test_data_loader, criterion, render=False):
    model.eval()
    val_loss = 0.0
    iou = 0.0
    for i, data in enumerate(test_data_loader):
        img, tmp, msk = data

        (img, tmp, msk) = (
            img.cuda(), tmp.cuda(), msk.cuda())
        y = model.forward(img, tmp)
        # y_pred = y.permute(0, 2, 3, 1)
        # y_pred = y_pred.contiguous().view(-1, 2)
        # y_true = pseudo_label.long().view(-1)
        
#        y = y.to(torch.float64)
#        msk = msk.to(torch.float64)
  
        if not torch.isfinite(y).all():
            torch.set_printoptions(profile="full")
            print("Error: y is not finite!")
            print(y)
            assert(False)
        
        if not torch.isfinite(msk).all():
            torch.set_printoptions(profile="full")
            print("Error: msk is not finite!")
            print(msk)
            assert(False)

        loss = criterion(y, msk)
        # print('\ntrue',loss.item())
        # print('if all black',criterion(torch.from_numpy(np.zeros((512*512, 2), dtype=np.float32)).cuda(), y_true).item())
        iou += IoU(msk, y)
        val_loss += loss.item()

        if render:
            vis_img = img.detach().cpu().permute(0, 2, 3, 1).numpy()
            vis_img *= [0.229, 0.224, 0.225]
            vis_img += [0.485, 0.456, 0.406]
            vis_img *= 255
            vis = y.permute(0, 2, 3, 1).detach().cpu().numpy()*255
            vis = vis.astype(np.uint8)

            for b in range(batch_size):
                vis_mini = vis[b]
                vis_img_mini = vis_img[b]
                vis_mini = cv.cvtColor(vis_mini, cv.COLOR_GRAY2BGR)
                added_image = cv.addWeighted(
                    vis_img_mini[:, :, ::-1].astype(np.uint8), 0.6, vis_mini, 0.4, 0)
                cv.imwrite('%s/pred-template/pred-%d.png' %
                           (output, i * batch_size + b), added_image)
                # cv.imwrite('%s/pred-monster/true-%d.png' %
                #            (output, i * batch_size + b), vis_img_mini[:, :, ::-1])
                # cv.imwrite('%s/pred-monster/pred-%d.png' %
                #            (output, i * batch_size + b), vis_mini)
                # cv.imwrite('%s/pred-monster/pseudo_label-%d.png' %
                #            (output, i * batch_size + b), pseudo_label.permute(0, 2, 3, 1).detach().cpu().numpy()[b].squeeze(-1)*255)

    val_loss = val_loss / len(test_data_loader)
    iou = iou / len(test_data_loader)
    print('Test loss = %.3f, iou = %.3f' % (val_loss, iou))
    return val_loss, iou


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True) 

#    catagory = ['pacman',
#                'monster-purple', 'monster-red', 'monster-yellow', 'monster-blue']
    # catagory = ['pacman']
    category = ['positive', 'negative']

#    net = TemplateMatching(z_dim=64,
#                           output_channel=512,
#                           pretrain=None, #'pretrained/vgg11_bn.pth',
#                           num_classes=1,
#                           freezed_pretrain=False).cuda()

    model = Network()
    model = model.cuda()

    train_data_set = UIScreen(dir='data', mode='train', category=category)
    train_data_loader = data.DataLoader(
        train_data_set,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
        shuffle=True)

    test_data_set = UIScreen(dir='data', mode='test', category=category)
    test_data_loader = data.DataLoader(
        test_data_set,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True,
        drop_last=True)

#    max_grad_norm = 1
#    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)    
    optimizer = optim.Adam(model.parameters(), lr=0.001,
                           betas=(0.9, 0.999), eps=1e-08)
    n = 0
    for p in model.parameters():
        n += 1

    print(n)

    criterion = BCELoss()
#    criterion = CrossEntropyLoss()

    best_iou = 0.0

    if checkpoint:
        print('Loading checkpoint')
        checkpoint = torch.load(checkpoint)
#        net.load_pretrain(checkpoint['TemplateMatching'], exclude=None, strict=True, log=True)
        model.load_state_dict(checkpoint['DTOID'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_iou = checkpoint['best_iou']
        # best_iou = 0.4

    train(model, train_data_loader, test_data_loader, criterion=criterion,
          optimizer=optimizer, epochs=epochs, best_iou=best_iou)
    val_loss, val_iou = val(model, test_data_loader, criterion, True)
