import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from models import basenet
from models import dataloader
import utils
from PIL import Image

class ImdbModel():
    def __init__(self, opt):
        super(ImdbModel, self).__init__()
        self.epoch = 0
        self.device = opt['device']
        self.save_path = opt['save_folder']
        self.print_freq = opt['print_freq']
        self.init_lr = opt['optimizer_setting']['lr']
        self.log_writer = SummaryWriter(os.path.join(self.save_path, 'logfile'))
        
        self.set_network(opt)
        self.set_data(opt)
        self.set_optimizer(opt)

    def set_network(self, opt):
        """Define the network"""
        
        # self.network = basenet.ResNet18(num_classes=opt['output_dim']).to(self.device)
        self.network = basenet.ResNet50(n_classes=opt['output_dim'], pretrained=True).to(self.device)
    def forward(self, x):
        out, feature = self.network(x)
        return out, feature

    def set_data(self, opt):
        """Set up the dataloaders"""
        
        data_setting = opt['data_setting']

        # with open(data_setting['train_data_path'], 'rb') as f:
        #     train_array = pickle.load(f)

        # mean = tuple(np.mean(train_array / 255., axis=(0, 1, 2)))
        # std = tuple(np.std(train_array / 255., axis=(0, 1, 2)))
        # normalize = transforms.Normalize(mean=mean, std=std)

        with open(data_setting['train_data_path'], 'rb') as f:
            lines = f.readlines()
            imglist = []
            for line in lines:
                path, _, _ = line.split()
                img = Image.open(path).convert('RGB')
                img = transforms.Resize((224, 224))(img)
                img = transforms.ToTensor()(img)
                imglist.append(img)
            print(img.shape)
            img_list = torch.stack(imglist).numpy()
            print(img_list.shape)
            mean = tuple(np.mean(img_list / 255., axis=(0, 2, 3)))
            std = tuple(np.std(img_list / 255., axis=(0, 2, 3)))
            print("mean:{}".format(mean))
            print("std:{}".format(std))
            normalize = transforms.Normalize(mean=mean, std=std)

        if data_setting['augment']:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #normalize
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                #normalize
            ])

        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            #normalize
        ])

        train_data = dataloader.ImdbDataset(data_setting['train_data_path'], 
                                             transform_train)
        test_male_data = dataloader.ImdbDataset(data_setting['test_male_path'], 
                                                  transform_test)
        test_female_data = dataloader.ImdbDataset(data_setting['test_female_path'], 
                                                 transform_test)

        self.train_loader = torch.utils.data.DataLoader(
                                 train_data, batch_size=opt['batch_size'],
                                 shuffle=True, num_workers=1)
        self.test_male_loader = torch.utils.data.DataLoader(
                                      test_male_data, batch_size=opt['batch_size'],
                                      shuffle=False, num_workers=1)
        self.test_female_loader = torch.utils.data.DataLoader(
                                     test_female_data, batch_size=opt['batch_size'],
                                     shuffle=False, num_workers=1)
    
    def set_optimizer(self, opt):
        optimizer_setting = opt['optimizer_setting']
        self.optimizer = optimizer_setting['optimizer']( 
                            params=self.network.parameters(), 
                            lr=optimizer_setting['lr']
                            )
        
    def _criterion(self, output, target):
        return F.cross_entropy(output, target)
        
    def state_dict(self):
        state_dict = {
            'model': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch
        }  
        return state_dict
    
    def load_state_dict(self, state_dict):
        self.network.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.epoch = state_dict['epoch']

    def log_result(self, name, result, step):
        self.log_writer.add_scalars(name, result, step)

    def adjust_lr(self):
        lr = self.init_lr * (0.1 ** (self.epoch // 50))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _train(self, loader):
        """Train the model for one epoch"""
        
        self.network.train()
        self.adjust_lr()
        
        train_loss = 0
        total = 0
        correct = 0
        for i, (images, targets) in enumerate(loader):
            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs, _ = self.forward(images)
            loss = self._criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            accuracy = correct*100. / total

            train_result = {
                'accuracy': correct*100. / total,
                'loss': loss.item(),
            }
            self.log_result('Train iteration', train_result,
                            len(loader)*self.epoch + i)

            if self.print_freq and (i % self.print_freq == 0):
                print('Training epoch {}: [{}|{}], loss:{}, accuracy:{}'.format(
                    self.epoch, i+1, len(loader), loss.item(), accuracy
                ))
                # print("gt:{}".format(targets))
                # print("pred:{}".format(predicted))
                
        self._train_accuracy = accuracy
        self.epoch += 1

    def _test(self, loader):
        """Test the model performance"""
        
        self.network.eval()

        total = 0
        correct = 0
        test_loss = 0
        output_list = []
        feature_list = []
        predict_list = []
        gt_list = []
        with torch.no_grad():
            for i, (images, targets) in enumerate(loader):
                images, targets = images.to(self.device), targets.to(self.device)
                outputs, features = self.forward(images)
                loss = self._criterion(outputs, targets)
                test_loss += loss.item()

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                gt_list.extend(targets.tolist())
                predict_list.extend(predicted.tolist())
                output_list.append(outputs.cpu().numpy())
                feature_list.append(features.cpu().numpy())

        test_result = {
            'accuracy': correct*100. / total,
            'gt_labels': gt_list,
            'predict_labels': predict_list,
            'outputs': np.vstack(output_list),
            'features': np.vstack(feature_list)
        }
        return test_result

    def train(self):
        self._train(self.train_loader)
        utils.save_state_dict(self.state_dict(), os.path.join(self.save_path, 'ckpt.pth'))

    def test(self):
        # Test and save the result
        self.network.load_state_dict(torch.load(os.path.join(self.save_path, 'ckpt.pth'))['model'])
        test_male_result = self._test(self.test_male_loader)
        test_female_result = self._test(self.test_female_loader)
        utils.save_pkl(test_male_result, os.path.join(self.save_path, 'test_male_result.pkl'))
        utils.save_pkl(test_female_result, os.path.join(self.save_path, 'test_female_result.pkl'))
        
        # Output the classification accuracy on test set
        info = ('Test on male images accuracy: {}\n' 
                'Test on female images accuracy: {}'.format(test_male_result['accuracy'],
                                                          test_female_result['accuracy']))
        utils.write_info(os.path.join(self.save_path, 'test_result.txt'), info)
        
    


            
