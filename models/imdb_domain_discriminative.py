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
from models.imdb_core import ImdbModel
import utils

class ImdbDomainDiscriminative(ImdbModel):
    def __init__(self, opt):
        super(ImdbDomainDiscriminative, self).__init__(opt)
        self.prior_shift_weight = np.array(opt['prior_shift_weight'])

    def set_data(self, opt):
        """Set up the dataloaders"""
        
        data_setting = opt['data_setting']

        # with open(data_setting['train_data_path'], 'rb') as f:
        #     train_array = pickle.load(f)

        # mean = tuple(np.mean(train_array / 255., axis=(0, 1, 2)))
        # std = tuple(np.std(train_array / 255., axis=(0, 1, 2)))
        # normalize = transforms.Normalize(mean=mean, std=std)

        if data_setting['augment']:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # normalize,
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                # normalize,
            ])

        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # normalize,
        ])

        train_data = dataloader.ImdbDatasetFor16cls(data_setting['train_data_path'], 
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
    
            
    def _test(self, loader):
        """Test the model performance"""
        
        self.network.eval()

        total = 0
        correct = 0
        test_loss = 0
        output_list = []
        feature_list = []
        target_list = []
        with torch.no_grad():
            for i, (images, targets) in enumerate(loader):
                images, targets = images.to(self.device), targets.to(self.device)
                outputs, features = self.forward(images)
                loss = self._criterion(outputs, targets)
                test_loss += loss.item()

                output_list.append(outputs)
                feature_list.append(features)
                target_list.append(targets)
                
        outputs = torch.cat(output_list, dim=0)
        features = torch.cat(feature_list, dim=0)
        targets = torch.cat(target_list, dim=0)
        
        accuracy_sum_prob_wo_prior_shift, pred1 = self.compute_accuracy_sum_prob_wo_prior_shift(outputs, targets)
        accuracy_sum_prob_w_prior_shift, pred2 = self.compute_accuracy_sum_prob_w_prior_shift(outputs, targets)
        accuracy_max_prob_w_prior_shift, pred3 = self.compute_accuracy_max_prob_w_prior_shift(outputs, targets)
        
        test_result = {
            'accuracy_sum_prob_wo_prior_shift': accuracy_sum_prob_wo_prior_shift,
            'accuracy_sum_prob_w_prior_shift': accuracy_sum_prob_w_prior_shift,
            'accuracy_max_prob_w_prior_shift': accuracy_max_prob_w_prior_shift,
            'prediction_sum_prob_wo_prior_shift': pred1,
            'prediction_sum_prob_w_prior_shift': pred2,
            'prediction_max_prob_w_prior_shift': pred3,
            'targets': target_list,
            'outputs': outputs.cpu().numpy(),
            'features': features.cpu().numpy()
        }
        return test_result
    
    def compute_accuracy_sum_prob_wo_prior_shift(self, outputs, targets):
        probs = F.softmax(outputs, dim=1).cpu().numpy()
        targets = targets.cpu().numpy()
    
        predictions = np.argmax(probs[:, :8] + probs[:, 8:], axis=1)
        accuracy = (predictions == targets).mean() * 100.
        return accuracy, predictions
    
    def compute_accuracy_sum_prob_w_prior_shift(self, outputs, targets):
        probs = F.softmax(outputs, dim=1).cpu().numpy() * self.prior_shift_weight
        targets = targets.cpu().numpy()
        
        predictions = np.argmax(probs[:, :8] + probs[:, 8:], axis=1)
        accuracy = (predictions == targets).mean() * 100.
        return accuracy, predictions
    
    def compute_accuracy_max_prob_w_prior_shift(self, outputs, targets):
        probs = F.softmax(outputs, dim=1).cpu().numpy() * self.prior_shift_weight
        targets = targets.cpu().numpy()
        
        predictions = np.argmax(np.stack((probs[:, :8], probs[:, 8:])).max(axis=0), axis=1)
        accuracy = (predictions == targets).mean() * 100.
        return accuracy, predictions

    def test(self):
        # Test and save the result
        state_dict = torch.load(os.path.join(self.save_path, 'ckpt.pth'))
        self.load_state_dict(state_dict)
        test_male_result = self._test(self.test_male_loader)
        test_female_result = self._test(self.test_female_loader)
        utils.save_pkl(test_male_result, os.path.join(self.save_path, 'test_male_result.pkl'))
        utils.save_pkl(test_female_result, os.path.join(self.save_path, 'test_female_result.pkl'))
        
        # Output the classification accuracy on test set for different inference
        # methods
        info = ('Test on male images accuracy sum prob without prior shift: {}\n' 
                'Test on male images accuracy sum prob with prior shift: {}\n' 
                'Test on male images accuracy max prob with prior shift: {}\n' 
                'Test on female images accuracy sum prob without prior shift: {}\n'
                'Test on female images accuracy sum prob with prior shift: {}\n'
                'Test on female images accuracy max prob with prior shift: {}\n'
                .format(test_male_result['accuracy_sum_prob_wo_prior_shift'],
                        test_male_result['accuracy_sum_prob_w_prior_shift'],
                        test_male_result['accuracy_max_prob_w_prior_shift'],
                        test_female_result['accuracy_sum_prob_wo_prior_shift'],
                        test_female_result['accuracy_sum_prob_w_prior_shift'],
                        test_female_result['accuracy_max_prob_w_prior_shift']))
        utils.write_info(os.path.join(self.save_path, 'test_result.txt'), info)
    
    