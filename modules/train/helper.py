import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from modules.utils.utils import log_scalar
import mlflow
import os
from datetime import date




class Helper():

    def __init__(self,config):
        self.config=config


    def set_parameter_requires_grad(self, model):
        if self.config.feature_extract:
            for param in model.parameters():
                param.requires_grad = False

    def train_model(self, model, dataloaders, criterion, optimizer, device, num_epochs=25, is_inception=False):
        since = time.time()
        val_acc_history = []

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        if is_inception and phase == 'train':
                            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                            outputs, aux_outputs = model(inputs)
                            loss1 = criterion(outputs, labels)
                            loss2 = criterion(aux_outputs, labels)
                            loss = loss1 + 0.4 * loss2
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
                if phase == 'train':
                    log_scalar('training_loss', epoch_loss, epoch)
                    log_scalar('training_accuracy', float(epoch_acc), epoch)
                if phase == 'val':
                    log_scalar('val_loss', epoch_loss, epoch)
                    log_scalar('val_accuracy', float(epoch_acc), epoch)
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)

            print()

        time_elapsed = time.time() - since
        mlflow.log_param('Training time', time_elapsed)
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
        mlflow.log_param('Best val Acc', float(best_acc))
        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, val_acc_history

    def initialize_model(self):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        model_ft = None
        input_size = 0

        if self.config.model_name == "resnet":
            """ resnet152
            """
            model_ft = models.resnet152(pretrained=self.config.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.config.feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.config.num_classes)
            input_size = 224

        elif self.config.model_name == "alexnet":
            """ Alexnet
            """
            model_ft = models.alexnet(pretrained= self.config.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.config.feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, self.config.num_classes)
            input_size = 224

        elif self.config.model_name == "vgg16":
            """ VGG16_bn
            """
            model_ft = models.vgg16(pretrained=self.config.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.config.feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs,  self.config.num_classes)
            input_size = 224

        elif  self.config.model_name == "squeezenet":
            """ Squeezenet
            """
            model_ft = models.squeezenet1_0(pretrained= self.config.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.config.feature_extract)
            model_ft.classifier[1] = nn.Conv2d(512,  self.config.num_classes, kernel_size=(1, 1), stride=(1, 1))
            model_ft.num_classes =  self.config.num_classes
            input_size = 224

        elif  self.config.model_name == "densenet":
            """ Densenet201
            """
            model_ft = models.densenet201(pretrained= self.config.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.config.feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, self.config.num_classes)
            input_size = 224

        elif  self.config.model_name == "inception":
            """ Inception v3 
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            model_ft = models.inception_v3(pretrained= self.config.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.config.feature_extract)
            # Handle the auxilary net
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, self.config.num_classes)
            # Handle the primary net
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.config.num_classes)
            input_size = 299

        else:
            print("Invalid model name, exiting...")
            exit()

        return model_ft, input_size

    def dataloader(self,input_size):
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        print("Initializing Datasets and Dataloaders...")

        # Create training and validation datasets
        image_datasets = {x: datasets.ImageFolder(os.path.join(self.config.data_dir, x), data_transforms[x]) for x in
                          ['train', 'val']}
        # Create training and validation dataloaders
        dataloaders_dict = { x: torch.utils.data.DataLoader(image_datasets[x], batch_size= self.config.batch_size, shuffle=True,
            num_workers=4) for x in ['train', 'val']}

        return dataloaders_dict

    def save_model(self, model_ft):
        mlflow.pytorch.log_model(self.config.model_ft, "models")
        mlflow.pytorch.save_model(model_ft, self.config.save_model + str(date.today()) + '/')
        #   mlflow.log_metric('history',hist)
        torch.save(model_ft.state_dict(), self.config.save_model + self.config.model_name + "/" + str(date.today()) + '.pth')

    def mlflow_log(self):
        mlflow.log_param('dataset', self.config.data_dir)
        mlflow.log_param('model name', self.config.model_name)
        mlflow.log_param('number of classes', self.config.num_classes)
        mlflow.log_param('Batch size', self.config.batch_size)
        mlflow.log_param('epochs', self.config.num_epochs)
        mlflow.log_param('feature extracted', self.config.feature_extract)
        mlflow.log_param('pre-trained', self.config.pre_trained)

    def input_size(self):
        if self.config.model_name=='inception':
            size=299
        else:
            size=224
        return size