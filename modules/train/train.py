from modules.train.helper import Helper
from modules.config.config_manager import ConfigurationManager
import torch
import torch.optim as optim
import torch.nn as nn
import mlflow
import mlflow.pytorch


class Train():

    def __init__(self,config):
        self.config=config
        self.helper=Helper(config)

    def run_training(self, dataloaders_dict, device):

        model_ft, input_size = self.helper.initialize_model()

        # Print the model we just instantiated
        print(model_ft)


        # Send the model to GPU
        model_ft = model_ft.to(device)

        # Gather the parameters to be optimized/updated in this run. If we are
        #  finetuning we will be updating all parameters. However, if we are
        #  doing feature extract method, we will only update the parameters
        #  that we have just initialized, i.e. the parameters with requires_grad
        #  is True.
        params_to_update = model_ft.parameters()
        print("Params to learn:")
        if self.config.feature_extract:
            params_to_update = []
            for name, param in model_ft.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t", name)
        else:
            for name, param in model_ft.named_parameters():
                if param.requires_grad == True:
                    print("\t", name)

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

        with mlflow.start_run() as run:
            self.helper.mlflow_log()

            # Setup the loss fxn
            criterion = nn.CrossEntropyLoss()

            # Train and evaluate
            model_ft, hist = self.helper.train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, device=device,
                                         is_inception=(self.config.model_name == "inception"))
            print("Training complete")

        self.helper.save_model(model_ft)

        print("Model saved")

        return model_ft





