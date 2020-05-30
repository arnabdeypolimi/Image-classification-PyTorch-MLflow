from modules.train.helper import Helper
from modules.config.config_manager import ConfigurationManager
import torch
import torch.optim as optim
import torch.nn as nn
import mlflow
import mlflow.pytorch


class Train():


    def run_training(self, config, dataloaders_dict, device):

        model_ft, input_size = Helper.initialize_model(config.model_name, config.num_classes, config.feature_extract,
                                                       use_pretrained=config.pre_trained)

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
        if config.feature_extract:
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
            Helper.mlflow_log(config)

            # Setup the loss fxn
            criterion = nn.CrossEntropyLoss()

            # Train and evaluate
            model_ft, hist = Helper.train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=config.num_epochs,
                                         is_inception=(config.model_name == "inception"))
            print("Training complete")

        Helper.save_model(model_ft, config.model_name, config.save_model)

        print("Model saved")

        return model_ft





