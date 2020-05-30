import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib as plt

class Matrix():
    classes = ["A008", "A009", "A022", "A023", "A027", "A028", "A029", "A031", "A032", "A033", "A052", "A041", "A043",
               "A045", "A048", "A050", "A051"]

    def create_matrix(self,config, dataloaders_dict, device, model_ft):

        confusion_matrix = torch.zeros(config.num_classes, config.num_classes)
        with torch.no_grad():
            for i, (inputs, classes) in enumerate(dataloaders_dict['val']):
                inputs = inputs.to(device)
                classes = classes.to(device)
                outputs = model_ft(inputs)
                _, preds = torch.max(outputs, 1)
                for t, p in zip(classes.view(-1), preds.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1


        df_cm = pd.DataFrame(confusion_matrix.numpy(), index = [i for i in self.classes],
                          columns = [i for i in self.classes])


        df_cm=df_cm.astype(np.int64)


        df_cm.to_csv(config.save_confusion_mat)


        # plt.figure(figsize = (16,8))
        f, ax = plt.subplots(figsize=(18, 10))
        sns.heatmap(df_cm,annot=True, fmt="d")