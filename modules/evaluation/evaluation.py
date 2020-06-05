import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib as plt

class Evaluation():
    classes = []

    def __init__(self, config):
        self.config=config
        self.classes=self.config.classes
        if len(self.classes)>0:
            self.classes=config.classes
        else:
            self.classes=list(range(self.config.num_classes))

    def generate_confusion_matrix(self, dataloaders_dict, device, model_ft):
        confusion_matrix = torch.zeros(self.config.num_classes, self.config.num_classes)
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

        df_cm.to_csv(self.config.save_confusion_mat)

        # plt.figure(figsize = (16,8))
        # f, ax = plt.subplots(figsize=(18, 10))
        # sns.heatmap(df_cm,annot=True, fmt="d")

        return df_cm