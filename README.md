# Image-classification-PyTorch-MLflow

The repository contain code for image classification using PyTorch. I have also used MLflow to track the experiments. The code is a modified version of the PyTorch [fine-tune](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html) tutorial.
This code has features like MLflow, Confustion matrix generation, and Model saving.

This repository only contain the code for training the models. Data pre-processing and dataset creation code is not present in this repository but can be added in future on request. 

## Dataset folder structure should be:
```
root
└── dataset
        ├── Train
        │   ├── Class 1
        │   │   ├── Sample 1
        │   │   ├── .........
        │   │   └── Sample N
        │   ├── ........
        │   └── Class N
        │       ├── Sample 1
        │       ├── .........
        │       └── Sample N
        └── Val
            ├── Class 1
            │   ├── Sample 1
            │   ├── .........
            │   └── Sample N
            ├── ........
            └── Class N
                ├── Sample 1
                ├── .........
                └── Sample N
```

## TODO:
- add Docker file 
- add Requirements
- add data processing code
- add dataset creation code
