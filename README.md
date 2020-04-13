# Image-classification-PyTorch-MLflow

The repository contain code for image classification using `PyTorch`. I have also used `MLflow` to track the experiments. The code is a modified version of the PyTorch [fine-tune](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html) tutorial.
This code has features like `MLflow`, Confustion matrix generation, and Model saving.

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

## Training steps 

- set the parameters and paths in the code according to your need 
- after finshing the training it will create confusion matrix 
- the notebook will also save trained model with `pytorch` and `mlflow`  
    
## MLflow 

`MLflow` helps in tracking experiments, packaging code into reproducible runs, and sharing and deploying models. You can 
find more information about `MLflow` [Here](https://mlflow.org/).
I have used `MLflow` to track my experiments and save parameters used for a particular training. We tracked 7 parameters in
this case which can be seen later. 

- Install MLflow from PyPI via ```pip install mlflow```
- The MLflow Tracking UI will show runs logged in `./mlruns` at [http://localhost:5000](http://localhost:5000). Start it with:
`mlflow ui`

## TODO:
- add Docker file 
- add Requirements
- add data processing code
- add dataset creation code
