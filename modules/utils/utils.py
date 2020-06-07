import numpy as np
import mlflow
import mlflow.pytorch

def str2bool(v):
    return v.lower() in ("yes", "true", "1")

def log_scalar(name, value, step):
    """Log a scalar value to both MLflow and TensorBoard"""
    mlflow.log_metric(name, value, step=step)

