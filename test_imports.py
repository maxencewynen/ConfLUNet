import glob
import os
import random
import re
import shutil
import socket
import time
from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from scipy import ndimage, stats
from scipy.ndimage import label, maximum_filter, generate_binary_structure, labeled_comprehension
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.measure import label
from sklearn import metrics
from sklearn.model_selection import train_test_split
import wandb

from monai.config import KeysCollection
from monai.data import CacheDataset, DataLoader, MetaTensor, write_nifti, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import (
    AddChanneld, AsDiscrete, Compose, ConcatItemsd, LoadImaged, Lambdad, MapTransform, MaskIntensityd,
    NormalizeIntensityd, RandAffined, RandCropByPosNegLabeld, RandFlipd, RandRotate90d, RandScaleIntensityd,
    RandShiftIntensityd, RandSpatialCropSamplesd, RandSpatialCropd, RandScaleIntensityd, RandSpatialCropSamplesd,
    Spacingd, ToTensord
)
from monai.utils import set_determinism
assert torch.cuda.is_available()