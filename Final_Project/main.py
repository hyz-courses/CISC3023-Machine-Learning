# Basics
import os
import copy
import time
from itertools import product
from typing import List, Callable, Any, Union

# Pre-processing
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd

# Model Training
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold


def get_labels(data_type: str) -> np.ndarray:
    """
    Get ground truth labels from .csv file.
    :param data_type: Type of data: Training or Testing.
    """
    df = pd.read_csv(f"./Wound/{data_type}/myData.csv", delimiter=";")
    return df.to_numpy()


def get_images(data_type: str,
               image_names: np.ndarray,
               augmentation: Union[Callable[[np.ndarray, Any], np.ndarray], None] = None,
               flatten=True,
               **kwargs) -> np.ndarray:
    """
    Get the images from directory.
    :param data_type: Type of data: Training or Testing.
    :param image_names: Names of images from ground truth.
    :param augmentation: Augmentation function.
    :param flatten: Whether to flatten the images.
    :param kwargs: Other arguments to pass to augmentation function.
    """
    images = []
    for i_name in image_names:
        img = Image.open(os.path.join(f"./Wound/{data_type}/", i_name))
        img = img.resize((32, 32), Image.BICUBIC)
        img = np.array(img)
        if augmentation:
            img = augmentation(img, **kwargs)
        images.append(img.flatten() if flatten else img)

    images = np.array(images)

    return images


def add_black_edge(img: np.array, w: int = 4) -> np.array:
    """
    Image augmentation. Add an inner black edge to an image.
    :param img: Image to be processed.
    :param w: Width of the edge.
    """
    if w > min(img.shape[0:2]) // 2:
        raise ValueError("Width of the edge must be smaller than half of the shorter side of an image.")

    new_img = np.zeros_like(img)
    new_img[w:-w, w:-w, :] = img[w:-w, w:-w, :]
    return new_img


def stretch(img: np.ndarray, f: List[float]) -> np.ndarray:
    """
    Image augmentation. Stretch an image on the width and height side.
    :param img: Image to be augmented.
    :param f: Factor tuple. Width and Height.
    """
    fw, fh = f
    if fw < 1 or fh < 1:
        raise ValueError("Width and height factors should be greater than or equal to 1.")

    # New widths
    new_width = int(img.shape[1] * fw)
    new_height = int(img.shape[0] * fh)

    # Adjust image
    img_pil = Image.fromarray(img)
    img_resized = img_pil.resize((new_width, new_height), Image.BICUBIC)

    # Crop regions
    # Keep 32x32 size
    left = (new_width - 32) // 2
    top = (new_height - 32) // 2
    right = left + 32
    bottom = top + 32

    # Crop image
    img_cropped = img_resized.crop((left, top, right, bottom))

    # Convert to numpy array
    img_stretched = np.array(img_cropped)

    return img_stretched


def train(ModelInstance, desc: str = "DESC", n_fold: int = 3, save: bool = False):
    """
    Train the model. Output would be of shape:
    {
        "x": {"MSE": 114514, "model": ModelInstance},
        "y": {"MSE": 114514, "model": ModelInstance},
        "w": {"MSE": 114514, "model": ModelInstance},
        "h": {"MSE": 114514, "model": ModelInstance},
    }
    :param ModelInstance: Instance of a model class.
    :param desc: Description of the saved file.
    :param n_fold: Number of folds.
    :param save: Whether to save the experiment object.
    """
    model_name = ModelInstance.__class__.__name__
    semantic_y = ["File Name", "x", "y", "w", "h"]

    # Data Augmentation
    # Change some useless information
    Y_ori = get_labels(data_type="Training")
    X_ori = get_images(data_type="Training", image_names=Y_ori[:, 0])

    # Add black edge
    Y_be = get_labels(data_type="Training")
    X_be = get_images(data_type="Training", image_names=Y_be[:, 0], augmentation=add_black_edge, w=4)

    # Stretch height
    Y_sh = get_labels(data_type="Training")
    X_sh = get_images(data_type="Training", image_names=Y_sh[:, 0], augmentation=stretch, f=[1.0, 1.05])
    Y_sh[:, 4] *= 1.05

    # Stretch Width
    Y_sw = get_labels(data_type="Training")
    X_sw = get_images(data_type="Training", image_names=Y_sw[:, 0], augmentation=stretch, f=[1.05, 1.0])
    Y_sw[:, 3] *= 1.05

    X = np.concatenate((X_ori, X_be, X_sh, X_sw))
    Y = np.concatenate((Y_ori, Y_be, Y_sh, Y_sw))

    # Print Model configurations
    print(f"Training model {model_name}. Description: {desc}\nStarted at: {time.time()}")
    # Predict all for x, y, w, h
    exp = {}
    for i in range(1, Y.shape[1]):
        # Totally 4 labels to predict.
        # Select one of them.
        y = Y[:, i]

        # Split original data into 3 parts
        # to perform cross-validation
        kf = KFold(n_splits=n_fold, shuffle=True, random_state=1919810)
        splits = kf.split(X)

        # Record MSE of each fold.
        # Keep the model with the smallest MSE
        mse_scores = []
        cur_best_model = None
        cur_smallest_MSE = np.inf
        for train_index, val_index in splits:
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            model = copy.deepcopy(ModelInstance)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            mse_scores.append(mse)

            if cur_smallest_MSE > mse_scores[-1]:
                cur_best_model = copy.deepcopy(model)

        exp[semantic_y[i]] = {
            "Best MSE": cur_smallest_MSE,
            "Best Fold": np.argmin(mse_scores),
            "Avg MSE": np.mean(mse_scores),
            "model": copy.deepcopy(cur_best_model)
        }
        del cur_best_model

        print(f"{semantic_y[i]} - Avg MSE={np.mean(mse_scores):.4f}, "
              f"Best MSE={np.min(mse_scores):.4f} at index {np.argmin(mse_scores)}")

    # Save models
    print(f"Ended at {time.time()}")
    if save:
        time_str = str(time.time()).replace(".", "")
        pickle.dump(exp, open(f"./save_models/{model_name}_{desc}_{time_str}.sav", "wb"))
    return exp


if __name__ == "__main__":
    """
    Hyper parameters for all trainings:
    - n_fold: Number of folds for cross-validation.
    
    Hyper parameters for Random Forest Regressor:
    - n_estimators: Number of estimators.
    - bootstrap: Bootstrap or not.
    - max_depth: Maximum Depth of the tree.
    - min_samples_split: Minimum sample number that allows a leaf to be split again.
    - min_samples_leaf: Minimum sample number a leaf requires.
    """
    rfr_nest = [10, 20, 30, 40, 50]
    rfr_maxd = [11, 13, 15, 17, 19]
    rfr_mins = [2, 4, 6, 8, 10]
    rfr_minl = [6, 8, 10, 12, 14]

    rfr_grid0 = product(rfr_nest, rfr_maxd)
    rfr_grid1 = product(rfr_mins, rfr_minl)

    reg_nest_10 = RandomForestRegressor(n_estimators=10)
    reg_nest_100 = RandomForestRegressor(n_estimators=100)
    train(reg_nest_10, desc="nest-10")
    train(reg_nest_100, desc="nest-100")
