""" 
Implemented inspired by the work of @ultralytics
https://docs.ultralytics.com/pt/guides/kfold-cross-validation/#k-fold-dataset-split
"""
import datetime
import shutil
from pathlib import Path
from collections import Counter
import os
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from ultralytics import YOLO

class KFoldCrossValidation:
    def __init__(
                    self, 
                    dataset_path, 
                    yaml_file, 
                    ksplit=5, 
                    save_csv=False,
                    extensions=['.jpg', '.jpeg', '.png'],
                    train=False,
                    data='yolov8s-p2.yaml',
                    **kwargs
                ):
        assert os.path.exists(dataset_path), "Dataset path does not exist."
        self.dataset_path = Path(dataset_path)
        self.labels = sorted(self.dataset_path.rglob("*labels/*.txt"))
        
        assert os.path.exists(yaml_file), "Please provide a YAML file for the dataset."
        self.yaml_file = yaml_file
        
        assert ksplit > 1, "Please provide a value greater than 1 for ksplit."
        self.ksplit = ksplit
        self.classes = self._get_classes()
        self.cls_idx = sorted(self.classes.keys())
        self.labels_df = self._get_labels_df()
        self.kfolds = self._get_kfolds()
        self.folds = [f'split_{n}' for n in range(1, self.ksplit + 1)]
        self.folds_df = self._get_folds_df()
        self.fold_lbl_distrb = self._get_fold_lbl_distrb()
        self.supported_extensions = extensions
        self.images = self._get_images()
        self.save_path = self._get_save_path()
        self.ds_yamls = self._get_ds_yamls()
        self.save_csv = save_csv
        self.train = train
        self.results = {}
        self.data = data
        """ 
        kwargs:
        - batch: int
        - epochs: int
        - patience: int
        - weights: str (optional)
        - optimizer:str
        - lr0: float
        - lrf: float
        - momentum: float
        - weight_decay: float
        - label_smoothing: float
        - dropout: float
        - val: bool
        - imgsz: int
        - device: str
        - project: str
        - name: str
        - exist_ok: bool
        - single_cls: bool
        """
        self.kwargs = kwargs
        self.weights = kwargs.get('weights', None)
        
    def __call__(self):
        self._copy_images_and_labels()
        if self.save_csv:
            self._save_csv()
        if self.train:
            self._train()
    
    def _load_model(self):
      if self.weights is not None:
        return YOLO(self.weights)
      else:
        return YOLO(self.data)
    
    def _train(self):
      model =  self._load_model()
      for k in range(self.ksplit):
        print(f"Training on split_{k+1}")
        dataset_yaml = self.ds_yamls[k]
        model.train(dataset_yaml, kwargs=self.kwargs)
        self.results[k] = model.metrics
    
    def _get_classes(self):
        with open(self.yaml_file, 'r', encoding="utf8") as y:
            return yaml.safe_load(y)['names']

    def _get_labels_df(self):
        indx = [l.stem for l in self.labels]
        labels_df = pd.DataFrame([], columns=self.cls_idx, index=indx)
        for label in self.labels:
            print('label', label)
            lbl_counter = Counter()
            with open(label,'r') as lf:
                lines = lf.readlines()
            for l in lines:
                lbl_counter[l.split(' ')[0]] += 1
            labels_df.loc[label.stem] = lbl_counter
        return labels_df.fillna(0.0)

    def _get_kfolds(self):
        kf = KFold(n_splits=self.ksplit, shuffle=True, random_state=20)
        return list(kf.split(self.labels_df))
    
    def _get_folds_df(self):
        folds_df = pd.DataFrame(index=self.labels_df.index, columns=self.folds)
        for idx, (train, val) in enumerate(self.kfolds, start=1):
            folds_df[f'split_{idx}'].loc[self.labels_df.iloc[train].index] = 'train'
            folds_df[f'split_{idx}'].loc[self.labels_df.iloc[val].index] = 'val'
        return folds_df
    
    def _get_fold_lbl_distrb(self):
        fold_lbl_distrb = pd.DataFrame(index=self.folds, columns=self.cls_idx)
        for n, (train_indices, val_indices) in enumerate(self.kfolds, start=1):
            train_totals = self.labels_df.iloc[train_indices].sum()
            val_totals = self.labels_df.iloc[val_indices].sum()
            ratio = val_totals / (train_totals + 1E-7)
            fold_lbl_distrb.loc[f'split_{n}'] = ratio
        return fold_lbl_distrb
    
    def _get_images(self):
        images = []
        for ext in self.supported_extensions:
            images.extend(sorted((self.dataset_path / 'images').rglob(f"*{ext}")))
        return images
    
    def _get_save_path(self):
        save_path = Path(self.dataset_path / f'{datetime.date.today().isoformat()}_{self.ksplit}-Fold_Cross-val')
        save_path.mkdir(parents=True, exist_ok=True)
        return save_path

    def _get_ds_yamls(self):
        ds_yamls = []
        for split in self.folds_df.columns:
            split_dir = self.save_path / split
            split_dir.mkdir(parents=True, exist_ok=True)
            (split_dir / 'train' / 'images').mkdir(parents=True, exist_ok=True)
            (split_dir / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
            (split_dir / 'val' / 'images').mkdir(parents=True, exist_ok=True)
            (split_dir / 'val' / 'labels').mkdir(parents=True, exist_ok=True)
            dataset_yaml = split_dir / f'{split}_dataset.yaml'
            ds_yamls.append(dataset_yaml)
            with open(dataset_yaml, 'w') as ds_y:
                yaml.safe_dump({
                    'path': split_dir.as_posix(),
                    'train': 'train',
                    'val': 'val',
                    'names': self.classes
                }, ds_y)
        return ds_yamls
    
    def _copy_images_and_labels(self):
        for image, label in zip(self.images, self.labels):
            for split, k_split in self.folds_df.loc[image.stem].items():
                img_to_path = self.save_path / split / k_split / 'images'
                lbl_to_path = self.save_path / split / k_split / 'labels'
                shutil.copy(image, img_to_path / image.name)
                shutil.copy(label, lbl_to_path / label.name)

    def _save_csv(self):
        self.folds_df.to_csv(self.save_path / "kfold_datasplit.csv")
        self.fold_lbl_distrb.to_csv(self.save_path / "kfold_label_distribution.csv")


