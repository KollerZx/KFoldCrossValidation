from KFoldCrossValidation import KFoldCrossValidation

if __name__ == '__main__':
    dataset_path = '/media/DADOS/datasets_aceno/custom_train/custom-dataset/union/'
    yaml_file = '/media/DADOS/datasets_aceno/custom_train/data.yaml'
    kfold = KFoldCrossValidation(dataset_path, yaml_file, ksplit=2, save_csv=True)
    kfold()