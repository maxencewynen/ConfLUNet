import pandas as pd
from create_splits import flexible_stratified_split

from batchgenerators.utilities.file_and_folder_operations import save_json

def create_nnunet_splits_json(folds_df, save_path, folds_col='cv_fold'):
    pass

if __name__ == "__main__":
    import os
    import glob

    random_seed = 42
    test_size = 0.2
    n_bins = 5 
    instances_data_path = '/media/secondary/conflunet_missing_data/ConfluentLesions_INsIDER/train_test_splits_bins-6_seed-42.csv'
    dataset_dir  = '/media/chuv2/MultipleSclerosis/nataliia/MSSeg/data/split4/Dataset401_MultiModality_INsIDER_Instances/'
    folds = 5

    instances_data = pd.read_csv(instances_data_path)
    developement_df = instances_data[instances_data['train'] == True]
    folds_df = flexible_stratified_split(developement_df, stratify_cols=['N_lesions', 'Total_volume'], test_size=test_size, n_bins=n_bins, random_state=random_seed, folds=folds) 
    folds_df.to_csv(instances_data_path.replace('.csv', '_folds.csv'), index=False)

    masks = glob.glob(os.path.join(dataset_dir,'labelsTr/*.nii.gz'))
    masks_filenames = [os.path.basename(mask).split('.nii')[0] for mask in masks]
    folds_lists = [{'train':masks_filenames.copy(), 'val':[]} for _ in range(folds)]
    for mask in masks:
        mask_filename = os.path.basename(mask).split('.nii')[0]
        print(mask_filename)
        subject_id, tp = mask_filename.split('_')
        subject_id = subject_id.replace('sub-','P')
        cv_fold = folds_df[(folds_df['subject_id'] == subject_id)]['cv_fold'].values[0] 
        folds_lists[cv_fold]['train'].remove(mask_filename)
        folds_lists[cv_fold]['val'].append(mask_filename)

    save_json(folds_lists, os.path.join(dataset_dir,'splits_final.json'))
