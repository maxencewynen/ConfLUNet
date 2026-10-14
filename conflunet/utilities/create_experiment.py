import os
import shutil
import pandas as pd

from utils import generate_dataset_json, synth_strip

class SEQUENCES:
    MP2RAGE = 'MP2RAGE'
    MPRAGE = 'MPRAGE'
    FLAIR = 'FLAIR'
    T1W = 'T1W'

class FOLDERS:
    IMAGES_TRAIN = 'imagesTr'
    IMAGES_TEST = 'imagesTs'
    LABELS_TRAIN = 'labelsTr'
    LABELS_TEST = 'labelsTs'
    DATASET = 'Dataset'

def create_nnUNet_folder_struct(main_dataset_folder:str):
    dataset_name = os.path.split(main_dataset_folder)[1]
    assert dataset_name.startswith(FOLDERS.DATASET)
    os.makedirs(main_dataset_folder, exist_ok=True)
    for folder in [FOLDERS.IMAGES_TRAIN, FOLDERS.IMAGES_TEST, FOLDERS.LABELS_TRAIN, FOLDERS.LABELS_TEST]:
        os.makedirs(os.path.join(main_dataset_folder, folder), exist_ok=True)

if __name__ == '__main__':
    import glob
    main_img_bids_path = '/media/chuv2/MultipleSclerosis/nataliia/CL-Mock-BIDS/'
    main_img_split_path = '/media/chuv2/MultipleSclerosis/nataliia/MSSeg/data/split4/'
    main_lbl_path = '/media/secondary/conflunet_missing_data/ConfluentLesions_INsIDER/'
    splits_csv = '/media/secondary/conflunet_missing_data/ConfluentLesions_INsIDER/train_test_splits_bins-6_seed-42.csv'
    skull_stripping = True
    dataset_name = 'Dataset401_MultiModality_INsIDER_Instances'
    new_subjects_path = '/media/secondary/conflunet_missing_data/Missing_INsIDER_Lausanne/' # These correspong when the missing ones which are not in the BIDS (yet)
    subjects_missing = []
    # some images are not present in the split. The labels are notin the BIDS because is really horrible to follow the stadard for it
    splits_df = pd.read_csv(splits_csv)

    dataset_path = os.path.join(main_img_split_path, dataset_name)
    subject_missing_path = os.path.join(main_img_split_path, dataset_name,'subjects_missing.txt')
    create_nnUNet_folder_struct(dataset_path)
    nnunet_img_train = os.path.join(dataset_path, FOLDERS.IMAGES_TRAIN)
    nnunet_img_test = os.path.join(dataset_path, FOLDERS.IMAGES_TEST)
    nnunet_lbl_train = os.path.join(dataset_path, FOLDERS.LABELS_TRAIN)
    nnunet_lbl_test = os.path.join(dataset_path, FOLDERS.LABELS_TEST)
    
    n_train = 0
    #for row in splits_df.sample(n=3).itertuples():
    for row in splits_df.itertuples():
        subject_id = row.subject_id.replace("P", 'sub-')
        tp = row.TP.split('TP')[1]
        lbl_pth = os.path.join(main_lbl_path, row.filename)
        img_dir = os.path.join(
            main_img_bids_path, 
            'merged_insider', 
            subject_id, 
            'ses-0' + tp, 
            'anat', 
            '*.nii.gz'
        )
        print(img_dir) 
        img_pth = glob.glob(img_dir)
        subject_missing = False
        if len(img_pth) == 0:
            img_dir = os.path.join(new_subjects_path, 'INsIDER_'+row.subject_id, '*.nii.gz')
            img_pth = glob.glob(img_dir)
        #print(main_img_bids_path)
        if len(img_pth) == 0:
            subject_missing = True
            subjects_missing.append(subject_id)


        if not os.path.exists(lbl_pth):
            print('Label',lbl_pth, 'not found')

        for img in img_pth:
            print('Processing',img)
            mp2rage = True if SEQUENCES.MP2RAGE.lower() in img.lower() else False
            nnunet_suffix = '0001' if mp2rage else '0000' # FLAIR images with 0000 suffix 
            nnunet_label_filename = f"{subject_id}_TP-0{tp}.nii.gz"
            nnunet_image_filename = f"{subject_id}_TP-0{tp}_{nnunet_suffix}.nii.gz"
            save_img_pth_base = nnunet_img_train if row.train else nnunet_img_test
            save_img_pth = os.path.join(save_img_pth_base, nnunet_image_filename)
            save_lbl_pth = os.path.join(nnunet_lbl_train, f"{nnunet_label_filename}.nii.gz") if row.train else os.path.join(nnunet_lbl_test, f"{nnunet_label_filename}.nii.gz")
            if not os.path.exists(save_img_pth):
                print('Skull Stripping', img)
                if skull_stripping: # not sure how the skull stripping was done before so for homogenization I re-do it
                    synth_strip(os.path.dirname(img), os.path.basename(img),save_img_pth_base, nnunet_image_filename)
                else:
                    shutil.copyfile(img, save_img_pth)
                shutil.copyfile(lbl_pth, save_lbl_pth)    
        n_train = n_train +1 if row.train and not subject_missing else n_train 

    
    ## generate json 
    description = f"This dataset takes skull stripped FLAIR (_0000) and MP2RAGE (_0001) images from INsIDER employing instances segmentation"
    generate_dataset_json(dataset_path, 
                          channel_names={0:'FLAIR', 1:'MP2RAGE'},
                          labels={'background':0, "Instances": 1}, 
                          num_training_cases=n_train, 
                          file_ending='.nii.gz', 
                          dataset_name=dataset_name, 
                          description=description)
    
    # Write subjects_missing to a text file
    with open(subject_missing_path, 'w') as f:
        for subject in subjects_missing:
            f.write(f"{subject}\n")
       
