import numpy as np
import os
import nibabel as nib
from tqdm import tqdm

# global variables
brats_dir = 'BRATS2018/'
cls_dir = os.path.join(brats_dir, 'CLS')
train_dir = os.path.join(cls_dir, 'train')
val_dir = os.path.join(cls_dir, 'val')
train_list_txt = os.path.join(cls_dir, 'train.txt')
val_list_txt = os.path.join(cls_dir, 'val.txt')
# global variables

if not os.path.exists(cls_dir):
    os.makedirs(cls_dir)

if not os.path.exists(train_dir):
    os.makedirs(train_dir)

if not os.path.exists(val_dir):
    os.makedirs(val_dir)

if not os.path.exists(train_list_txt):
    os.mknod(train_list_txt)

if not os.path.exists(val_list_txt):
    os.mknod(val_list_txt)

def _process_single_case_3D(case_dir, dataset_type='train'):
    case_name = case_dir[14:]
    label = case_dir[10:13]
    t1 = nib.load(os.path.join(case_dir, case_name + '_t1.nii.gz')).get_data()
    t1ce = nib.load(os.path.join(case_dir, case_name + '_t1ce.nii.gz')).get_data()
    t2 = nib.load(os.path.join(case_dir, case_name + '_t2.nii.gz')).get_data()
    flair = nib.load(os.path.join(case_dir, case_name + '_flair.nii.gz')).get_data()

    assert t1.shape == (240, 240, 155)
    assert t1ce.shape == (240, 240, 155)
    assert t2.shape == (240, 240, 155)
    assert flair.shape == (240, 240, 155)

    if dataset_type == 'train':
        data_dir = train_dir
        data_list = train_list_txt
    else:
        data_dir = val_dir
        data_list = val_list_txt


    sc = np.array([t1, t1ce, t2, flair])
    assert sc.shape == (4, 240, 240, 155)

    if label == 'LGG':
        grade = np.array([0])
    else:
        grade = np.array([1])

    np.save(os.path.join(data_dir, case_name + '_scan'), sc)
    np.save(os.path.join(data_dir, case_name + '_grade'), grade)

    with open(data_list, 'a') as l:
        l.writelines(case_name + '\n')

def process_training_3D_case(case_dir):
    _process_single_case_3D(case_dir, dataset_type='train')

def process_val_3D_case(case_dir):
    _process_single_case_3D(case_dir, dataset_type='val')

if __name__ == "__main__":
    LGG_list = sorted(os.listdir(os.path.join(brats_dir, 'LGG')))[:20]
    LGG_list = list(map(lambda x: 'LGG/' + x, LGG_list))
    HGG_list = sorted(os.listdir(os.path.join(brats_dir, 'HGG')))[:40]
    HGG_list = list(map(lambda x: 'HGG/' + x, HGG_list))

    training_rate = 0.85
    LGG_training_num = int(len(LGG_list) * training_rate)
    HGG_training_num = int(len(HGG_list) * training_rate)

    LGG_training_list = LGG_list[:LGG_training_num]
    HGG_training_list = HGG_list[:HGG_training_num]
    LGG_val_list = LGG_list[LGG_training_num:]
    HGG_val_list = HGG_list[HGG_training_num:]

    for c in tqdm(LGG_training_list + HGG_training_list):
        process_training_3D_case(os.path.join(brats_dir, c))
    
    for c in tqdm(LGG_val_list + HGG_val_list):
        process_val_3D_case(os.path.join(brats_dir, c))

    num_train_cases = sum([1 for line in open(train_list_txt)])
    num_val_cases = sum([1 for line in open(val_list_txt)])
    assert(len(os.listdir(train_dir)) == 2 * num_train_cases)
    assert(len(os.listdir(val_dir)) == 2 * num_val_cases)
