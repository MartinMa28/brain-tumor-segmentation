import numpy as np
import os
import nibabel as nib
from concurrent.futures import ThreadPoolExecutor
import sys

# global variables
dataset_type = sys.argv[1]
brats_base_dir = './BRATS2018/{}'.format(dataset_type)
seg_base_dir = './BRATS2018/SEG_{}'.format(dataset_type)
train_dir = os.path.join(seg_base_dir, 'train')
val_dir = os.path.join(seg_base_dir, 'val')
train_list_txt = os.path.join(seg_base_dir, 'train.txt')
val_list_txt = os.path.join(seg_base_dir, 'val.txt')

if not os.path.exists(seg_base_dir):
    os.makedirs(seg_base_dir)

if not os.path.exists(train_dir):
    os.makedirs(train_dir)

if not os.path.exists(val_dir):
    os.makedirs(val_dir)

if not os.path.exists(train_list_txt):
    os.mknod(train_list_txt)

if not os.path.exists(val_list_txt):
    os.mknod(val_list_txt)

def _process_single_case(case_name, dataset_type='train'):
    case_path = os.path.join(brats_base_dir, case_name)
    
    t1 = nib.load(os.path.join(case_path, case_name + '_t1.nii.gz'))
    t1ce = nib.load(os.path.join(case_path, case_name + '_t1ce.nii.gz'))
    t2 = nib.load(os.path.join(case_path, case_name + '_t2.nii.gz'))
    flair = nib.load(os.path.join(case_path, case_name + '_flair.nii.gz'))
    seg = nib.load(os.path.join(case_path, case_name + '_seg.nii.gz'))
    
    # convert nib to numpy ndarray
    t1 = t1.get_data()
    t1ce = t1ce.get_data()
    t2 = t2.get_data()
    flair = flair.get_data()
    seg = seg.get_data()

    assert t1.shape == (240, 240, 155)
    assert t1ce.shape == (240, 240, 155)
    assert t2.shape == (240, 240, 155)
    assert flair.shape == (240, 240, 155)
    assert seg.shape == (240, 240, 155)

    if dataset_type == 'train':
        data_dir = train_dir
        data_list = train_list_txt
    else:
        data_dir = val_dir
        data_list = val_list_txt

    # only keep the slices that are most likely to have tumor inside
    for i in range(20, 155):
        print('processing {}_{}'.format(case_name, i))
        if os.path.exists(os.path.join(data_dir, case_name + '_{}_scan'.format(i)) + '.npy'):
            print('skip {}_{}'.format(case_name, i))
            continue
        
        str_i = str(i).zfill(3)
        # sc: brain scan in CxHxW
        # 4 channels are t1, t1ce, t2, and flair respectively
        sc = np.array([t1[:, :, i], t1ce[:, :, i], t2[:, :, i], flair[:, :, i]])
        
        np.save(os.path.join(data_dir, case_name + '_{}_scan'.format(str_i)), sc)

        wt = (seg[:, :, i] > 0).astype(np.uint8)                                     # whole tumor
        et = (seg[:, :, i] == 4).astype(np.uint8)                                    # enhancing tumor
        tc = np.logical_or(seg[:, :, i] == 1, seg[:, :, i] == 4).astype(np.uint8)    # tumor core
        seg_i = seg[:, :, i].astype(np.uint8)
        seg_i = seg_i + (seg_i == 4).astype(np.uint8) * (3 * np.ones((seg_i.shape[0], seg_i.shape[1])) - seg_i)

        np.save(os.path.join(data_dir, case_name + '_{}_wt'.format(str_i)), wt)
        np.save(os.path.join(data_dir, case_name + '_{}_et'.format(str_i)), et)
        np.save(os.path.join(data_dir, case_name + '_{}_tc'.format(str_i)), tc)
        np.save(os.path.join(data_dir, case_name + '_{}_seg'.format(str_i)), seg_i)

        with open(data_list, 'a') as l:
            l.writelines(case_name + '_{}\n'.format(str_i))
    
    return case_name + dataset_type

def process_training_case(case_name):
    _process_single_case(case_name)


def process_validating_case(case_name):
    _process_single_case(case_name, 'val')

if __name__ == "__main__":
    training_rate = 0.85
    case_list = sorted(os.listdir('./BRATS2018/{}/'.format(dataset_type)))
    subset_list = case_list[:100]
    training_num = int(len(subset_list) * training_rate)
    train_list = subset_list[:training_num]
    val_list = subset_list[training_num:]
    
    # with ThreadPoolExecutor(max_workers=4) as executor:
    #     executor.map(process_training_case, train_list)

    # with ThreadPoolExecutor(max_workers=4) as executor:
    #     executor.map(process_validating_case, val_list)
    list(map(process_training_case, train_list))
    list(map(process_validating_case, val_list))

    num_train_cases = sum([1 for line in open(train_list_txt)])
    num_val_cases = sum([1 for line in open(val_list_txt)])
    assert(len(os.listdir(train_dir)) == 5 * num_train_cases)
    assert(len(os.listdir(val_dir)) == 5 * num_val_cases)