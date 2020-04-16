#%%
import os
import numpy as np
from shutil import copyfile
import pandas as pd
from tqdm import tqdm

#%%

def set_split(root, dest, train_split, valid_split):

    bbox_filename = 'bb_info.txt'

    df_train = pd.DataFrame()
    df_valid = pd.DataFrame()
    df_test = pd.DataFrame()

    # For all dirs in the root directory
    for class_dir in tqdm(os.listdir(root), desc='Copying Files - Spliting to Sets'):

        # Get the path of each dir
        class_dir_path = os.path.join(root, class_dir)

        # Ensure that it is a dir and not file
        if os.path.isdir(class_dir_path):

            # Calculate number of images in each dir
            dir_size = len([file for file in os.listdir(class_dir_path) if file.endswith(".jpg")])
            
            # Define sizes of each set (test_size not needed)
            train_size = int(round(train_split*dir_size))
            valid_size = int(round(valid_split*dir_size))

            # Import BBox Info for each class-dir as a DataFrame
            df = pd.read_csv(os.path.join(class_dir_path, bbox_filename), delim_whitespace=True)
            df['class'] = int(class_dir) # add a class column

            # Shuffle image indices of each dir
            indices = np.unique(df['img'].tolist())
            np.random.shuffle(indices)

            # Create a dictionary and split them into three datasets (lists)
            set_indices = {
                'train' : indices[:train_size],
                'valid' : indices[train_size:(train_size+valid_size)],
                'test'  : indices[(train_size+valid_size):]
            }

            # Split Bbox Info for each dataset by 
            df_train = df_train.append(df[df['img'].isin(set_indices['train'])])
            df_valid = df_valid.append(df[df['img'].isin(set_indices['valid'])])
            df_test  = df_test.append(df[df['img'].isin(set_indices['test'])])
   
            # Split the data (images)
            for x in ['train','valid', 'test']:
                for idx in set_indices[x]:
                    filename = str(idx) + '.jpg'
                    if not os.path.exists(os.path.join(dest, x, class_dir)):
                            os.makedirs(os.path.join(dest, x, class_dir))
                        
                    copyfile(os.path.join(class_dir_path, filename), os.path.join(dest, x, class_dir, filename))
            

    bbox_dict = {
        'train': df_train,
        'valid': df_valid,
        'test': df_test
    }
    
    print('\nDone! Saved files to ' + os.path.abspath(dest))
    return bbox_dict


#%%

root = "../UECFOOD100"
dest = "../splitUECFood100"
train_split = 0.70
valid_split = 0.20
# test_split is set to  (1 - train_split - valid_split)


bbox = set_split(root, dest, train_split, valid_split)

# %%
