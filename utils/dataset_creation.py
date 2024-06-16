#  STEP 2
# create dataset of 3 folders

import os
import shutil
import random

def copyImageAndJSON(image, dataset_folder, newFolder):
    json_name = image[:len(image)-3]+'son'
    shutil.copy(dataset_folder+'/imgs/'+image, dataset_folder+'/'+newFolder+'/'+image)
    shutil.copy(dataset_folder+'/imgs/'+json_name, dataset_folder+'/'+newFolder+'/'+json_name)

def main():
    
    img_list = []
    dataset_folder = 'dataset'
    
    
    # creation of train, val, test folders
    
    if not os.path.exists(dataset_folder+'/train'): os.mkdir(dataset_folder+'/train')
    if not os.path.exists(dataset_folder+'/valid'): os.mkdir(dataset_folder+'/valid')
    if not os.path.exists(dataset_folder+'/test'): os.mkdir(dataset_folder+'/test')
    
    for image in os.listdir(dataset_folder+'/imgs'):
        if image.endswith('.jpeg'):
            img_list.append(image)
    
    #shuffle the elements
    random.shuffle(img_list)
    
    #store total number of elements
    num_of_elements = len(img_list)         # in our case 1500
    
    
    #define num of items per folder
    train_size = num_of_elements*.7
    val_size = num_of_elements*.2
    test_size = num_of_elements*.1
    
    #distributing images in lists
    train_items = img_list[:int(train_size)]  # 0 - 1049 [1050 items] -> 70%
    valid_items = img_list[int(train_size):int(train_size+val_size)]  # 1050 - 1349 [300 items] -> 20%
    test_items = img_list[int(train_size+val_size):]  # 1350 - 1499 [150 items] -> 10%
    
    # copying images and json in their relative folder
    for image in train_items:
        copyImageAndJSON(image, dataset_folder, 'train')
        
    for image in valid_items:
        copyImageAndJSON(image, dataset_folder, 'valid')
        
    for image in test_items:
        copyImageAndJSON(image, dataset_folder, 'test')
    
if __name__ == "__main__":
    main()