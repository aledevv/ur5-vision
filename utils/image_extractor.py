# STEP 0
# Download dataset of models (named assigns) and keep it in the same folder of the scripts

#STEP 1
# run this script to create images folder containing an image with random name and its relative json file of features

# WARNING cambia nome cartella sull'altro quandro ricrei il JSON

import os
import shutil
import numpy as np
import random, string

def randomword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))

def main():
    
    #create new folder
    if not os.path.exists('dataset'): os.mkdir('dataset')
    if not os.path.exists('dataset/imgs'): os.mkdir('dataset/imgs')
    
    img_idx = 0         #image index
    main_folder = 'assigns'
    for assign_folder in os.listdir(main_folder):
        if not assign_folder.startswith('.'):
            for scene in os.listdir(main_folder+'/'+assign_folder):
                if not scene.startswith('.'):
                    for file in os.listdir(main_folder+'/'+assign_folder+'/'+scene):
                        if file.endswith('.json'): # images we want gave the same name of json files
                            image_name = file[:len(file)-3]+'peg'
                            
                            #move both image and json
                            shutil.copy(main_folder+'/'+assign_folder+'/'+scene+'/'+image_name, 'dataset/imgs')
                            shutil.copy(main_folder+'/'+assign_folder+'/'+scene+'/'+file, 'dataset/imgs')
                            #rename both image and json
                            new_file_name = randomword(8)
                            os.renames('dataset/imgs/'+image_name,'dataset/imgs/'+new_file_name+'.jpeg')
                            os.renames('dataset/imgs/'+file,'dataset/imgs/'+new_file_name+'.json')
                            
                            img_idx+=1
                            

if __name__ == "__main__":
    main()