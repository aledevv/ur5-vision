# step 3 create a JSON file of annotations for each folder

import json
import os
import numpy as np
import cv2
import string, random

###################################
#          GLOBAL VARS            #
###################################

images_list_dict = []           # dictionary containing all the images of the dataset
annotations_list_dict = []      # dictionary containing all the annotations of the object inside the images

_json = {
    "info": {
        "year": "2023",
        "version": "1",
        "description": "Created with a script",
        "contributor": "",
        "url": "",
        "date_created": ""
    },
    "licenses": [
        {
            "id": 1,
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "name": "CC BY 4.0"
        }
    ],
    "categories": [
        {
            "id": 0,
            "name": "blocks",
            "supercategory": "none"
        },
        {
            "id": 1,
            "name": "X1-Y1-Z2",
            "supercategory": "blocks"
        },
        {
            "id": 2,
            "name": "X1-Y2-Z1",
            "supercategory": "blocks"
        },
        {
            "id": 3,
            "name": "X1-Y2-Z2",
            "supercategory": "blocks"
        },
        {
            "id": 4,
            "name": "X1-Y2-Z2-CHAMFER",
            "supercategory": "blocks"
        },
        {
            "id": 5,
            "name": "X1-Y2-Z2-TWINFILLET",
            "supercategory": "blocks"
        },
        {
            "id": 6,
            "name": "X1-Y3-Z2",
            "supercategory": "blocks"
        },
        {
            "id": 7,
            "name": "X1-Y3-Z2-FILLET",
            "supercategory": "blocks"
        },
        {
            "id": 8,
            "name": "X1-Y4-Z1",
            "supercategory": "blocks"
        },
        {
            "id": 9,
            "name": "X1-Y4-Z2",
            "supercategory": "blocks"
        },
        {
            "id": 10,
            "name": "X2-Y2-Z2",
            "supercategory": "blocks"
        },
        {
            "id": 11,
            "name": "X2-Y2-Z2-FILLET",
            "supercategory": "blocks"
        }
    ],
    'images': [],
    'annotations': []
}

###################################
#           FUNCTIONS             #
###################################

def extractAnnotations(json_file_path, img_id, annotation_id):
    with open(json_file_path) as json_data:     # open the passed json file
        data = json.load(json_data)
    
    num_of_objects=len(data)            # num of objects in the image
    
    #for each object, get useful data and write its annotation
    for current_obj in data:
        # get label
        label = data[current_obj]['y']
        # get bbox
        bbox = data[current_obj]['bbox']
        
        # FIXING BBOX
        
        # get 3d bbox pixels
        
        bbox3D = data[current_obj]['3d_bbox_pixel_space']
        
        # fixing width
        bbox[2]=bbox[2]-bbox[0]
        
        #fixing height
        bbox[3]=bbox3D[-1][1]-bbox[1]  # bottom left corner of the 3D bbox
        
        #write the annotation
        newAnnotation(annotation_id, img_id, label, bbox)
        annotation_id+=1
        
    return annotation_id
    



''' IMAGES has fields:
- id: (progressive)
- licence: 1
- file_name: (the name of the img)
- height: (img height)
- width: (img width)
- date_captured: x
'''

def newImage(img_id, image_name, height, width):
    
    img_dict = {'id': img_id,
                'licence': 1,
                'file_name': image_name,
                'height': height,
                'width': width,
                'date_captured': 'x'}
    
    addToImagesDict(img_dict)


def addToImagesDict(dictToAdd): # Use dictionaries that are as json
    images_list_dict.append(dictToAdd)




''' ANNOTATIONS has fields:
- id: (progressive)
- image_id: (id of one of the images in IMAGES)
- category_id: (label in classes vector)
- bbox: (vector of 4 values [x0,y0,width,height])
- area: (bbox area)
- segmentation: []
- iscrowd: 0
'''    



def newAnnotation(id, image_id, label, bbox):
    
    ann_dict = {'id': id,
                'image_id': image_id,
                'category_id': getCategoryId(label),
                'bbox': bbox,
                'area': bbox[2]*bbox[3], # Area = width * height -> they are respectively at 3rd and 4th index of the bbox
                'segmentation': [],
                'iscrowd': 0}
    addToAnnotations(ann_dict)


def addToAnnotations(annToAdd): # Use dictionaries that are as json
    annotations_list_dict.append(annToAdd)
   
   
def getCategoryId(label):
    for i in range(0, len(_json["categories"])):
        if _json["categories"][i]["name"] == label:
            return _json["categories"][i]["id"]

   
def delete_JSON_files(path):
    for item in os.listdir(path):
        if item.endswith('.json') and not item.startswith('_annotations'):
            os.remove(path+'/'+item) 
   
''' #TO GET THE NAME OF THE CLASSES
def writeClasses(data):
    for current_obj in data:
        classes.append(data[current_obj]['y'])
        #print(data[current_obj]['y'])'''

def extractJSON(path):
    img_id=0           #image id
    annotation_id=0    #annotation id
    
    for file in os.listdir(path):
        if file.endswith('.json'): # file = json_file
            #image and json file have the same name, just setting the extension .jpeg to open the img afterwards
            img_file = file[:len(file)-3]+'peg'
            
            #open the relative image
            opened_img = cv2.imread(path+'/'+img_file,0)
            height, width = opened_img.shape[:2]
            
            #add image to images section of the json
            newImage(img_id, img_file, height, width)
            
            annotation_id=extractAnnotations(path+'/'+file, img_id, annotation_id)
                    
            img_id+=1
    
    
    # completa da _json
    _json["images"]=images_list_dict
    _json["annotations"]=annotations_list_dict
    
    
    # Writing the disctionary into a JSON file                     
    with open(path+'/_annotations.coco.json', 'w') as outfile:
 
        json.dump(_json, outfile, indent=7)
    
    # clear disctionaries for next iteration
    images_list_dict.clear()
    annotations_list_dict.clear()

    print(f'Finito {path}')


def main():

    classes = [ 'X1-Y1-Z2', 'X1-Y2-Z1', 'X1-Y2-Z2', 'X1-Y2-Z2-CHAMFER', 'X1-Y2-Z2-TWINFILLET',
                'X1-Y3-Z2', 'X1-Y3-Z2-FILLET', 'X1-Y4-Z1', 'X1-Y4-Z2', 'X2-Y2-Z2',
                'X2-Y2-Z2-FILLET']
    
    
    # let's get all data
    main_folder = 'dataset/'

    #print(os.path.join(main_folder, 'assign1'))

    for folder in os.listdir(main_folder): # train, valid, test
        if not folder.startswith('.') and not folder.startswith('imgs') and os.path.isdir(main_folder+'/'+folder): # exclude folder starting with .
            path = main_folder+'/'+folder
            #for each one extract json
            extractJSON(path)
            
            delete_JSON_files(path)
                            
    
    
    
    
if __name__ == "__main__":
    main()