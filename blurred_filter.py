import cv2
import os
import numpy as np
import shutil

#### Getting filenames of images
filepath = 'D:\\Documents\\College\\3rd Year\\2nd Sem\\CPE 020\\final_proj\\raw_badimages'
class_names = os.listdir(filepath)
class_names.sort()
##print(class_names)

#### Filtering blurred images using laplacian
save_dir = 'D:\\Documents\\College\\3rd Year\\2nd Sem\\CPE 020\\final_proj\\blur_filtbad'
if os.path.isdir(save_dir):
    pass
else:
    os.mkdir(save_dir)

i,j,k=0,0,0

for m,c in enumerate(class_names[:]):
    c_path = os.path.join(filepath,c)
    c_save_path = os.path.join(save_dir,c)
    if os.path.isdir(c_save_path) == True:
        pass
    else:
        os.mkdir(c_save_path)
    for n,image in enumerate(os.listdir(c_path)):
        path = os.path.join(c_path,image)
        img = cv2.imread(path)
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            score = np.max(cv2.convertScaleAbs(cv2.Laplacian(gray, 3)))
            #### If image score > 160, move to filtered2
            #### Else, print File blurred and continue to next image
            print(f'File: {image}    Score: {score}')
            if score >= 160:
                save_path = os.path.join(c_save_path,image)
                shutil.copy(path, save_path)
                print('File moved successfully')
                i+=1
            else:
                print(f'    File is a blurred')
                j+=1
                continue
        except:
                print("Image error")
                k+=1

print(f'There are {i} good imges \n There are {j} blurred images\nThere are {k} errors')
