import imagehash
from PIL import Image
import numpy as np
import os
import shutil
def alpharemover(image):
    if image.mode != 'RGBA':
        return image
    canvas = Image.new('RGBA', image.size, (255,255,255,255))
    canvas.paste(image, mask=image)
    return canvas.convert('RGB')
def with_ztransform_preprocess(hashfunc, hash_size=8):
    def function(path):
        image = alpharemover(Image.open(path))
        image = image.convert("L").resize((hash_size, hash_size), Image.ANTIALIAS)
        '''
        Warning (from warnings module):
          File "D:\Documents\College\3rd Year\2nd Sem\CPE 020\final_proj\scripts\duplicate_filter.py", line 15
            image = image.convert("L").resize((hash_size, hash_size), Image.ANTIALIAS)
        DeprecationWarning: ANTIALIAS is deprecated and will be removed in Pillow 10 (2023-07-01). Use LANCZOS or Resampling.LANCZOS instead.
        '''
        data = image.getdata()
        quantiles = np.arange(100)
        quantiles_values = np.percentile(data, quantiles)
        zdata = (np.interp(data, quantiles_values, quantiles) / 100 * 255).astype(np.uint8)
        image.putdata(zdata)
        return hashfunc(image)
    return function
dhash_z_transformed = with_ztransform_preprocess(imagehash.dhash, hash_size = 8)
#### Getting filenames of images
filepath = 'D:\\Documents\\College\\3rd Year\\2nd Sem\\CPE 020\\final_proj\\blur_filtbad'
class_names = os.listdir(filepath)
class_names.sort()
##print(class_names)
#### Filtering duplicate images using hash
save_dir = 'D:\\Documents\\College\\3rd Year\\2nd Sem\\CPE 020\\final_proj\\dup_filtbad'
if os.path.isdir(save_dir): pass
else: os.mkdir(save_dir)
img_ls=[]
i,j=0,0

for m,c in enumerate(class_names[:]):
#    if m!=4: continue
    c_path = os.path.join(filepath,c)
    c_save_path = os.path.join(save_dir,c)
    if os.path.isdir(c_save_path):
        pass
    else:
        os.mkdir(c_save_path)
    print(c_path)

    for n,img in enumerate(os.listdir(c_path)):
        if  n%500 ==0:
            print(f'Currently doing {n}')
        path = os.path.join(c_path,img)   
        if n==0:
            img_ls.append(dhash_z_transformed(path))
            continue
        temp = dhash_z_transformed(path)
        #### If image is not a duplicate, move it to filtered1 folder
        #### Else, print File is a duplicate and continue to next image
        if temp not in img_ls:
            save_path = os.path.join(c_save_path,img)
            shutil.copy(path, save_path)
##            if i%500 ==0:
##                print(f'File {i} moved successfully')
            i+=1
        else:
##            if j%500 ==0:
##                print(f'File {j} is a duplicate')
            j+=1
            continue
        img_ls.append(temp)
print(f'There are {i} good imges \n There are {j} dup images')
