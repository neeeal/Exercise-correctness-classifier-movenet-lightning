import os
import shutil
from random import shuffle

#### actual classes
filepath = 'D:\\Documents\\College\\3rd Year\\2nd Sem\\CPE 020\\final_proj\\final'
class_names = os.listdir(filepath)
class_names.sort()

#### Filtering blurred images using laplacian
save_dir = 'D:\\Documents\\College\\3rd Year\\2nd Sem\\CPE 020\\final_proj\\dataset'
if os.path.isdir(save_dir):
    pass
else:
    os.mkdir(save_dir)


for m,class_ in enumerate(class_names):
##    if m==1: continue
    c_path = os.path.join(filepath,class_)
    ex_names =  os.listdir(c_path)
    ex_names.sort()

    for ex in ex_names:
        ex_path = os.path.join(c_path,ex)
        ex_images = os.listdir(ex_path)
        shuffle(ex_images)
        shuffle(ex_images)

        for n,image in enumerate(ex_images):
            '''
                IMAGES PER CLASS PER IMAGE
            '''
            if n == 5000: break
            path = os.path.join(ex_path,image)
            
            save_path_ex = os.path.join(save_dir,ex)
            if os.path.isdir(save_path_ex): pass
            else: os.mkdir(save_path_ex)
            
            save_path_c = os.path.join(save_path_ex,class_)
            if os.path.isdir(save_path_c): pass
            else: os.mkdir(save_path_c)
            
            shutil.copy(path, save_path_c)
            if n%500==0: print(f'{class_} {ex} {n} file(s) moved successfully')

print('Execution finished')
