import os
import shutil
import cv2


#### Getting filenames to be set as classnames
filepath = 'D:\\Documents\\College\\3rd Year\\2nd Sem\\CPE 020\\final_proj\\raw_bad\\bad_dataset'
class_names = os.listdir(filepath)
class_names.sort()
##print(class_names)


#### Extracting Images from videos
save_dir = 'D:\\Documents\\College\\3rd Year\\2nd Sem\\CPE 020\\final_proj\\raw_badimages'
if os.path.isdir(save_dir):
    pass
else:
    os.mkdir(save_dir)

#### Only 5 classes to be extracted
#### Can also extract all if needed   
for m,c in enumerate(class_names[:]):
    path = os.path.join(filepath,c)
    save_path = os.path.join(save_dir,c)
    if os.path.isdir(save_path):
        pass
    else:
        os.mkdir(save_path)
    
    for n,img in enumerate(os.listdir(path)):
        print(f'Currently processing class {m}    file #{n}')
        # Opens the Video file
        cap= cv2.VideoCapture(os.path.join(path,img))
        i=0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            cv2.imwrite(f'{save_path}/{c}_{n}_{img.split(".")[0]}_frame_{i}.jpg',frame)
            #cv2.imwrite(save_dir+img+c+str(i)+'.jpg',frame)
            i+=1
    cap.release()
    cv2.destroyAllWindows()

print(f'There are {len(os.listdir(save_dir))} images')
