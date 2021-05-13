import numpy as np
import glob


lidar_train_list = glob.glob("/home/gaoming/Documents/padsToKit/training/lidar/*.bin")
label_train_list = glob.glob("/home/gaoming/Documents/padsToKit/training/label/*.txt")

database_size = len(lidar_train_list)



label_train_list_filename = []
lidar_train_list_filename = []

for i in range(len(lidar_train_list)):
    label_train_list_filename.append(int(label_train_list[i][-10:-4]))
    lidar_train_list_filename.append(int(lidar_train_list[i][-10:-4]))
'''

different_lidar_label = []
for i in range(len(lidar_train_list)):
    if lidar_train_list_filename[i] not in label_train_list_filename:
        different_lidar_label.append(lidar_train_list_filename[i])



print(len(lidar_train_list))
print(len(label_train_list))
print(len(different_lidar_label))

'''
label_train_list_filename = sorted(label_train_list_filename)
lidar_train_list_filename = sorted(lidar_train_list_filename)

rate = 0.6
split = int(database_size * rate)

lidar_train_list_filename_splited = label_train_list_filename[:split]
lidar_vail_list_filename_splited = label_train_list_filename[split:]



train_txt = "/home/gaoming/Documents/ImageSets/train.txt"
with open(train_txt,'w') as f:
    for line in lidar_train_list_filename_splited:
        f.write("%06d\n"%line)

vail_txt = "/home/gaoming/Documents/ImageSets/val.txt"
with open(vail_txt,'w') as f:
    for line in lidar_vail_list_filename_splited:
        f.write("%06d\n"%line)



