from pandaset import DataSet
import numpy as np
import math
from pandaset import geometry
from pathlib import Path
import os
from mayavi import mlab
import matplotlib.pyplot as plt
import pandas as pd

import tracemalloc
import sys,os

# dir string
kitti_root = ''
dir_img_test = ''
dir_img_train = ''
dir_label_test = ''
dir_label_train = ''
dir_lidar_test = ''
dir_lidar_train = ''
pd_root = ''
idx = 0

def cuboids_to_boxes(cuboids0, poses):
    matri_ret = []
    numb_ob = 0
    for i, row in cuboids0.iterrows():
        
            # 　坐标信息
            box = row["position.x"], row["position.y"], row["position.z"], row["dimensions.x"], row["dimensions.y"],  row["dimensions.z"], row["yaw"]
            # 将中心点，长宽高和航向角信息转变为８个顶点的信息
            corners = geometry.center_box_to_corners(box)
            rotate_corners = geometry.lidar_points_to_ego(corners, poses)  
            # 将８个顶点的坐标位置进行坐标系转换
            s = rotate_corners
            p0 = [s[0][0], s[0][1], s[0][2]]
            p1 = [s[1][0], s[1][1], s[1][2]]
            p2 = [s[2][0], s[2][1], s[2][2]]
            p3 = [s[3][0], s[3][1], s[3][2]]
            p4 = [s[4][0], s[4][1], s[4][2]]
            p5 = [s[5][0], s[5][1], s[5][2]]
            p6 = [s[6][0], s[6][1], s[6][2]]
            p7 = [s[7][0], s[7][1], s[7][2]]
            
            x = (p0[0] + p1[0] + p2[0] + p3[0] + p4[0] + p5[0] + p6[0] + p7[0]) / 8
            y = (p0[1] + p1[1] + p2[1] + p3[1] + p4[1] + p5[1] + p6[1] + p7[1]) / 8
            z = (p0[2] + p1[2] + p2[2] + p3[2] + p4[2] + p5[2] + p6[2] + p7[2]) / 8

            l = math.sqrt(math.pow((p1[0] - p0[0]), 2) + math.pow(p1[1] - p0[1], 2))
            w = math.sqrt(math.pow((p3[0] - p0[0]), 2) + math.pow(p3[1] - p0[1], 2))
            h = math.sqrt(math.pow(p4[2] - p0[2], 2))
            sina = float((p0[0] - p1[0]) / l)
            cosa = float((p0[1] - p1[1]) / l)
            yaw = math.atan(sina / cosa)
            obj = [row["label"], x,y, z, w, l, h, yaw] 
            matri_ret.append(obj)
            numb_ob = numb_ob + 1

    return np.array(matri_ret), numb_ob

def filterByDensity(cubiod_pandas,lidar_pandas):
    '''
    this method is used to filter the data of machine lidar

    using density to swipe out the machine lidar data of sensor_id = -1, 
    which is mixture of solid lidar and machine lidar
    '''

    # calculate the minimal density
    d_cubiod_pandas = cubiod_pandas[cubiod_pandas['cuboids.sensor_id']==1] # only retian solid lidar data
    min_density = float('inf')
    for index, row in d_cubiod_pandas.iterrows():
        box = [row["position.x"], row["position.y"], row["position.z"], row["dimensions.x"], row["dimensions.y"],  row["dimensions.z"], row["yaw"]]
        #将中心点，长宽高和航向角信息转变为８个顶点的信息
        corners = geometry.center_box_to_corners(box)
        x = [p[0] for p in corners]
        y = [p[1] for p in corners]
        z = [p[2] for p in corners]
        volume = row["dimensions.x"]*row["dimensions.y"]*row["dimensions.z"]
        constrain_x1 = min(x)
        constrain_x2 = max(x)
        constrain_y1 = min(y)
        constrain_y2 = max(y)
        constrain_z1 = min(z)
        constrain_z2 = max(z)
        constrain_lidar0 = lidar_pandas[(lidar_pandas['x']>=constrain_x1) & 
                                        (lidar_pandas['x']<=constrain_x2) &
                                        (lidar_pandas['y']>=constrain_y1) &
                                        (lidar_pandas['y']<=constrain_y2) &
                                        (lidar_pandas['z']>=constrain_z1) &
                                        (lidar_pandas['z']<=constrain_z2) 
                                            ]
        density = constrain_lidar0.shape[0]/volume
        if (density<min_density) & (density!=0):
            min_density = density

    
    # use the minimal density to filter the cuboids
    cubiod_pandas = cubiod_pandas[(cubiod_pandas['cuboids.sensor_id']==1) |(cubiod_pandas['cuboids.sensor_id']==-1)]
    for index, row in cubiod_pandas.iterrows():
        position_x = row['position.x']
        position_y = row['position.y']
        position_z = row['position.z']
        dimensions_x = row['dimensions.x']
        dimensions_y = row['dimensions.y']
        dimensions_z = row['dimensions.z']
        volume = dimensions_x*dimensions_y*dimensions_z
        constrain_x1 = position_x - dimensions_x/2
        constrain_x2 = position_x + dimensions_x/2
        constrain_y1 = position_y - dimensions_y/2
        constrain_y2 = position_y + dimensions_y/2
        constrain_z1 = position_z - dimensions_z/2
        constrain_z2 = position_z + dimensions_z/2
        constrain_lidar = lidar_pandas[(lidar_pandas['x']>=constrain_x1) & 
                                        (lidar_pandas['x']<=constrain_x2) &
                                        (lidar_pandas['y']>=constrain_y1) &
                                        (lidar_pandas['y']<=constrain_y2) &
                                        (lidar_pandas['z']>=constrain_z1) &
                                        (lidar_pandas['z']<=constrain_z2) 
                                            ]

        density = constrain_lidar.shape[0]/volume

        if (density<min_density) | (density==0):
            cubiod_pandas = cubiod_pandas.drop(index)
    
    return cubiod_pandas
        
def imgProcess(seq,cubiod_pandas,j):
    '''
    i = frame
    1. used to filter annotation deeper
    2. used to generate 2d bounding box on img
    '''
    camera_name = "front_camera"
    points3d_lidar_xyz = cubiod_pandas.to_numpy()[:,5:8].astype(np.float) # position xyz
    choosen_camera = seq.camera[camera_name]
    projected_points2d, camera_points_3d, inner_indices = geometry.projection(lidar_points=points3d_lidar_xyz, 
                                                                            camera_data=choosen_camera[j],
                                                                            camera_pose=choosen_camera.poses[j],
                                                                            camera_intrinsics=choosen_camera.intrinsics,
                                                                            filter_outliers=True)
    cubiod_pandas = cubiod_pandas.iloc[inner_indices,:]  # use img to filter the cubiod_pandas

    
    # use filtered data to find the projected_2d
    cubiod_projected_2d = []# store the projected 2d point
    for i, row in cubiod_pandas.iterrows():
        box = row["position.x"], row["position.y"], row["position.z"], row["dimensions.x"], row["dimensions.y"],  row["dimensions.z"], row["yaw"]
        corners = geometry.center_box_to_corners(box)
        row_2d, row_3d, inner_indices = geometry.projection(lidar_points=corners, 
                                                                                camera_data=choosen_camera[j],
                                                                                camera_pose=choosen_camera.poses[j],
                                                                                camera_intrinsics=choosen_camera.intrinsics,
                                                                                filter_outliers=True)
        x = row_2d[:,0]
        y = row_2d[:,1]
        xmin = min(x)
        xmax = max(x)
        ymin = min(y)
        ymax = max(y)
        cur_2d =  [xmin,ymin,xmax,ymax]
        cubiod_projected_2d.append(cur_2d)

    cubiod_projected_2d = np.array(cubiod_projected_2d)
    cubiod_projected_2d = cubiod_projected_2d.astype(np.float32)
    return  cubiod_pandas, cubiod_projected_2d

def transfer(i):

        dataset = DataSet(pd_root)
        # load the environment data
        seq = dataset[i]
        seq.load()
        seq.load_lidar().load_cuboids() # load lidar and cuboids of annotations
        seq.lidar.set_sensor(1) # only solid lidar
        
        for j0 in range(0,80): # train data j0->frame
            print('now is processing {} sequence, {}th frame'.format(i,j0))
            lidar_pose = seq.lidar.poses[j0] # get lidar poses

            # annotation transfer
            lidar_pandas = seq.lidar[j0]
            cubiod_pandas = seq.cuboids[j0]
            cubiod_pandas = filterByDensity(cubiod_pandas,lidar_pandas)
            cubiod_pandas, cubiod_projected_2d = imgProcess(seq,cubiod_pandas,j0) # find 2d bounding box
            # annotation word->ego
            matric_res,num_obj = cuboids_to_boxes(cubiod_pandas, lidar_pose)
            cubiod_pandas['yaw'] = matric_res[:,7]
            cubiod_pandas.iloc[:,5:11] = matric_res[:,1:7]
            cubiod_pandas['label']=cubiod_pandas['label'].str.replace(' ','_')
            cubiod_pandas['label']=cubiod_pandas['label'].str.replace('-','_')
            # Aggregate data 
            '''
              dataType must be float32, since in the framework, np read the file as float32
              if directly use 'object' type to contain float32 data, the data would be randomly extend to 64 bit
              Therefore, firstly process number data and transfer it to string, and then add the label column which is string type 
            '''
            kitti = np.zeros(shape=(cubiod_pandas.shape[0],16),dtype= np.float32)
            kitti[:,4:8] = cubiod_projected_2d #cubiod projected 2d xmin、ymin、xmax、ymax
            kitti[:,8]= cubiod_pandas.iloc[:,10].astype(np.float32) #h
            kitti[:,9]= cubiod_pandas.iloc[:,8].astype(np.float32)  #w
            kitti[:,10]= cubiod_pandas.iloc[:,9].astype(np.float32)  #l
            kitti[:,11:14] = cubiod_pandas.iloc[:,5:8].astype(np.float32) # cubiod position
            kitti[:,14] = cubiod_pandas['yaw'].astype(np.float32) #rotation
            [rows,cols] = kitti.shape

            kitti_str = np.zeros(shape=(cubiod_pandas.shape[0],16),dtype='object')
            for i in range(rows):
                 for j in range(cols):
                    kitti_str[i,j] = str(kitti[i,j])
            
            kitti = kitti_str
            kitti[:,0] = cubiod_pandas['label'] #label
            print(kitti[0])
            # lidarPoint transfer
            lidar_np = lidar_pandas.to_numpy()[:,:4]
            ego_lidar_np = geometry.lidar_points_to_ego(lidar_np[:, :3], lidar_pose)
            lidar_np = np.column_stack((ego_lidar_np,lidar_np[:,3])) # aggregate data, lidar_np[:,3] is tensity

            # store
            global idx
            img = object
            if(j0<=63): # train dir
                path_lidar = os.path.join(dir_lidar_train,str(idx).rjust(6,'0')+'.bin')
                path_label = os.path.join(dir_label_train,str(idx).rjust(6,'0')+'.txt')
                img = seq.camera['front_camera'][j0]
                path_img = os.path.join(dir_img_train,str(idx).rjust(6,'0')+'.png')
            else: # test dir
                path_lidar = os.path.join(dir_lidar_test,str(idx).rjust(6,'0')+'.bin')
                path_label = os.path.join(dir_label_test,str(idx).rjust(6,'0')+'.txt')
                img = seq.camera['front_camera'][j0]
                path_img = os.path.join(dir_img_test,str(idx).rjust(6,'0')+'.png')

            
            lidar_np = lidar_np.astype(np.float32) # save lidar
            lidar_np.tofile(path_lidar)
            np.savetxt(path_label, kitti, delimiter=" ",fmt='%s') # annotation 
            img.save(path_img,'png')# img

            kitti = None
            lidar_pandas = None
            cubiod_pandas =None
            idx = idx+1
        



def createFile(init):
    path = Path(pd_root)
    if not os.path.exists(pd_root):
        print('can not find the root, please check it')
        exit(1)

    parentPath = str(path.parent.absolute())
    kitti_root = os.path.join(parentPath,'padsToKit')

    global dir_img_test,dir_img_train,dir_label_test,dir_label_train,dir_lidar_test,dir_lidar_train
    dir_img_test = os.path.join(kitti_root,'data_object_image','testing','image')
    dir_img_train = os.path.join(kitti_root,'data_object_image','training','image')  
    dir_label_test = os.path.join(kitti_root,'data_object_label','testing','label')
    dir_label_train = os.path.join(kitti_root,'data_object_label','training','label')
    dir_lidar_test = os.path.join(kitti_root,'data_object_velodyne','testing','velodyne')
    dir_lidar_train = os.path.join(kitti_root,'data_object_velodyne','training','velodyne')
    if(init == '0'):
        os.makedirs(dir_img_test),os.makedirs(dir_img_train),os.makedirs(dir_label_test),os.makedirs(dir_label_train),os.makedirs(dir_lidar_test),os.makedirs(dir_lidar_train)
    
if __name__ == '__main__':
    pd_root = sys.argv[1]
    seqNum = sys.argv[2]
    init = sys.argv[3] #防止外部调用重复创建文件夹
    initIdx = sys.argv[4]
    idx = idx + int(initIdx)
    createFile(init) 
    transfer(seqNum)


        