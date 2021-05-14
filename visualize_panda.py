
import numpy as np
from mayavi import mlab

from pandaset import geometry
import math



idx = 4000


lidar_file_panda = "/home/gaoming/Documents/padsToKit/training/lidar/"+ str(idx).rjust(6,'0') + ".bin"
label_file_panda = '/home/gaoming/Documents/padsToKit/training/label/'+ str(idx).rjust(6,'0') + ".txt"
lidar_panda = np.fromfile(lidar_file_panda, dtype = np.float32).reshape(-1,4)
#lidar_panda[:,3] = lidar_panda[:,3]/255 
#print("this is panda lidar point\n",lidar_panda[0:1010])





map_name_from_general_to_detection = {
    'Car' : 'car',
    'Medium_sized_Truck' : 'car',
    'Pickup_Truck' : 'car',
    'Rolling_Containers' : 'ignore',
    'Pedestrian' : 'pedestrian',
    'Signs' : 'ignore',
    'Bicycle' : 'bicycle',
    'Cones' : 'ignore',
    'Pedestrian_with_Object' : 'pedestrian',
    'Motorized_Scooter' : 'bicycle',
    'Bus' : 'ignore',
    'Motorcycle' : 'bicycle',
    'Personal_Mobility_Device' : 'ignore',
    'Other_Vehicle_Pedicab' : 'ignore',
    'Emergency_Vehicle' : 'ignore',
    'Pylons' : 'ignore',
    'Tram/Subway' : 'ignore',
    'Animals_Other' : 'ignore',
    'Construction_Signs' : 'ignore',
    'Other_Vehicle_Construction_Vehicle' : 'ignore',
    'Towed_Object' : 'ignore',
    'Temporary_Construction_Barriers' : 'ignore',
    'Semi_truck' : 'ignore',
    'Road_Barriers' : 'ignore',
    'Other_Vehicle___Uncommon' : 'ignore',
    'Train' : 'ignore',
    'Animals_Bird' : 'ignore',
    'Animals___Other' : 'ignore',
    'Tram_/_Subway' : 'ignore',
    'Other_Vehicle___Construction_Vehicle' : 'ignore',
    'Misc' : 'ignore',
    'DontCare' : 'ignore'
}
'''
classes = {'truck', 'pedicab', 'ignore', 'car', 'construction_vehicle', 
           'animals', 'cone', 'train', 'bicycle', 'pedestrian', 'bus', 
           'motorcycle', 'barrier'
}
'''
classes = {
    'car','bicycle','pedestrian'
}
def remove_point_outside_range(point):
    pc_area_scope = np.array([[-40, 40], [-1, 3], [0, 100]], 'float32')
    flag_list = []
    for i in range(point.shape[0]):
        if pc_area_scope[2][0]  < point[i][0] and point[i][0] < pc_area_scope[2][1]:
            if pc_area_scope[1][0]  < point[i][2] and point[i][2] < pc_area_scope[1][1]:
                if pc_area_scope[0][0]  < point[i][1] and point[i][1] < pc_area_scope[0][1]:
                    flag_list.append(i)
    print(flag_list)
    return point[flag_list]

points_v = lidar_panda


def get_label_anno(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()
    content = [line.strip().split(' ') for line in lines]
    annotations['name'] = np.array([x[0] for x in content])
    annotations['truncated'] = np.array([float(x[1]) for x in content])
    annotations['occluded'] = np.array([int(x[2]) for x in content])
    annotations['alpha'] = np.array([float(x[3]) for x in content])
    annotations['bbox'] = np.array([[float(info) for info in x[4:8]]
                                    for x in content]).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array([[float(info) for info in x[8:11]]
                                          for x in content
                                          ]).reshape(-1, 3)[:, [2, 0, 1]]
    annotations['location'] = np.array([[float(info) for info in x[11:14]]
                                        for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array([float(x[14])
                                          for x in content]).reshape(-1)
    if len(content) != 0 and len(content[0]) == 16:  # have score
        annotations['score'] = np.array([float(x[15]) for x in content])
    else:
        annotations['score'] = np.zeros([len(annotations['bbox'])])
    return annotations
gt_area_scope = {
            'car' : np.array([[-40, 40], [-1, 3], [0, 100]], 'float32'),
            'bicycle' : np.array([[-40, 40], [-1, 3], [0, 100]], 'float32'),
            'pedestrian':np.array([[-40, 40], [-1, 3], [0, 100]],'float32')}





def obj_to_boxes3d_with_classes(annotation_data):
        gt_boxes_3d = {obj_class : [] for obj_class in classes}
        dimensions = annotation_data['dimensions']
        location = annotation_data['location']
        yaw = annotation_data['rotation_y'] 
        name = annotation_data['name']

        for obj in range(len(name)):

            object_class = map_name_from_general_to_detection[
                name[obj]]
            
            if map_name_from_general_to_detection[
                name[obj]] == 'ignore':
                continue
            l = dimensions[obj][0]
            h = dimensions[obj][1]
            w = dimensions[obj][2]

            x = location[obj][0]
            y = location[obj][1]
            z = location[obj][2]

            rot_y =  np.deg2rad(yaw[obj])
            
            gt_scope = gt_area_scope[object_class]
            
            if gt_scope[0, 0] < y < gt_scope[0, 1] and gt_scope[
                    1, 0] < z < gt_scope[1, 1] and gt_scope[
                       2, 0] < x < gt_scope[2, 1]:
                gt_boxes_3d[object_class].append(
                    [x, y, z, h, w, l, rot_y])
            
            #np.concatenate([loc,dims, rots[..., np.newaxis]], axis=1)
            #gt_boxes_3d[map_name_from_general_to_detection[
                #name[obj]]].append([x, y, z, h, w, l, rot_y])
        
        for obj_class in gt_boxes_3d.keys():
            boxes = gt_boxes_3d[obj_class]
            gt_boxes_3d[obj_class] = np.array(
                boxes) if len(boxes) > 0 else np.zeros((0, 7))
        return gt_boxes_3d




def draw_gt_boxes3d(gt_boxes3d, fig, color=(1, 1, 1)):

    """

    Draw 3D bounding boxes

    Args:

        gt_boxes3d: numpy array (3,8) for XYZs of the box corners

        fig: figure handler

        color: RGB value tuple in range (0,1), box line color

    """
    gt_boxes3d = geometry.center_box_to_corners(gt_boxes3d)
    gt_boxes3d = gt_boxes3d.T
    for k in range(0, 4):

        i, j = k, (k + 1) % 4

        mlab.plot3d([gt_boxes3d[0, i], gt_boxes3d[0, j]], [gt_boxes3d[1, i], gt_boxes3d[1, j]],

                    [gt_boxes3d[2, i], gt_boxes3d[2, j]], tube_radius=None, line_width=2, color=color, figure=fig)



        i, j = k + 4, (k + 1) % 4 + 4

        mlab.plot3d([gt_boxes3d[0, i], gt_boxes3d[0, j]], [gt_boxes3d[1, i], gt_boxes3d[1, j]],

                    [gt_boxes3d[2, i], gt_boxes3d[2, j]], tube_radius=None, line_width=2, color=color, figure=fig)



        i, j = k, k + 4

        mlab.plot3d([gt_boxes3d[0, i], gt_boxes3d[0, j]], [gt_boxes3d[1, i], gt_boxes3d[1, j]],

                    [gt_boxes3d[2, i], gt_boxes3d[2, j]], tube_radius=None, line_width=2, color=color, figure=fig)

    return fig 

def draw_boxes(gt_boxes_3d_with_class,fig):
    for c,boxes in gt_boxes_3d_with_class.items():
        for box in boxes:
            mlab.text3d(box[0], box[1], box[2], c, scale=(1, 1, 1))
            draw_gt_boxes3d((box[0],box[1],box[2],box[5],box[4],box[3],box[6]), fig)



points_v = remove_point_outside_range(points_v)
print(points_v.shape)
annotation = get_label_anno(label_file_panda)
gt_boxes_3d = obj_to_boxes3d_with_classes(annotation)



fig = mlab.figure(figure=None, bgcolor=(0,0,0),engine=None, size=(1920, 1080))
mlab.points3d(points_v[:,0], points_v[:,1],points_v[:,2], points_v[:,3], mode='point')


draw_boxes(gt_boxes_3d,fig)


mlab.show()