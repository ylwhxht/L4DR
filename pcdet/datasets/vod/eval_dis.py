from pcdet.datasets.vod.kitti_object_eval_python import eval as kitti_eval
import copy
import numpy as np
import pickle
from numba import cuda
import warnings
from numba.core.errors import NumbaPerformanceWarning

# Suppress only NumbaPerformanceWarning
warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)
def read_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

gt_path="/mnt/8tssd/AdverseWeather/view_of_delft_PUBLIC/lidar/kitti_infos_val.pkl"
result_path='/home/hx/OpenPCDet-master/output/VoD_models/PP_lidar/half_fog/eval/eval_all_default/0.000/epoch_77/val/result.pkl'
#'/home/hx/OpenPCDet-master/output/VoD_models/PP_lidar/half_fog/eval/eval_all_default/0.000/epoch_72/val/result.pkl'
#'/home/hx/OpenPCDet-master/output/VoD_models/PP_lidar/half_fog/eval/epoch_71/val/0.030/result.pkl'
#'/home/hx/OpenPCDet-master/output/VoD_models/PP_lidar/half_fog/eval/epoch_71/val/0.060/result.pkl'
#'/home/hx/OpenPCDet-master/output/VoD_models/PP_lidar/half_fog/eval/epoch_71/val/0.030/result.pkl'
#result_path='/home/hx/OpenPCDet-master/output/VoD_models/PP_RLF/concat_half_fog/eval/epoch_76/val/0.030/result.pkl'
try:
    gt_pkl=read_pkl(gt_path)
except:
    gt_path=None
try:
    det_pkl=read_pkl(result_path)
except:
    result_path=None
if gt_path is None:
    print("输入gt路径")
    gt_path=input()
if result_path is None:
    print("输入result路径")
    result_path=input()


class_names=['Car','Pedestrian','Cyclist']
gt_pkl=read_pkl(gt_path)
det_pkl=read_pkl(result_path)
dis=[0,1,2,3,4]
for i in dis:
    if(i==0):
        eval_det_annos = copy.deepcopy(det_pkl)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in gt_pkl]
        

        for j in range(len(eval_det_annos)):
            x = eval_det_annos[j]['boxes_lidar'][:,0]
            y = eval_det_annos[j]['boxes_lidar'][:,1]
            z = eval_det_annos[j]['boxes_lidar'][:,2]
            mask=np.sqrt(x*x+y*y+z*z)<15
            for key in eval_det_annos[j].keys():
                if key!='frame_id':
                    eval_det_annos[j][key]= eval_det_annos[j][key][mask]
        for j in range(len(eval_gt_annos)):
            x = eval_gt_annos[j]['gt_boxes_lidar'][:,0]
            y = eval_gt_annos[j]['gt_boxes_lidar'][:,1]
            z = eval_gt_annos[j]['gt_boxes_lidar'][:,2]
            mask=np.sqrt(x*x+y*y+z*z)<15
            for key in eval_gt_annos[j].keys():
                if key!='frame_id':
                    eval_gt_annos[j][key]= eval_gt_annos[j][key][mask]
                 
        print("0-15m:")
        result,str=kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)
        print(result)
    elif(i==1):
        eval_det_annos = copy.deepcopy(det_pkl)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in gt_pkl]
        #eval_gt_annos = [copy.deepcopy(info['annos']) for info in gt_pkl]
        for j in range(len(eval_gt_annos)):
            while(len(eval_gt_annos[j]['gt_boxes_lidar'])<len(eval_gt_annos[j]['name'])):
                new_row = np.array([[0, 0, 0, 0, 0, 0, 0]])
                eval_gt_annos[j]['gt_boxes_lidar'] = np.vstack([eval_gt_annos[j]['gt_boxes_lidar'], new_row])
        for j in range(len(eval_det_annos)):
            x = eval_det_annos[j]['boxes_lidar'][:,0]
            y = eval_det_annos[j]['boxes_lidar'][:,1]
            z = eval_det_annos[j]['boxes_lidar'][:,2]
            mask = (np.sqrt(x*x+y*y+z*z) >= 15) & (np.sqrt(x*x+y*y+z*z) < 30)
            for key in eval_det_annos[j].keys():
                if key!='frame_id':
                    eval_det_annos[j][key]= eval_det_annos[j][key][mask]
        for j in range(len(eval_gt_annos)):
            x = eval_gt_annos[j]['gt_boxes_lidar'][:,0]
            y = eval_gt_annos[j]['gt_boxes_lidar'][:,1]
            z = eval_gt_annos[j]['gt_boxes_lidar'][:,2]
            mask = (np.sqrt(x*x+y*y+z*z) >= 15) & (np.sqrt(x*x+y*y+z*z) < 30)
            for key in eval_gt_annos[j].keys():
                if key!='frame_id':
                    eval_gt_annos[j][key]= eval_gt_annos[j][key][mask]
        print("15-30m:")
        result,str=kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)
        print(result)
    elif(i==2):
        eval_det_annos = copy.deepcopy(det_pkl)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in gt_pkl]
        for j in range(len(eval_gt_annos)):
            while(len(eval_gt_annos[j]['gt_boxes_lidar'])<len(eval_gt_annos[j]['name'])):
                new_row = np.array([[0, 0, 0, 0, 0, 0, 0]])
                eval_gt_annos[j]['gt_boxes_lidar'] = np.vstack([eval_gt_annos[j]['gt_boxes_lidar'], new_row])
        for j in range(len(eval_det_annos)):
            x = eval_det_annos[j]['boxes_lidar'][:,0]
            y = eval_det_annos[j]['boxes_lidar'][:,1]
            z = eval_det_annos[j]['boxes_lidar'][:,2]
            mask = (np.sqrt(x*x+y*y+z*z) >= 30) & (np.sqrt(x*x+y*y+z*z) < 45)
            for key in eval_det_annos[j].keys():
                if key!='frame_id':
                    eval_det_annos[j][key]= eval_det_annos[j][key][mask]
        for j in range(len(eval_gt_annos)):
            x = eval_gt_annos[j]['gt_boxes_lidar'][:,0]
            y = eval_gt_annos[j]['gt_boxes_lidar'][:,1]
            z = eval_gt_annos[j]['gt_boxes_lidar'][:,2]
            mask = (np.sqrt(x*x+y*y+z*z) >= 30) & (np.sqrt(x*x+y*y+z*z) < 45)
            for key in eval_gt_annos[j].keys():
                if key!='frame_id':
                    eval_gt_annos[j][key]= eval_gt_annos[j][key][mask]        
        #eval_gt_annos = [copy.deepcopy(info['annos']) for info in gt_pkl]
        print("30~45m:")
        result,str=kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)
        print(result)
    elif(i==3):
        eval_det_annos = copy.deepcopy(det_pkl)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in gt_pkl]
        for j in range(len(eval_gt_annos)):
            while(len(eval_gt_annos[j]['gt_boxes_lidar'])<len(eval_gt_annos[j]['name'])):
                new_row = np.array([[0, 0, 0, 0, 0, 0, 0]])
                eval_gt_annos[j]['gt_boxes_lidar'] = np.vstack([eval_gt_annos[j]['gt_boxes_lidar'], new_row])
        for j in range(len(eval_det_annos)):
            x = eval_det_annos[j]['boxes_lidar'][:,0]
            y = eval_det_annos[j]['boxes_lidar'][:,1]
            z = eval_det_annos[j]['boxes_lidar'][:,2]
            mask = (np.sqrt(x*x+y*y+z*z) >= 40)
            for key in eval_det_annos[j].keys():
                if key!='frame_id':
                    eval_det_annos[j][key]= eval_det_annos[j][key][mask]
        for j in range(len(eval_gt_annos)):
            x = eval_gt_annos[j]['gt_boxes_lidar'][:,0]
            y = eval_gt_annos[j]['gt_boxes_lidar'][:,1]
            z = eval_gt_annos[j]['gt_boxes_lidar'][:,2]
            mask = (np.sqrt(x*x+y*y+z*z) >= 40)
            for key in eval_gt_annos[j].keys():
                if key!='frame_id':
                    eval_gt_annos[j][key]= eval_gt_annos[j][key][mask]        
        #eval_gt_annos = [copy.deepcopy(info['annos']) for info in gt_pkl]
        print("45~inf:")
        result,str=kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)
        print(result)
    else:
        print("Over All:")
        eval_det_annos = copy.deepcopy(det_pkl)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in gt_pkl]
        #eval_gt_annos = [copy.deepcopy(info['annos']) for info in gt_pkl]
        result,str=kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)
        print(result)