import argparse
import os,sys
import json
import cv2
import torch
import torchvision.transforms as transforms 
import lpips
import numpy as np


def load_gt_image(path):
    img_bgra = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = 255 * np.ones((img_bgra.shape[0],img_bgra.shape[1],3), dtype=np.uint8)
    img[img_bgra[:,:,3] >=255,:] = img_bgra[img_bgra[:,:,3] >=255,0:3]
    return img

def parse_args():
    args = argparse.ArgumentParser(description='Computes LPIPS loss.')
    args.add_argument('data_path', type=str, help='path to the data directory where transforms_test.json file exists.')
    args.add_argument('results_path', type=str, help='folder where results are saved')
    return args.parse_args()

def check_if_dir_exists(path_:str):
    if not os.path.exists(path_):
        print('[Error] Path doesnot exist : {}'.format(path_))
        sys.exit(-1)

    if not os.path.isdir(path_):
        print('[Error] Path is not a directory. {}'.format(path_))
        sys.exit(-1)

def check_if_file_exists(path_:str):
    if not os.path.exists(path_):
        print('[Error] File doesnot exist : {}'.format(path_))
        sys.exit(-1)

    if not os.path.isfile(path_):
        print('[Error] Path is not a File. {}'.format(path_))
        sys.exit(-1)

def read_json_file(file_name:str):
    """Reads data from the json file

    Args:
        file_name (str): path of the input json file

    Returns:
        Any: data in the json file
    """
    with open(file_name,'r') as f:
        data = json.load(f)
        return data

def im2tensor(image, imtype=np.uint8, cent=1., factor=255./2.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))
def compute_lpips(args):
    check_if_dir_exists(args.data_path)
    check_if_dir_exists(args.results_path)

    print('='*10,'\t', args.results_path, '='*10)
    test_file_path = os.path.join(args.data_path, 'transforms_test.json')
    check_if_file_exists(test_file_path)
    
    transform = transforms.Compose([transforms.PILToTensor()])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_ = read_json_file(test_file_path)

    loss_fn = lpips.LPIPS(net='vgg').to(device)

    level_wise_lpips = {}
    for i,file_ in enumerate(data_['frames']):
        gt_img_tensor =  im2tensor( load_gt_image(os.path.join(args.data_path, file_['file_path'])+'.png')[:,:,::-1] )
        pred_img_tensor =  im2tensor( cv2.imread(os.path.join(args.results_path, 'color_{:03d}.png'.format(i)))[:,:,::-1] )

        gt_img_tensor = gt_img_tensor.to(device)
        pred_img_tensor = pred_img_tensor.to(device)

        lpips_value = (loss_fn.forward(gt_img_tensor, pred_img_tensor)).detach().cpu().numpy()

        camera_level = file_['camera_level']
        if camera_level not in level_wise_lpips.keys():
            level_wise_lpips[camera_level] = []
        print('Images : {} \t : {:4f}'.format(i,lpips_value[0,0,0,0] ))
        level_wise_lpips[camera_level].append(lpips_value[0,0,0,0])

    total = 0.
    denom = 0
    for level in level_wise_lpips.keys():
        average = sum(level_wise_lpips[level])/len(level_wise_lpips[level])
        total += average *  len(level_wise_lpips[level])
        denom += len(level_wise_lpips[level])
        print('Level {} : {:.6f}'.format(level,average ))
    print('Total : {:.6f}'.format(total/denom))
    
if __name__=="__main__":
    compute_lpips(parse_args())

