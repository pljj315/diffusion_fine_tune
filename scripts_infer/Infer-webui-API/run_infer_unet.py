import json
import requests
import copy
from PIL import Image
import time
import cv2
import argparse
from logger import logger
import os
import shutil
import numpy as np
import base64
import io
import glob
import random
from diffusers.utils import  make_image_grid

def pil_to_base64(pil_image):
    with io.BytesIO() as stream:
        pil_image.save(stream, "PNG", pnginfo=None)
        base64_str = str(base64.b64encode(stream.getvalue()), "utf-8")
        return "data:image/png;base64," + base64_str

def encode_file_to_base64(path):
    with open(path, 'rb') as file:
        return base64.b64encode(file.read()).decode('utf-8')
    
def check_params_dct(params, unet_output_path, max_resolution=8000):
    code = 0
    error_str = ""
    
    unet_output_path = os.path.dirname(os.path.abspath(unet_output_path))
    if not os.path.exists(unet_output_path):
        code = 4002
        error_str = "unet_output_path %s is not exists" % unet_output_path
        return {'code': code, 'message': error_str }, None
               
    if "steps" in params:
        try:
            steps = int(float(params['steps']))
        except Exception as e:
            error_str = "steps is not int"
            code = 4003  
            return {'code': code, 'message': error_str }, None   
    if "cfg_scale" in params:
        try:
            cfg_scale = float(params['cfg_scale'])
        except Exception as e:
            error_str = "cfg_scale is not float"
            code = 4004 
            return {'code': code, 'message': error_str }, None     
    if "resolution" in params:
        try:
            steps = int(float(params['resolution']))
        except Exception as e:
            error_str = "resolution is not int"
            code = 4006
            return {'code': code, 'message': error_str }, None     
    if "seed" in params:
        if type(params['seed']) != int:
            error_str = "seed is not int"
            code = 4007
            return {'code': code, 'message': error_str }, None
    if "retry" in params:
        if type(params['retry']) != int:
            error_str = "retry is not int"
            code = 4007
            return {'code': code, 'message': error_str }, None                
    if "prompt" in params:
        if type(params['prompt']) != str:
            error_str = "prompt is not string"
            code = 4008
            return {'code': code, 'message': error_str }, None   
    if "negative_prompt" in params:
        if type(params['negative_prompt']) != str:
            error_str = "negative_prompt is not string"
            code = 4009
            return {'code': code, 'message': error_str } , None
    if "stages" in params:
        if len(params['stages']) > 0:
            denoising_strengths = params['stages']['denoising_strengths']
            step_ratios = params['stages']['step_ratios']
            if len(denoising_strengths) != len(step_ratios):
                error_str = "the length of denoising_strengths is not equal to step_ratios"
                code = 4010
                return {'code': code, 'message': error_str } , None       
             
    return {'code': code, 'message': error_str }


def test_server_with_dct(url_unet,  params_unet,  output_json):
    logger.info('======== Start Server ========')
    start_time = time.time()    
    logger.info('Check params')
    
    os.makedirs( params_unet['output_path'], exist_ok=True)
    status_dict = check_params_dct(params_unet,  params_unet['output_path'], params_unet['output_path'])
    if status_dict == 0:
        raise ValueError("参数定义有问题")
        
    # 定义API的payload参数：
    payload_unet = { "retry": 1, "stages": {}}
    payload_unet.update(params_unet) 
    payload_unet['seed'] = int(float(params_unet['seed']))
    payload_unet['height'] = int(float(params_unet['height']))
    payload_unet['width'] = int(float(params_unet['width']))
    payload_unet['steps'] = int(float(params_unet['steps']))
    payload_unet['cfg_scale'] = float(params_unet['cfg_scale'])
    payload_unet['batch_size'] = int(float(params_unet['batch_size']))
    payload_unet['target_h'] = int(float(params_unet['target_h']))
    payload_unet['target_w'] = int(float(params_unet['target_w']))
    
    img_seg_paths = []

    prompts = []

    step = 18500
    unet_model_files = f'***.safetensors'

    unet_output_path = os.path.join( params_unet['output_path'], f"unet_inpaint_{step}_ori_bg")
    os.makedirs(unet_output_path,exist_ok=True)
    print('unet_output_path:',unet_output_path)


    for img_path, seg_path in img_seg_paths:
        for prom_id, prompt in enumerate(prompts):
            with open(unet_model_files, "rb") as file2:
                import hashlib
                m2 = hashlib.sha256()
                file2.seek(0x100000)
                m2.update(file2.read(0x10000))
                hash_code2 =  m2.hexdigest()[0:8]

            tmo_img = cv2.imread(img_path)
            mask = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
            hw=np.argwhere(mask>0)
            min_h,max_h = np.min(hw[:,0]), np.max(hw[:,0])
            min_w,max_w = np.min(hw[:,1]), np.max(hw[:,1])
            crop_mask = mask[min_h:max_h, min_w:max_w]
            crop_img = tmo_img[min_h:max_h, min_w:max_w,:]
            crop_max = max(max_h-min_h,max_w-min_w)
            target_size = random.randint(300,400)
            ratio = target_size/crop_max
            target_mask = cv2.resize(crop_mask,None,fx=ratio,fy=ratio)
            target_img = cv2.resize(crop_img,None,fx=ratio,fy=ratio)
            resize_h, resize_w, _ = target_img.shape
            leftup_h,leftup_w = random.randint(0,200), random.randint(0,1024-resize_w)
            _img = np.zeros([1024,1024,3],np.uint8)
            _img[leftup_h:leftup_h+resize_h,leftup_w:leftup_w+resize_w,:] = target_img
            _img = _img[:,:,::-1]
            _mask = np.zeros([1024,1024],np.uint8)
            _mask[leftup_h:leftup_h+resize_h,leftup_w:leftup_w+resize_w] = target_mask

            re_img = Image.fromarray(_img)
            re_mask = Image.fromarray(_mask)

            _model = "img2img" 
            payload_unet['prompt'] = prompt
            re_mask.save('tmp_mask.png')
            re_img.save('tmp_img.png')
            
            payload_unet['mask'] = 'tmp_mask.png'
            payload_unet['init_images'] = ['tmp_img.png']

            payload_unet['model'] = f"jh_inpaint_{step}" + ' ['+hash_code2+']'
            payload_unet['denoising_strength'] = params_unet["denoising_strength"]

            payload_unet_json = json.dumps(payload_unet)

            ttt = copy.deepcopy(payload_unet)
            ttt['init_images'] = []
            ttt['mask'] = []
            print("参数：",ttt)
            try:
                img_name = os.path.split(os.path.splitext(img_path)[0])[1]
                save_name = os.path.join(unet_output_path ,'%s_%d_%s_%s.jpg'%(_model, prom_id, img_name, str(params_unet["denoising_strength"])) )

                logger.info('Request %s'%_model)           
                response = requests.post(url=f'{url_unet}/sdapi/v1/{_model}', data=payload_unet_json).json()
                for __i in range(1):
                    result = response['images'][__i]
                    image = Image.open(io.BytesIO(base64.b64decode(result.split(",", 1)[0])))
                    path = os.path.splitext(save_name)[0]+'_%d.jpg'%__i
                    images_list = [re_img, re_mask, image]
                    grid = make_image_grid(images_list, rows=1, cols=len(images_list))
                    grid.save(path)
                print('-----------------save img %s'%save_name)

            except Exception as e:
                print(e)
                code = 5001
                logger.error(f'Cannot request to txt2img!')   
                with open(output_json, 'w') as f:
                    json.dump({'code' : code, 'message': repr(e)}, f)  
            
                    
    # logger.info(response)
    logger.info('Time elapsed: {}'.format(time.time() - start_time))
    
    with open(output_json, 'w') as f:
        json.dump({'code' : 0, 'message': "SUCCESS"}, f)  
    
    logger.info('======== End Server ========')

    return True



if __name__ == "__main__":

    url_unet = "http://127.0.0.1:7965"
    params_unet = json.load(open("params_unet.json","r")) 
    output_json = "output_status.json"

    if test_server_with_dct(url_unet,  params_unet,  output_json):
        print("test_server_ok, return images")
