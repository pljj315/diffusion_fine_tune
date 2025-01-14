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
    
def check_params_dct(params, output_path, max_resolution=8000):
    code = 0
    error_str = ""
    
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if not os.path.exists(output_dir):
        code = 4002
        error_str = "output dir %s is not exists" % output_dir
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
    if "height" in params:
        try:
            steps = int(float(params['height']))
        except Exception as e:
            error_str = "height is not int"
            code = 4006
            return {'code': code, 'message': error_str }, None   
    if "width" in params:
        try:
            steps = int(float(params['width']))
        except Exception as e:
            error_str = "width is not int"
            code = 4007
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
             
    return {'code': code, 'message': error_str }, params


def test_server_with_dct(url, dct):
    logger.info('======== Start Server ========')
    start_time = time.time()    
    logger.info('Check params')
    print("dct info:" , dct)

    status_dict, params = check_params_dct(dct['params'], dct['output_path'])
    logger.info('Read port')
    url = url
    #limit size to maxium resolution
    input_h = dct['target_h']
    input_w = dct['target_w']

    stages = {}
    if "specifics" in params:
        specifics = params['specifics']
    if "stages" in params:
        stages = params['stages']
        
    params["output_path"] = dct['output_path']

    payload = {
        "seed": -1, 
        "height": input_h,
        "width": input_w,
        "retry": 1,
        "stages": stages,
    }
    payload.update(params) 
    
    if "prompt_distribution" in params:
        prompt_distribution = params["prompt_distribution"]
        prob = []
        attris = []
        for key in prompt_distribution:
            attris.append(key)
            prob.append(prompt_distribution[key])
        normal_prob = [p / sum(prob) for p in prob]
        value = np.random.choice(attris, p=normal_prob)      
        payload["prompt"] += ", "+value
        
    if "negprompt_distribution" in params:
        negprompt_distribution = params["negprompt_distribution"]
        prob = []
        attris = []
        for key in negprompt_distribution:
            attris.append(key)
            prob.append(negprompt_distribution[key])
        normal_prob = [p/sum(prob) for p in prob]
        value = np.random.choice(attris, p = normal_prob)      
        payload["negative_prompt"] += ", " + value    

    payload['steps'] = int(float(payload['steps']))
    payload['cfg_scale'] = float(payload['cfg_scale'])
    payload['height'] = int(float(payload['height']))
    payload['width'] = int(float(payload['width']))
    
    prompts = []
    imgs = glob.glob('/*')
    imgs.sort()
    img_seg_paths = [(imgs[i],imgs[i+1]) for i in range(0,len(imgs),2)]

    lora2emd_dict = {'xxx':'xxx'}
    model_files = []
    for mode_path in model_files[::-1]:
        pipeline_tmp = mode_path[mode_path.rfind('_')+1:]
        pipeline_idx = os.path.splitext(pipeline_tmp)[0]
        idx_pipeline = pipeline_idx

        train_name = os.path.basename(mode_path[:mode_path.rfind('_')])
        out_path = os.path.join(dct['output_path'], train_name)
        os.makedirs(out_path,exist_ok=True)

        tar_path = mode_path
        for lora_file in lora2emd_dict:
      
          for img_path, seg_path in img_seg_paths:
            for prom_id, tmp_prompt in enumerate(prompts):
                pro_name = os.path.split(os.path.splitext(img_path)[0])[-1]
                # pro_tmp_prompt = tmp_prompt.replace('商品', '')
                pro_tmp_prompt = tmp_prompt.replace('商品', pro_name)
                
                with open(tar_path, "rb") as file:
                    import hashlib
                    m = hashlib.sha256()
                    file.seek(0x100000)
                    m.update(file.read(0x10000))
                    hash_code =  m.hexdigest()[0:8]
                # import pdb
                # pdb.set_trace()
                tmo_img = cv2.imread(img_path)
                mask = cv2.imread(seg_path,cv2.IMREAD_GRAYSCALE)

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
                # leftup_h,leftup_w = random.randint(0,1024-resize_h), random.randint(0,1024-resize_w)
                leftup_h,leftup_w = random.randint(0,200), random.randint(0,1024-resize_w)
                _img = np.zeros([1024,1024,3],np.uint8)
                _img[leftup_h:leftup_h+resize_h,leftup_w:leftup_w+resize_w,:] = target_img
                _img = _img[:,:,::-1]
                _mask = np.zeros([1024,1024],np.uint8)
                _mask[leftup_h:leftup_h+resize_h,leftup_w:leftup_w+resize_w] = target_mask

                re_img = Image.fromarray(_img)
                re_mask = Image.fromarray(_mask)
                tmo_img  = tmo_img[:,:,::-1]
                # re_img = Image.fromarray(tmo_img)
                # re_mask = Image.fromarray(mask)

                h,w,_ = tmo_img.shape
                max_l = max(h,w)
                # _h = (h/(max_l/2048))//64*64
                # _w = (w/(max_l/2048))//64*64
                _h, _w = 1024,1024

                tmp_h = int((h/(max_l/1024))//64*64)
                tmp_w = int((w/(max_l/1024))//64*64)

                payload['target_h'] = _h
                payload['target_w'] = _w
                payload['height'] = _h
                payload['width'] = _w

                
                payload['prompt'] = pro_tmp_prompt
                payload['alwayson_scripts']['controlnet']['args'][0]["image"]={
                                'image':pil_to_base64(re_img),
                                "mask": pil_to_base64(re_mask)  # base64, None when not need
                            }
                # payload['alwayson_scripts']['controlnet']['args'][1]["image"]={
                #                 'image':pil_to_base64(re_mask),
                #                 "mask": None  # base64, None when not need
                #             }

                batch_size =4
                payload['batch_size'] = batch_size
                seed = random.randint(0, 99999999)
                payload['seed'] = seed

                for _model in ['txt2img']:

                    # for inpaint_product_prompt in [pro_name, '']: 
                    for inpaint_product_prompt in ['']:  
                        _weight_list = [ 1.0]
                        payload['alwayson_scripts']['controlnet']['args'][0]['inpaint_product_prompt']= inpaint_product_prompt
                        if inpaint_product_prompt !='':
                            attn_change = 'attn'
                        else:
                            attn_change = ''


                        for tmp_weight in _weight_list:
                            if _model=='img2img':
                                denoise_list = [0.35]
                            else:
                                denoise_list = [None]
                            for denoise in denoise_list:
                #             for tmp_weight in [0.6]:
                                if denoise:
                                    # xxx=encode_file_to_base64(img_paths[prom_id])
                                    xxx=pil_to_base64(tmo_img)
                                    payload['init_images'] = [xxx]
                                    # payload['init_images'] = [img_paths[prom_id]]

                                    payload['denoising_strength'] = denoise
                                payload['alwayson_scripts']['controlnet']['args'][0]['weight'] = tmp_weight
                                payload['alwayson_scripts']['controlnet']['args'][0]['model'] = train_name+'_'+idx_pipeline+' ['+hash_code+']'
                                # print(train_name+'_'+idx_pipeline+' ['+hash_code+']')
                                payload_json = json.dumps(payload)

                                ttt = copy.deepcopy(payload)
                                ttt['init_images'] = []
                                ttt['alwayson_scripts']['controlnet']['args'][0]["image"] = []
                                # ttt['alwayson_scripts']['controlnet']['args'][1]["image"] = []
                                # print(ttt)
                                img_name = os.path.split(os.path.splitext(img_path)[0])[1]
                                cur_seed = ttt['seed']
                                save_name = os.path.join(out_path,f'{idx_pipeline}_{_model}_{prom_id}_{img_name}_{str(cur_seed)}_{str(attn_change)}.jpg')
                                try:
                                    if denoise:
                                        save_name = os.path.join(out_path,'%s_%s_%d_%s_%s_%s.jpg'%(idx_pipeline, _model, prom_id, img_name, str(tmp_weight), str(denoise)) )
                                    if os.path.isfile(save_name):
                                        continue
                                    if os.path.isfile(os.path.splitext(save_name)[0]+'_0.jpg'):
                                        continue
                                    logger.info('Request %s'%_model)
                                    response = requests.post(url=f'{url}/sdapi/v1/{_model}', data=payload_json).json()
                                    
                                    images = []
                                    for __i in range(batch_size):
                                        result = response['images'][__i]
                                        image = Image.open(io.BytesIO(base64.b64decode(result.split(",", 1)[0])))
                    #                     if input_w <= dct['target_w']:
                    #                         image =image.resize((dct['target_w'], dct['target_h']), Image.BILINEAR)
                    #                     else:
                    #                         image =image.resize((dct['target_w'], dct['target_h']), Image.LANCZOS)
                                        images.append(image)
                                    grid = make_image_grid(images, rows=1, cols=batch_size)
                                    grid.save(os.path.splitext(save_name)[0]+'.jpg')
                                    print('---------save::', os.path.splitext(save_name)[0]+'.jpg')

                                except Exception as e:
                                    print(e)
                                    code = 5001
                                    logger.error(f'Cannot request to txt2img!')   
                                    with open(dct['output_json'], 'w') as f:
                                        json.dump({'code' : code, 'message': repr(e)}, f)  
                
    # logger.info(response)
    logger.info('Time elapsed: {}'.format(time.time() - start_time))
    
    with open(dct['output_json'], 'w') as f:
        json.dump({'code' : 0, 'message': "SUCCESS"}, f)  
    
    logger.info('======== End Server ========')
    return True


def args_to_dct(args):
    dct = {}
    dct['target_h']     = args.target_h
    dct['target_w']     = args.target_w
    dct['output_path']  = args.output_path
    params = json.load(open(args.params_path,"r")) 
    dct['params']       = params
    dct['output_json']  = args.output_json
    # dct['port']         = args.port
    return dct

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_h", type=int, default=1024)
    parser.add_argument("--target_w", type=int, default=1024)
    parser.add_argument("--output_path",  default="*",help="output image path", )
    parser.add_argument("--params_path", type=str, help="params json",  default="params_img2img_product.json")
    parser.add_argument("--output_json", type=str, help="output status json",  default="output_status.json")
    parser.add_argument("--port", type=str, help="port",  default="7965")
    args = parser.parse_args()
    dct = args_to_dct(args)
    url = "http://127.0.0.1:%s"%args.port
    if test_server_with_dct(url, dct):
        print("test_server_ok, return images")
