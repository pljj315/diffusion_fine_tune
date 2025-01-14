import shutil
import glob
import os

model_files = []
model_files += glob.glob('/unet-*/*.safetensors')

for mode_path in model_files:
    idx_pipeline = os.path.basename(os.path.dirname(mode_path))
    idx_pipeline = idx_pipeline[idx_pipeline.find('-')+1:]
    train_name = os.path.basename(os.path.dirname(os.path.dirname(mode_path)))

    tar_path = os.path.join('stable-diffusion-api/models/Stable-diffusion', train_name+'_'+ idx_pipeline+'.safetensors')
    
    # shutil.move(mode_path, tar_path)
    os.system('ln -s %s %s'%(mode_path, tar_path))

    print(mode_path, tar_path)

    # shutil.move(tar_path,mode_path )

# ln -s '原文件' '软链接目标文件'
# 注意软链接的删除操作、取消链接操作的不同   