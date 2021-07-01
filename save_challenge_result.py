import SimpleITK as sitk
import numpy as np
import glob

data_type = ".nii"
data_path_012 = '/lrde/home/zz/Heart/atriaseg2018/MyoPS2020/result_segmentation/result_012_/test'
data_path_012_image = '/lrde/home/zz/Heart/atriaseg2018/MyoPS2020/test20'
data_012= glob.glob(data_path_012+"/pre/*")
for data_name in data_012:        
    name=data_name[data_name.rindex("/")+0:-4]
    print(name)
    pre=sitk.ReadImage(data_path_012+"/pre_restore"+name+data_type) 
    image = sitk.ReadImage(data_path_012_image+name+"_C0.nii.gz") 
    pre = sitk.GetArrayFromImage(pre)
    pre[pre==1] = 1220
    pre[pre==2] = 2221
    pre = pre.astype('int16')

    last_ouput = sitk.GetImageFromArray(pre)
    last_ouput.CopyInformation(image)
    sitk.WriteImage(last_ouput, '/lrde/home/zz/Heart/atriaseg2018/MyoPS2020/test/'+str(name)+'_seg.nii.gz')

