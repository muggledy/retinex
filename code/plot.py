import cv2
import numpy as np
import re
import matplotlib.pyplot as plt
import os.path
from .tools import get_dir
import time

def get_now_time():
    return time.strftime('%Y.%m.%d.%H.%M.%S',time.localtime(time.time()))

def save_img(img,name,parent_dir=get_now_time()):
    save_dir=os.path.join(get_dir(),'temp/'+parent_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cv2.imwrite(os.path.join(save_dir,name),img)

def contrast_plot(src_img,retinex_imgs,labels=None,save=True):
    '''draw the original image and a series of contrast imgs after using enhanced 
       algorithm, you can set label for each of them'''
    n=len(retinex_imgs)
    h=int(np.sqrt(n))
    w=int(np.ceil(n/h))+1
    if save:
        parent_dir=get_now_time()
    for i,img in enumerate([src_img]+retinex_imgs,0):
        if i==0:
            plt.subplot(h,w,1)
        else:
            plt.subplot(h,w,int(i+np.ceil(i/(w-1))))
        if len(img.shape)==2:
            plt.imshow(img,cmap='gray')
        else:
            plt.imshow(img[:,:,::-1])
        if save:
            save_img(img,'%d.%s.png'%(i,labels[i]),parent_dir)
        if labels!=None:
            plt.xlabel(labels[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()

def demo_viper_with_retinex(viper_path,method,n=20,**kwargs):
    '''specially designed for ./img/VIPeR.v1.0/ images testing, you can pass different 
       image enhancement algorithm to arg method. n is the num of test images, and all 
       kwargs will be passed to func method'''
    dir=os.path.normpath(os.path.join(viper_path,'cam_a/'))
    tt=np.random.choice(os.listdir(dir),n)
    ti=[os.path.join(dir,i) for i in tt]
    dir2=os.path.normpath(os.path.join(viper_path,'cam_b/'))
    camblist=os.listdir(dir2)
    p1=re.compile(r'(\d{3})_.*?.bmp')
    p1r=p1.findall(','.join(tt))
    s=','.join(camblist)
    rr=[]
    for i in p1r:
        p2=re.compile(str(i)+'_'+'.*?.bmp')
        rr.append(p2.findall(s)[0])
    ti2=[os.path.join(dir2,i) for i in rr]
    s=[]
    k=0
    for cam in [ti,ti2]:
        for i in cam:
            img=cv2.imread(i)
            s.append(img)
        for i in range(n):
            ii=i+k
            img=method(s[ii],**kwargs)
            s.append(img)
        k=n*2
    for i,img in enumerate(s):
        plt.subplot(4,n,i+1)
        plt.axis('off')
        plt.imshow(img[:,:,::-1])
    plt.show()
