#cython: language_level=3

import time
import numpy as np
cimport numpy as np

np.import_array()

cdef extern from "fgb.h":
	void fast_gauss_blur(double *image,double *result,int w,int h,int channel_num,double sigma,int box_num)

def _fgb(np.ndarray[double,ndim=3,mode="c"] image not None,np.ndarray[double,ndim=3,mode="c"] result not None,double sigma,int box_num=3):
	fast_gauss_blur(<double*>np.PyArray_DATA(image),<double*>np.PyArray_DATA(result),image.shape[2],image.shape[1],image.shape[0],sigma,box_num)

def fastGaussBlur(img,sigma):
	'''can process single-channel or 3-channel image'''
	if len(img.shape)==2:
		img=img[...,None]
	dtype=img.dtype
	img=img.astype('double')
	img=np.rollaxis(img,-1)
	result=np.zeros_like(img)
	img=np.ascontiguousarray(img) #https://stackoverflow.com/questions/22105529/np-ascontiguousarray-versus-np-asarray-with-cython
	result=np.ascontiguousarray(result)
	_fgb(img,result,sigma)
	result=np.squeeze(result).astype(dtype)
	if len(result.shape)==2:
		return result
	return np.rollaxis(result,0,3)