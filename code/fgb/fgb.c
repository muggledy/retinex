#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

/*
  Implementation of one Fast Gaussian Blur(in linear time) 
  based on article of Ivan Kuckir: http://blog.ivank.net/fastest-gaussian-blur.html
  Reference: Fast Almost-Gaussian Filtering(DOI Bookmark:10.1109/DICTA.2010.30)
  See also https://stackoverflow.com/questions/98359/fastest-gaussian-blur-implementation
  In my test, it is comparable to cv2's filter2D(filter image with a gaussian kernel), 
  even faster. But i don't think it can overcome some optimized parallel scheme. Lastly, 
  this code has shortcoming, you can not set too big sigma without fixing, so now it is 
  just a demo for reference only
*/

int* boxes_for_gauss(double sigma,int n){
	int *bxs=(int*)malloc(sizeof(int)*n);
	int wl,wu,m;
	wl=floor(sqrt((12*sigma*sigma/n)+1));
	if(wl%2==0) wl--;
	wu=wl+2;
	m=round((12*sigma*sigma-n*wl*wl-4*n*wl-3*n)/(-4*wl-4));
	for(int i=0;i<n;i++)
		bxs[i]=i<m?wl:wu;
	return bxs;
}

void box_blur_h(double *channel,double *result,int w,int h,int r){
	double iarr=(double)1/(2*r+1),fv,lv,val;
	int ti,li,ri;
	for(int i=0;i<h;i++){
		ti=i*w;
		li=ti;
		ri=ti+r;
		fv=channel[ti];
		lv=channel[ti+w-1];
		val=(r+1)*fv;
		for(int j=0;j<r;j++) val+=channel[ti+j];
		for(int j=0;j<=r;j++){
			val+=channel[ri++]-fv;
			result[ti++]=val*iarr;
		}
		for(int j=r+1;j<w-r;j++){
			val+=channel[ri++]-channel[li++];
			result[ti++]=val*iarr;
		}
		for(int j=w-r;j<w;j++){ //too big sigma, too big r, if r>w/2, it will collapse! I have added a check of r
			val+=lv-channel[li++];
			result[ti++]=val*iarr;
		}
	}
}

void box_blur_t(double *channel,double *result,int w,int h,int r){
	double iarr=(double)1/(2*r+1),fv,lv,val;
	int ti,li,ri;
	for(int i=0;i<w;i++){
		ti=i;
		li=ti;
		ri=ti+r*w;
		fv=channel[ti];
		lv=channel[ti+w*(h-1)];
		val=(r+1)*fv;
		for(int j=0;j<r;j++) val+=channel[ti+j*w];
		for(int j=0;j<=r;j++){
			val+=channel[ri]-fv;
			result[ti]=val*iarr;
			ri+=w;
			ti+=w;
		}
		for(int j=r+1;j<h-r;j++){
			val+=channel[ri]-channel[li];
			result[ti]=val*iarr;
			li+=w;
			ri+=w;
			ti+=w;
		}
		for(int j=h-r;j<h;j++){
			val+=lv-channel[li];
			result[ti]=val*iarr;
			li+=w;
			ti+=w;
		}
	}
}

void box_blur(double *channel,double *result,int w,int h,int r){
	if(!(r>=0&&r<=(int)(w-2)/2) || !(r>=0&&r<=(int)(h-2)/2)){ //check of r
		printf("too big sigma, cut off!\n");
		r=min((int)(w-2)/2,(int)(h-2)/2);
	}
	for(int i=0;i<w*h;i++) result[i]=channel[i];
	box_blur_h(result,channel,w,h,r);
	box_blur_t(channel,result,w,h,r);
}

void fast_gauss_blur_gray(double *channel,double *result,int w,int h,int *bxs,int box_num){
	for(int i=0;i<box_num;i++){
		box_blur(channel,result,w,h,(*(bxs+i)-1)/2);
	}
}

/* Main Function */
void fast_gauss_blur(double *image,double *result,int w,int h,int channel_num,double sigma,int box_num){
	int *bxs;
	bxs=boxes_for_gauss(sigma,box_num);
	for(int i=0;i<channel_num;i++){
		fast_gauss_blur_gray(image+w*h*i,result+w*h*i,w,h,bxs,box_num);
	}
}

int main(){
	
	return 0;
}