#!/usr/bin/python
__author__ = 'dereyly'
import sys
#sys.path.append('/usr/lib/python2.7/dist-packages')
import time
import numpy as np
import cv2

#import scipy.io as sio

# manual set path to caffe
sys.path.insert(0,'/home/dereyly/progs/caffe-elu/python')
import caffe

# global variable
face_sz=(224,224)

# Path to classificators and detectors
MODEL_FILE='proto/VGG_FACE_deploy_new.prototxt'
#download source http://www.robots.ox.ac.uk/~vgg/software/vgg_face/
#link= http://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_caffe.tar.gz
PRETRAINED='data/VGG_FACE.caffemodel'
#PRETRAINED='/home/dereyly/data/vgg_face_caffe/VGG_FACE.caffemodel'

mean_val=np.array([129.1863,104.7624,93.5940])

dir_lfw='/home/dereyly/ImageDB/FACE/lfw/'

is_norm_dist = False






def test_lfw(fname_test):
    #init caffe
    caffe.set_mode_gpu()
    net = caffe.Net(MODEL_FILE,
                PRETRAINED,
                caffe.TEST)
    #reshape batch size
    batch_size=50
    _shape=[batch_size,3,face_sz[0],face_sz[1]]
    net.blobs['data'].reshape(_shape[0],_shape[1],_shape[2],_shape[3])

    #plain init
    k=0
    acc=0.0
    err=0.0
    start = time.time()
    fnames=[]
    fnames.append('')
    fnames.append('')
    feats=[]
    feats.append([])
    feats.append([])
    N=6000 #-- number of probs for ROC
    roc=np.zeros((N,2))
    is_same=False
    count = 0
    is_batch_ready = False

    with open(fname_test, 'r') as f:
        for line in f.readlines():
            if k>=6000:
                break
            data=line.split('\t')
            if len(data)<3:
                continue
            if len(data)==3:
                fnames[0]='%s/%s_%04d.jpg' % (data[0],data[0],int(data[1]))
                fnames[1]='%s/%s_%04d.jpg' % (data[0],data[0],int(data[2]))
                is_same = True
            if len(data)==4:
                fnames[0]='%s/%s_%04d.jpg' % (data[0],data[0],int(data[1]))
                fnames[1]='%s/%s_%04d.jpg' % (data[2],data[2],int(data[3]))
                is_same = False

            for i,fname in enumerate(fnames):
                fname=dir_lfw+fname
                img = cv2.imread(fname)
                img2=img[25:205,35:215,:]
                #img2=img[18:242,18:242,:]
                img2=cv2.resize(img2,face_sz,interpolation = cv2.INTER_LINEAR)

                for j in range(1): #Oversample
                    im=np.copy(img2)
                    im = im.astype(np.float32, copy=False)
                    im -= mean_val
                    im=np.transpose(im,(2,0,1))
                    net.blobs['data'].data[count]=im
                    count+=1

                    if count>=batch_size:
                        is_batch_ready=True
                        net.forward()
                        feats=net.blobs['fc7'].data

                        if is_norm_dist:
                            feats=feats/np.sum(feats**2)
                            feats=np.sign(feats)*np.sqrt(np.abs(feats))

                        count=0

            if not is_batch_ready:
                continue
            else:
                is_batch_ready = False

            for z in range(0,batch_size,2):
                dist=np.sum((feats[z]-feats[z+1])**2)
                if is_norm_dist:
                    dist*=100.0
                else:
                    dist/=4096 #impirecal const set dist from 0 to 10

                for i in range(0,N):
                    th=(10.0*i)/N
                    if dist<th and is_same:
                        roc[i,0]+=1
                    if dist<th and not is_same:
                        roc[i,1]+=1

                # just for display accuracy
                if dist<5.0 and is_same:
                        acc+=1
                if dist<5.0 and not is_same:
                        err+=1
                k+=1

            end = time.time()
            sys.stderr.write('{0}  time, {1}  --- acc=  {2} err={3} ---  {4}'.format(
                k,(end - start)/k,acc/k,err/k,'\r') )
            #print k,(end - start)/k,acc/k,err/k
        roc/=3000.0
        mAP=np.max((roc[:,0]+1-roc[:,1])/2)
        print mAP
        #uncoment to save in mat
        #sio.savemat('roc_vgg.mat', {'roc':roc})





test_lfw('/home/dereyly/ImageDB/FACE/lfw/pairs.txt')