# conv2d.py
# Released under a MIT license
import numpy as np
from common.util import im2col, col2im
import time

def conv2d(Im, W, mode='valid', flip=False, sumdepth=True) :
    """
    Calculate 4D tensor convolution and corellation
    
    INPUT
    Im   : Images  numpy array [batch_no , im_channel, im_row, im_col]
    W    : Filters numpy array [filter_no, w_channel,   w_row,  w_col]
    mode : [valid or full]
    filp : [False or True] If True W is rotated by 180 degree
    
    CAUTION
        if sumdepth == False, batch_no and filter_no must be 1, i.e input array is [1, ch, row, col]
        and the im_channel has to be the same as the w_channel
    
    OUTPUT
    if sumdepth == True
        numpy array [batch_no, filter_no, im_row-w_row+1, im_col-w_col+1]
    else    
        
    """
    if mode not in ['valid', 'full'] :
        raise Exception("The mode value must be either valid or full.")

    if flip not in [True, False] :
        raise Exception("The flip value must be either True or False.")

    
    if(flip == True) :
        #      행뒤집기      열뒤집기  == rot 180, 새로 만드는 행렬도 float32형으로 해야 한다.
        W = (W[:,:,::-1,:])[:,:,:,::-1].astype(np.float32)
    
    w_shape  = W.shape
    
    if(mode == 'full') :
        #calc. padding size
        pad_c, pad_r = w_shape[3]-1 , w_shape[2]-1
         
        #im padding , 새로 만드는 행렬도 float32형으로 해야 한다.
        pad_im = np.zeros((Im.shape[0], Im.shape[1], Im.shape[2]+pad_r*2, Im.shape[3]+pad_c*2)).astype(np.float32)
        pad_im[:, :, pad_r:-pad_r, pad_c:-pad_c] =  Im
        Im = pad_im
    
    im_shape = Im.shape
    
    #calc. output size
    row = im_shape[2]-w_shape[3]+1
    col = im_shape[2]-w_shape[3]+1    
  
    #convolution할때 axis=1(깊이 방향)으로도 다 더하는 모드(일반적인 convolution)
    if sumdepth == True:
        out = np.asarray([np.multiply(Im[b][:, r:r+w_shape[2], c:c+w_shape[3]], W[f]).sum() 
                            for b in range(im_shape[0]) for f in range(w_shape[0]) \
                            for r in range(row) for c in range(col)]).reshape(im_shape[0], w_shape[0], row, col)              
    #convolution할때 깊이 방향으로는 더하지 않고 행과 열방향으로만 더하는 모드
    else :
        out = np.asarray([np.multiply(Im[b][:, r:r+w_shape[2], c:c+w_shape[3]], W[f]).sum(axis=(1,2)) 
                            for b in range(im_shape[0]) for f in range(w_shape[0]) \
                            for r in range(row) for c in range(col)]).reshape(row, col, -1).transpose(2,0,1)[np.newaxis,:]              
    
    #print( "DEBUG mode:{0:5s}, flip:{1:2b}, loop:{2:4d}, time:{3}, Im:{4}, W:{5}".format(mode, flip, row*col, end-start, Im.shape, W.shape) )
    return out
    
def fconv2d(Im, W, mode='valid', flip=False, stride=1, pad=0) :
    """
    Calculate 4D tensor convolution and corellation by using im2col
    
    INPUT
    Im     : Images  numpy array [batch_no , ch, im_row, im_col]
    W      : Filters numpy array [filter_no, ch,  w_row,  w_col]
    mode   : [valid or full]
    filp   : [False or True] If True W is rotated by 180 degree
    stride : 
    pad    :
    OUTPUT
    numpy array [batch_no, filter_no, 1 + int((IH + 2*pad - FH) / stride), 1 + int((IW + 2*pad - FW) / stride)]
    """
    
    if mode not in ['valid', 'full'] :
        raise Exception("The mode value must be either valid or full.")

    if flip not in [True, False] :
        raise Exception("The flip value must be either True or False.")

    
    if(flip == True) :
        #      행뒤집기      열뒤집기  == rot 180, 새로 만드는 행렬도 float32형으로 해야 한다.
        W = (W[:,:,::-1,:])[:,:,:,::-1].astype(np.float32)
    
    FN, C, FH, FW = W.shape
    
    if(mode == 'full') :
        #calc. padding size
        pad_c, pad_r = FW-1 , FH-1
         
        #im padding , 새로 만드는 행렬도 float32형으로 해야 한다.
        pad_im = np.zeros((Im.shape[0], Im.shape[1], Im.shape[2]+pad_r*2, Im.shape[3]+pad_c*2)).astype(np.float32)
        pad_im[:, :, pad_r:-pad_r, pad_c:-pad_c] =  Im
        Im = pad_im
        
        #ignore pad param
        pad = 0
      
    N, C, IH, IW = Im.shape

    #calc. output size
    out_h = 1 + int((IH + 2*pad - FH) / stride)
    out_w = 1 + int((IW + 2*pad - FW) / stride)

    col = im2col(Im, FH, FW, stride, pad).astype(np.float32)
    col_W = W.reshape(FN, -1).T

    out = np.dot(col, col_W)
    out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
    return out
    
if __name__ == "__main__" :
    """
    conv2d.xlsx 에 있는 예제 테스트
    """
    Im = np.arange(100).reshape(2, 2, 5, 5)
    W = np.arange(36).reshape(2, 2, 3, 3)
    
    #valid
    start = time.time()    
    O = fconv2d(Im, W)
    print(O)
    end = time.time()
    print("%f[s]" %(end-start))
    
    #rot180, valid
    start = time.time()    
    O = fconv2d(Im, W, flip=True)
    print(O)
    end = time.time()
    print("%f[s]" %(end-start))
    
    #rot180, full
    start = time.time()    
    O = fconv2d(Im, W, mode='full', flip=True)
    print(O)
    end = time.time()
    print("%f[s]" %(end-start))
    
