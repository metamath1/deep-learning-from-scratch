# coding: utf-8
import numpy as np
import common.conv2d as cv2
from common.functions import *
from common.util import im2col, col2im
import time

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        #mask의 모양
        #Relu1 (100, 30, 24, 24)
        #Relu2 (100, 100)
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    def __init__(self, W, b):
        self.W =W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        # 가중치와 편향 매개변수의 미분
        self.dW = None
        self.db = None

    def forward(self, x):
        # 텐서 대응
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 입력 데이터 모양 변경(텐서 대응)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 손실함수
        self.y = None    # softmax의 출력
        self.t = None    # 정답 레이블(원-핫 인코딩 형태)
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx


class Dropout:
    """
    http://arxiv.org/abs/1207.0580
    """
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None # 합성곱 계층은 4차원, 완전연결 계층은 2차원  

        # 시험할 때 사용할 평균과 분산
        self.running_mean = running_mean
        self.running_var = running_var  
        
        # backward 시에 사용할 중간 데이터
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)
        
        return out.reshape(*self.input_shape)
            
    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
            
        out = self.gamma * xn + self.beta 
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        # 중간 데이터（backward 시 사용）
        self.x = None   
        self.col = None
        self.col_W = None
        
        # 가중치와 편향 매개변수의 기울기
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W
        
        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx

##################################################################################
# 기본적인 convolution 연산을 이용하여 구현한 CONV층 부분 시작
##################################################################################
class Convolution2:
    """
    식빵 형태의 4차원 어레이 형태를 그대로 두고 정말 있는 그대로 컨벌루션해서 순,역전파를 수행하는 방식으로 구현
    단 4차원 어레이에 대한 컨벌루션을 수행함에 있어 for루프 4개가 중첩되는 상황을 피하기위해
    어레이를 적층시켜 데이터 반복을 통해 for 루프를 2개만 사용함
    메모리 낭비도 심하고 속도도 느림(for 4개 중첩보다는 비교할 수 없이 빠르긴하지만....
    """
    def __init__(self, W, b, stride=1, pad=0):
        #stride는 1만 지원
        if self.stride != 1 :
            raise Exception("stride는 1만 지원합니다.")
        
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        # 중간 데이터（backward 시 사용）
        self.x = None   
        
        # 가중치와 편향 매개변수의 기울기
        self.dW = None
        self.db = None

        #
        self.debug = False
        
    def forward(self, x):
        self.x = x
        
        N,  C,  H,  W = x.shape
        FN, C, FH, FW = self.W.shape
        
        x2 = np.concatenate(x.repeat(FN, axis=0), axis=0)[np.newaxis, :]
        W2 = np.concatenate(self.W, axis=0)[np.newaxis, :].repeat(N, axis=0).reshape(1, -1, FH, FW)
        
        if self.debug :
            start = time.time()    
        
        out = cv2.conv2d(x2, W2, sumdepth=False)
        
        if self.debug :
            end = time.time()
            print("forward : %f[s]" %(end-start))
        
        o_shape = out.shape
        
        out = out.reshape(int(o_shape[1]/C), C, *o_shape[2:]).sum(axis=1).reshape(N, FN, *o_shape[2:])+self.b.reshape(FN, 1, 1) 
        
        return out
        
    def backward(self, dout):
        N,  C,  H,  W = self.x.shape
        FN, C, FH, FW = self.W.shape
        x_shape = self.x.shape
        w_shape = self.W.shape 
        
        ##############################
        # dC/dB
        ##############################
        if self.debug :
            start = time.time()    
        
        self.db = np.sum(dout, axis=(2,3)) #한판을 다 더한다
        self.db = np.sum(self.db, axis=0)  #미니배치끼리 다 더한다
        
        if self.debug :
            end = time.time()
            print("dC/dB : %f[s]" %(end-start))
        
        ##############################
        # dC/dW
        ##############################
        if self.debug :
            start = time.time()    
        
        x2 = np.concatenate(self.x.repeat(FN, axis=0), axis=0)[np.newaxis,:]
        dout2 = np.concatenate(dout.repeat(C,axis=1), axis=0)[np.newaxis,:]

        self.dW = cv2.conv2d(x2, dout2, sumdepth=False)
        dw_shape = self.dW.shape
        self.dW = self.dW.reshape(N, -1, *dw_shape[2:]).sum(axis=0).reshape(FN, -1, *dw_shape[2:])
        
        if self.debug :
            end = time.time()
            print("dC/dW : %f[s]" %(end-start))
        
        ##############################
        # W*delta
        ##############################
        if self.debug :
            start = time.time()    
        
        W2 = np.concatenate(np.concatenate(self.W.transpose(1,0,2,3), axis=0)[np.newaxis,:].repeat(N, axis=0), axis=0)[np.newaxis,:]
        dout2 = np.concatenate(dout.repeat(C, axis=0), axis=0)[np.newaxis,:]
        
        dx = cv2.conv2d(W2, dout2, mode='full', flip=True, sumdepth=False)
        dx_shape = dx.shape
        dx = dx.reshape(int(dx_shape[1]/FN), FN, *dx_shape[2:]).sum(axis=1).reshape(N, -1, *dx_shape[2:])
        
        if self.debug :
            end = time.time()
            print("W*delta : %f[s]" %(end-start))
        
        return dx        
        
class Convolution3:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        # 중간 데이터（backward 시 사용）
        self.x = None   
        
        # 가중치와 편향 매개변수의 기울기
        self.dW = None
        self.db = None
        
        self.debug = False

    def forward(self, x):
        self.x = x
        out = cv2.fconv2d(x, self.W, stride=self.stride, pad=self.pad)
        
        return out
        
    def backward(self, dout):
        N,  C,  H,  W  = self.x.shape
        FN, FC, FH, FW = self.W.shape
        DN, DC, DH, DW = dout.shape
        
        ##############################
        # dC/dB
        ##############################
        if self.debug :
            start = time.time()    
            
        #CONV2, CONV3 모두 같음
        self.db = np.sum(dout, axis=(2,3)) #한판을 다 더한다
        self.db = np.sum(self.db, axis=0)  #미니배치끼리 다 더한다
        
        if self.debug :
            end = time.time()
            print("dC/dB : %f[s]" %(end-start))

        
        ##############################
        # dC/dW , x * dout
        ##############################
        if self.debug :
            start = time.time()    
        
        #flatten 
        Xcol = np.asarray(np.hsplit(im2col(self.x.transpose(1,0,2,3), DH, DW, self.stride, self.pad),  N)).reshape(-1, DH*DW)
        Dcol = np.asarray(np.hsplit(im2col(dout.transpose(1,0,2,3)  , DH, DW, self.stride, self.pad), DN)).reshape(-1, DH*DW).T
        #Xcol 과 Dcol을 미니배치 수만큼 돌면서 부분부분 곱함.
        r = int(Xcol.shape[0] / N) 
        c = int(Dcol.shape[1] / N)
        self.dW = np.asarray([np.dot(Xcol[i*r:i*r+r,:], Dcol[:,i*c:i*c+c]) for i in range(N)]).sum(axis=0)
        dWH, dWW = self.dW.shape
        self.dW = self.dW.T.reshape(dWW, -1, FH, FW)
        
        if self.debug :
            end = time.time()
            print("dC/dW : %f[s]" %(end-start))
            
        ##############################
        # W*delta for backpropogation
        ##############################
        if self.debug :
            start = time.time()   
        
        Wt = self.W.transpose(1,0,2,3)
        dx = cv2.fconv2d(dout, Wt, 'full', True, self.stride, self.pad)
        
        if self.debug :
            end = time.time()
            print("W*delta : %f[s]" %(end-start))
        return dx        
        
##################################################################################
# 기본적인 convolution 연산을 이용하여 구현한 CONV층 부분 끝
##################################################################################        

        
class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        return dx
