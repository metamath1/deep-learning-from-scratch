# 『밑바닥부터 시작하는 딥러닝』 (원서 : ゼロから作る Deep Learning)

---

<img src="https://github.com/WegraLee/deep-learning-from-scratch/blob/master/cover_image.jpg" width="480">

---

이 저장소는 『[밑바닥부터 시작하는 딥러닝](http://www.hanbit.co.kr/store/books/look.php?p_code=B8475831198)』(한빛미디어, 2017)의 
7장 CNN부분의 부족한 코드를 추가로 구현한 소스를 제공하고 있습니다.
책의 구현은 이미지와 필터를 im2col 함수로 행렬화 시킨 다음 CONV층을 FC(Fully connected)층 처럼 처리하고 
역전파 역시 FC층 처럼 처리한 다음 col2im 함수로 마지막 컨벌루션 연산을 마무리하고 있습니다.
이에 대해서는 좀 더 자세한 설명이 필요한데 책에서 너무 간단하게 마무리하고 있는 아쉬움에
이 저장소에서 CONV층의 역전파를 직접 미분한 수식을 그대로 구현한 Convolution3 클래스를 제공합니다.

자세한 해설은 https://metamath1.github.io/ 를 참고하세요.


## 파일 구성

|파일 이름 |설명                                                |
|:--        |:--                                                |
|common/conv2d.py           | 컨벌루션의 정의 그대로 컨벌루션을 수행하는 함수 정의 |
|ch07/simple_convnet2.py    | simple_convnet.py에서 구성한 CONVNET과 동일한 구성으로 Convolution3 클래스를 이용한 파일 |
|ch07/train_convnet2.py     | simple_convnet2.py에서 구성한 CONVNET을 훈련 시키는 파일 |
|ch07/deep_convnet.py       | Convolution 클래스를 이용하여 CONV층이 2개인 CONVNET을 구성한 파일  |
|ch07/train_deepconvnet.py  | deep_convnet.py에서 구성한 CONVNET을 훈련 시키는 파일 |
|ch07/deep_convnet2.py      | Convolution3 클래스를 이용하여 CONV층이 2개인 CONVNET을 구성한 파일  |
|ch07/train_deepconvnet2.py | deep_convnet2.py에서 구성한 CONVNET을 훈련 시키는 파일 |

추가된 부분:
CNN 역전파 수식을 그대로 구현하여 작성된 CONV층 클래스인 Convolution3 클래스를 
common/layers.py에 추가

소스 코드 해설은 https://metamath1.github.io/ 에 올리도록 하겠습니다.



## 요구사항
소스 코드를 실행하려면 아래의 소프트웨어가 설치되어 있어야 합니다.

* 파이썬 3.x
* NumPy
* Matplotlib

※ Python은 3 버전을 이용합니다.



## 실행 방법

각 장의 폴더로 이동하여 파이썬 명령을 실행하세요.

```
$ cd ch07
$ python train_convnet.py      #책의 원래 코드를 실행 시킴
$ python train_convnet2.py     #책의 원래 코드와 동일한 구성을 사용하나 CONV층을 Convolution3을 사용한 네트워크를 실행 시킴
$ python train_deepconvnet.py  #CONV층을 2개로 구성하고 원래 클래스인 Convolution을 사용한 네트워크를 실행 시킴
$ python train_deepconvnet2.py #CONV층을 2개로 구성하고 Convolution3을 사용한 네트워크를 실행 시킴
```

## 실행 화면
```
$python train_deepconvnet2.py

epoch:1, train acc:0.093, test acc:0.107, time :3.694805383682251
[============================================================] 100.0% ...PREV.LOSS:  0.064142,CUR.LOSS:  0.051194
epoch:2, train acc:0.962, test acc:0.962, time :629.80482172966
[============================================================] 100.0% ...PREV.LOSS:  0.057680,CUR.LOSS:  0.105737
epoch:3, train acc:0.979, test acc:0.977, time :637.5025508403778
[============================================================] 100.0% ...PREV.LOSS:  0.085682,CUR.LOSS:  0.023816
epoch:4, train acc:0.982, test acc:0.98, time :622.2620611190796
[============================================================] 100.0% ...PREV.LOSS:  0.012926,CUR.LOSS:  0.039342
epoch:5, train acc:0.987, test acc:0.984, time :630.2803926467896
[============================================================] 100.0% ...PREV.LOSS:  0.027622,CUR.LOSS:  0.056404
epoch:6, train acc:0.988, test acc:0.98, time :666.4733510017395
[============================================================] 100.0% ...PREV.LOSS:  0.009774,CUR.LOSS:  0.101239
epoch:7, train acc:0.993, test acc:0.983, time :610.234854221344
[============================================================] 100.0% ...PREV.LOSS:  0.036452,CUR.LOSS:  0.003600
epoch:8, train acc:0.996, test acc:0.985, time :605.3498549461365
[============================================================] 100.0% ...PREV.LOSS:  0.015454,CUR.LOSS:  0.011166
epoch:9, train acc:0.994, test acc:0.988, time :612.0536942481995
[============================================================] 100.0% ...PREV.LOSS:  0.004363,CUR.LOSS:  0.019106
epoch:10, train acc:0.992, test acc:0.986, time :622.7371227741241
[============================================================] 100.0% ...PREV.LOSS:  0.005187,CUR.LOSS:  0.001605
epoch:11, train acc:0.991, test acc:0.986, time :616.2067458629608
[============================================================] 100.0% ...PREV.LOSS:  0.007423,CUR.LOSS:  0.000287
epoch:12, train acc:0.994, test acc:0.986, time :608.5603981018066
[============================================================] 100.0% ...PREV.LOSS:  0.004523,CUR.LOSS:  0.026652
epoch:13, train acc:0.99, test acc:0.977, time :619.7110705375671
[============================================================] 100.0% ...PREV.LOSS:  0.009430,CUR.LOSS:  0.003207
epoch:14, train acc:0.996, test acc:0.987, time :609.3691163063049
[============================================================] 100.0% ...PREV.LOSS:  0.007131,CUR.LOSS:  0.013983
epoch:15, train acc:0.994, test acc:0.989, time :638.6453542709351
[============================================================] 100.0% ...PREV.LOSS:  0.004287,CUR.LOSS:  0.000246
epoch:16, train acc:0.994, test acc:0.987, time :692.6074018478394
[============================================================] 100.0% ...PREV.LOSS:  0.009490,CUR.LOSS:  0.008154
epoch:17, train acc:0.999, test acc:0.986, time :661.1237473487854
[============================================================] 100.0% ...PREV.LOSS:  0.002986,CUR.LOSS:  0.000384
epoch:18, train acc:0.998, test acc:0.986, time :642.7736637592316
[============================================================] 100.0% ...PREV.LOSS:  0.001300,CUR.LOSS:  0.000515
epoch:19, train acc:0.992, test acc:0.988, time :646.7073917388916
[============================================================] 100.0% ...PREV.LOSS:  0.000509,CUR.LOSS:  0.000022
epoch:20, train acc:0.996, test acc:0.99, time :646.9866845607758
[============================================================] 100.0% ...PREV.LOSS:  0.000163,CUR.LOSS:  0.000925
=============== Final Test Accuracy ===============
test acc:0.992
Saved Network Parameters!
```

## 라이선스

이 저장소의 소스 코드는 [MIT 라이선스](http://www.opensource.org/licenses/MIT)를 따릅니다.
비상용뿐 아니라 상용으로도 자유롭게 이용하실 수 있습니다.
