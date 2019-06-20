# CS 181B HW7 Andrew Lai 9519687
Python 3 implementation of a convolutional neural network CS181B W19.

Run `python prog7.py` to run the training script, and `python prog72.py` to run the prediction script; I haven't run this on CSIL, so I'm not completely sure if it works there - this is designed to be with the CUDA/GPU version of TensorFlow compiled, using an Nvidia GTX 1080 Ti. For the record, here was the final output/best accuracy with 69 epochs:

```
Epoch 69 - accuracy: 78.33% (7833/10000) - time: 00:00:03.79
###########################################################################################################
Best accuracy pre session: 78.66, time: 00:04:29.43
```

Otherwise, the below is formatted as asked in the assignment description, i.e. 10 epochs, with training/testing accuracy included for each epoch. prog7.py and prog72.py respectively output to `train_out.txt` and `predict_out.txt`.

`train_out.txt`
```
2019-03-14 16:11:44.713170: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2019-03-14 16:11:44.815895: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:897] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2019-03-14 16:11:44.816453: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1405] Found device 0 with properties: 
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:22:00.0
totalMemory: 10.91GiB freeMemory: 10.03GiB
2019-03-14 16:11:44.816468: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-03-14 16:11:45.008606: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-03-14 16:11:45.008643: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-03-14 16:11:45.008649: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-03-14 16:11:45.008890: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 9695 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:22:00.0, compute capability: 6.1)

Initializing variables.

Epoch: 1/10

Global step:     1 - [>-----------------------------]   0% - acc: 0.0781 - loss: 2.3034 - 352.5 sample/sec
Global step:    11 - [=>----------------------------]   5% - acc: 0.1680 - loss: 2.2681 - 17751.3 sample/sec
Global step:    21 - [==>---------------------------]  10% - acc: 0.2852 - loss: 2.1879 - 18055.2 sample/sec
Global step:    31 - [====>-------------------------]  15% - acc: 0.3086 - loss: 2.1472 - 17113.9 sample/sec
Global step:    41 - [=====>------------------------]  20% - acc: 0.3086 - loss: 2.1453 - 17658.5 sample/sec
Global step:    51 - [=======>----------------------]  26% - acc: 0.3164 - loss: 2.1113 - 17015.2 sample/sec
Global step:    61 - [========>---------------------]  31% - acc: 0.3594 - loss: 2.1099 - 17735.5 sample/sec
Global step:    71 - [==========>-------------------]  36% - acc: 0.4062 - loss: 2.0634 - 16956.9 sample/sec
Global step:    81 - [===========>------------------]  41% - acc: 0.3672 - loss: 2.0878 - 17018.1 sample/sec
Global step:    91 - [=============>----------------]  46% - acc: 0.3672 - loss: 2.0907 - 17849.0 sample/sec
Global step:   101 - [==============>---------------]  51% - acc: 0.3945 - loss: 2.0517 - 17204.9 sample/sec
Global step:   111 - [================>-------------]  56% - acc: 0.4258 - loss: 2.0307 - 17726.7 sample/sec
Global step:   121 - [=================>------------]  61% - acc: 0.3594 - loss: 2.1026 - 17026.2 sample/sec
Global step:   131 - [===================>----------]  66% - acc: 0.3477 - loss: 2.1058 - 18220.9 sample/sec
Global step:   141 - [====================>---------]  71% - acc: 0.4102 - loss: 2.0313 - 17361.3 sample/sec
Global step:   151 - [======================>-------]  77% - acc: 0.4375 - loss: 2.0419 - 17719.1 sample/sec
Global step:   161 - [=======================>------]  82% - acc: 0.3828 - loss: 2.0716 - 17609.3 sample/sec
Global step:   171 - [=========================>----]  87% - acc: 0.4297 - loss: 2.0189 - 17846.9 sample/sec
Global step:   181 - [==========================>---]  92% - acc: 0.4453 - loss: 2.0133 - 17228.7 sample/sec
Global step:   191 - [============================>-]  97% - acc: 0.4492 - loss: 2.0209 - 17359.3 sample/sec

Epoch 1 - train accuracy: 35.23% - test accuracy: 43.80% (4380/10000) - time: 00:00:03.91
###########################################################################################################

Epoch: 2/10

Global step:   197 - [>-----------------------------]   0% - acc: 0.4492 - loss: 2.0087 - 17177.4 sample/sec
Global step:   207 - [=>----------------------------]   5% - acc: 0.4219 - loss: 2.0287 - 17836.8 sample/sec
Global step:   217 - [==>---------------------------]  10% - acc: 0.4570 - loss: 2.0131 - 17822.6 sample/sec
Global step:   227 - [====>-------------------------]  15% - acc: 0.4141 - loss: 2.0286 - 17534.2 sample/sec
Global step:   237 - [=====>------------------------]  20% - acc: 0.4766 - loss: 1.9853 - 16905.9 sample/sec
Global step:   247 - [=======>----------------------]  26% - acc: 0.4766 - loss: 1.9918 - 18051.8 sample/sec
Global step:   257 - [========>---------------------]  31% - acc: 0.5078 - loss: 1.9627 - 17271.6 sample/sec
Global step:   267 - [==========>-------------------]  36% - acc: 0.4844 - loss: 1.9653 - 16336.6 sample/sec
Global step:   277 - [===========>------------------]  41% - acc: 0.4375 - loss: 2.0279 - 18016.1 sample/sec
Global step:   287 - [=============>----------------]  46% - acc: 0.4766 - loss: 1.9764 - 16909.1 sample/sec
Global step:   297 - [==============>---------------]  51% - acc: 0.4805 - loss: 1.9781 - 17939.0 sample/sec
Global step:   307 - [================>-------------]  56% - acc: 0.5508 - loss: 1.9160 - 17857.9 sample/sec
Global step:   317 - [=================>------------]  61% - acc: 0.5039 - loss: 1.9599 - 17679.7 sample/sec
Global step:   327 - [===================>----------]  66% - acc: 0.4727 - loss: 1.9857 - 16852.0 sample/sec
Global step:   337 - [====================>---------]  71% - acc: 0.5234 - loss: 1.9552 - 16926.6 sample/sec
Global step:   347 - [======================>-------]  77% - acc: 0.5352 - loss: 1.9105 - 17843.9 sample/sec
Global step:   357 - [=======================>------]  82% - acc: 0.4492 - loss: 2.0178 - 16872.1 sample/sec
Global step:   367 - [=========================>----]  87% - acc: 0.5039 - loss: 1.9591 - 17939.9 sample/sec
Global step:   377 - [==========================>---]  92% - acc: 0.5312 - loss: 1.9210 - 17089.4 sample/sec
Global step:   387 - [============================>-]  97% - acc: 0.5273 - loss: 1.9337 - 17390.5 sample/sec

Epoch 2 - train accuracy: 48.40% - test accuracy: 52.59% (5259/10000) - time: 00:00:03.12
This epoch receive better accuracy: 52.59 > 43.80. Saving session...
###########################################################################################################

Epoch: 3/10

Global step:   393 - [>-----------------------------]   0% - acc: 0.5469 - loss: 1.9222 - 18228.1 sample/sec
Global step:   403 - [=>----------------------------]   5% - acc: 0.5234 - loss: 1.9451 - 17496.5 sample/sec
Global step:   413 - [==>---------------------------]  10% - acc: 0.5039 - loss: 1.9451 - 17029.7 sample/sec
Global step:   423 - [====>-------------------------]  15% - acc: 0.5273 - loss: 1.9313 - 18196.2 sample/sec
Global step:   433 - [=====>------------------------]  20% - acc: 0.5195 - loss: 1.9260 - 17242.8 sample/sec
Global step:   443 - [=======>----------------------]  26% - acc: 0.5547 - loss: 1.9193 - 17535.6 sample/sec
Global step:   453 - [========>---------------------]  31% - acc: 0.5195 - loss: 1.9240 - 17280.8 sample/sec
Global step:   463 - [==========>-------------------]  36% - acc: 0.5938 - loss: 1.8806 - 17425.5 sample/sec
Global step:   473 - [===========>------------------]  41% - acc: 0.4883 - loss: 1.9525 - 17394.7 sample/sec
Global step:   483 - [=============>----------------]  46% - acc: 0.5703 - loss: 1.8879 - 17484.0 sample/sec
Global step:   493 - [==============>---------------]  51% - acc: 0.5703 - loss: 1.9086 - 17417.9 sample/sec
Global step:   503 - [================>-------------]  56% - acc: 0.6016 - loss: 1.8633 - 17075.5 sample/sec
Global step:   513 - [=================>------------]  61% - acc: 0.5117 - loss: 1.9516 - 16884.1 sample/sec
Global step:   523 - [===================>----------]  66% - acc: 0.4766 - loss: 1.9808 - 16768.0 sample/sec
Global step:   533 - [====================>---------]  71% - acc: 0.5664 - loss: 1.8908 - 17027.8 sample/sec
Global step:   543 - [======================>-------]  77% - acc: 0.5859 - loss: 1.8538 - 17224.0 sample/sec
Global step:   553 - [=======================>------]  82% - acc: 0.5312 - loss: 1.9204 - 16708.3 sample/sec
Global step:   563 - [=========================>----]  87% - acc: 0.5547 - loss: 1.8988 - 17595.7 sample/sec
Global step:   573 - [==========================>---]  92% - acc: 0.6016 - loss: 1.8527 - 18116.4 sample/sec
Global step:   583 - [============================>-]  97% - acc: 0.6016 - loss: 1.8585 - 17040.0 sample/sec

Epoch 3 - train accuracy: 54.75% - test accuracy: 57.75% (5775/10000) - time: 00:00:03.12
This epoch receive better accuracy: 57.75 > 52.59. Saving session...
###########################################################################################################

Epoch: 4/10

Global step:   589 - [>-----------------------------]   0% - acc: 0.6133 - loss: 1.8475 - 17383.2 sample/sec
Global step:   599 - [=>----------------------------]   5% - acc: 0.5742 - loss: 1.8693 - 18154.1 sample/sec
Global step:   609 - [==>---------------------------]  10% - acc: 0.6250 - loss: 1.8345 - 18074.6 sample/sec
Global step:   619 - [====>-------------------------]  15% - acc: 0.6484 - loss: 1.8264 - 17633.8 sample/sec
Global step:   629 - [=====>------------------------]  20% - acc: 0.5586 - loss: 1.8966 - 17032.2 sample/sec
Global step:   639 - [=======>----------------------]  26% - acc: 0.5625 - loss: 1.8989 - 17358.5 sample/sec
Global step:   649 - [========>---------------------]  31% - acc: 0.6211 - loss: 1.8422 - 17032.2 sample/sec
Global step:   659 - [==========>-------------------]  36% - acc: 0.6523 - loss: 1.8172 - 17360.7 sample/sec
Global step:   669 - [===========>------------------]  41% - acc: 0.6250 - loss: 1.8294 - 16875.8 sample/sec
Global step:   679 - [=============>----------------]  46% - acc: 0.6562 - loss: 1.8086 - 17137.9 sample/sec
Global step:   689 - [==============>---------------]  51% - acc: 0.6445 - loss: 1.8401 - 17328.2 sample/sec
Global step:   699 - [================>-------------]  56% - acc: 0.5625 - loss: 1.8789 - 16876.6 sample/sec
Global step:   709 - [=================>------------]  61% - acc: 0.6641 - loss: 1.8051 - 17031.1 sample/sec
Global step:   719 - [===================>----------]  66% - acc: 0.5820 - loss: 1.8707 - 17312.8 sample/sec
Global step:   729 - [====================>---------]  71% - acc: 0.5703 - loss: 1.8767 - 17399.0 sample/sec
Global step:   739 - [======================>-------]  77% - acc: 0.6484 - loss: 1.8174 - 17296.9 sample/sec
Global step:   749 - [=======================>------]  82% - acc: 0.5859 - loss: 1.8712 - 16861.3 sample/sec
Global step:   759 - [=========================>----]  87% - acc: 0.6836 - loss: 1.7916 - 18019.1 sample/sec
Global step:   769 - [==========================>---]  92% - acc: 0.6445 - loss: 1.8143 - 16999.0 sample/sec
Global step:   779 - [============================>-]  97% - acc: 0.6836 - loss: 1.7736 - 17099.4 sample/sec

Epoch 4 - train accuracy: 62.03% - test accuracy: 62.48% (6248/10000) - time: 00:00:03.12
This epoch receive better accuracy: 62.48 > 57.75. Saving session...
###########################################################################################################

Epoch: 5/10

Global step:   785 - [>-----------------------------]   0% - acc: 0.6719 - loss: 1.7920 - 17263.8 sample/sec
Global step:   795 - [=>----------------------------]   5% - acc: 0.6562 - loss: 1.8073 - 17260.2 sample/sec
Global step:   805 - [==>---------------------------]  10% - acc: 0.6055 - loss: 1.8564 - 17697.8 sample/sec
Global step:   815 - [====>-------------------------]  15% - acc: 0.6133 - loss: 1.8415 - 17628.9 sample/sec
Global step:   825 - [=====>------------------------]  20% - acc: 0.6055 - loss: 1.8604 - 17059.5 sample/sec
Global step:   835 - [=======>----------------------]  26% - acc: 0.6523 - loss: 1.8040 - 17910.3 sample/sec
Global step:   845 - [========>---------------------]  31% - acc: 0.6641 - loss: 1.7923 - 16734.3 sample/sec
Global step:   855 - [==========>-------------------]  36% - acc: 0.6211 - loss: 1.8227 - 16885.1 sample/sec
Global step:   865 - [===========>------------------]  41% - acc: 0.7188 - loss: 1.7524 - 16769.6 sample/sec
Global step:   875 - [=============>----------------]  46% - acc: 0.7188 - loss: 1.7497 - 17070.1 sample/sec
Global step:   885 - [==============>---------------]  51% - acc: 0.6797 - loss: 1.7741 - 16890.2 sample/sec
Global step:   895 - [================>-------------]  56% - acc: 0.6289 - loss: 1.8177 - 17790.4 sample/sec
Global step:   905 - [=================>------------]  61% - acc: 0.6836 - loss: 1.7837 - 17493.1 sample/sec
Global step:   915 - [===================>----------]  66% - acc: 0.6289 - loss: 1.8324 - 18305.7 sample/sec
Global step:   925 - [====================>---------]  71% - acc: 0.6445 - loss: 1.8021 - 17310.9 sample/sec
Global step:   935 - [======================>-------]  77% - acc: 0.6680 - loss: 1.7762 - 17871.0 sample/sec
Global step:   945 - [=======================>------]  82% - acc: 0.6602 - loss: 1.7999 - 17236.4 sample/sec
Global step:   955 - [=========================>----]  87% - acc: 0.7188 - loss: 1.7477 - 17480.0 sample/sec
Global step:   965 - [==========================>---]  92% - acc: 0.6523 - loss: 1.7961 - 17143.7 sample/sec
Global step:   975 - [============================>-]  97% - acc: 0.6953 - loss: 1.7561 - 17670.4 sample/sec

Epoch 5 - train accuracy: 65.94% - test accuracy: 66.20% (6620/10000) - time: 00:00:03.12
This epoch receive better accuracy: 66.20 > 62.48. Saving session...
###########################################################################################################

Epoch: 6/10

Global step:   981 - [>-----------------------------]   0% - acc: 0.7266 - loss: 1.7364 - 17525.6 sample/sec
Global step:   991 - [=>----------------------------]   5% - acc: 0.7344 - loss: 1.7278 - 17163.7 sample/sec
Global step:  1001 - [==>---------------------------]  10% - acc: 0.6836 - loss: 1.7677 - 17027.0 sample/sec
Global step:  1011 - [====>-------------------------]  15% - acc: 0.6406 - loss: 1.8100 - 17102.2 sample/sec
Global step:  1021 - [=====>------------------------]  20% - acc: 0.6602 - loss: 1.7913 - 17371.9 sample/sec
Global step:  1031 - [=======>----------------------]  26% - acc: 0.6758 - loss: 1.7885 - 17065.5 sample/sec
Global step:  1041 - [========>---------------------]  31% - acc: 0.7109 - loss: 1.7538 - 17096.2 sample/sec
Global step:  1051 - [==========>-------------------]  36% - acc: 0.6602 - loss: 1.8037 - 17323.2 sample/sec
Global step:  1061 - [===========>------------------]  41% - acc: 0.6953 - loss: 1.7591 - 16861.8 sample/sec
Global step:  1071 - [=============>----------------]  46% - acc: 0.7227 - loss: 1.7356 - 17827.1 sample/sec
Global step:  1081 - [==============>---------------]  51% - acc: 0.7109 - loss: 1.7586 - 17985.0 sample/sec
Global step:  1091 - [================>-------------]  56% - acc: 0.6992 - loss: 1.7594 - 17007.6 sample/sec
Global step:  1101 - [=================>------------]  61% - acc: 0.7109 - loss: 1.7493 - 17995.9 sample/sec
Global step:  1111 - [===================>----------]  66% - acc: 0.6484 - loss: 1.8097 - 17760.4 sample/sec
Global step:  1121 - [====================>---------]  71% - acc: 0.7031 - loss: 1.7641 - 17710.9 sample/sec
Global step:  1131 - [======================>-------]  77% - acc: 0.7227 - loss: 1.7386 - 18030.6 sample/sec
Global step:  1141 - [=======================>------]  82% - acc: 0.7148 - loss: 1.7629 - 17171.1 sample/sec
Global step:  1151 - [=========================>----]  87% - acc: 0.6836 - loss: 1.7664 - 17369.7 sample/sec
Global step:  1161 - [==========================>---]  92% - acc: 0.7031 - loss: 1.7542 - 17412.8 sample/sec
Global step:  1171 - [============================>-]  97% - acc: 0.7227 - loss: 1.7355 - 17080.1 sample/sec

Epoch 6 - train accuracy: 69.65% - test accuracy: 67.79% (6779/10000) - time: 00:00:03.12
This epoch receive better accuracy: 67.79 > 66.20. Saving session...
###########################################################################################################

Epoch: 7/10

Global step:  1177 - [>-----------------------------]   0% - acc: 0.7695 - loss: 1.6955 - 17449.0 sample/sec
Global step:  1187 - [=>----------------------------]   5% - acc: 0.7539 - loss: 1.7035 - 17058.7 sample/sec
Global step:  1197 - [==>---------------------------]  10% - acc: 0.7148 - loss: 1.7558 - 17274.7 sample/sec
Global step:  1207 - [====>-------------------------]  15% - acc: 0.7188 - loss: 1.7419 - 17334.9 sample/sec
Global step:  1217 - [=====>------------------------]  20% - acc: 0.6875 - loss: 1.7690 - 16994.2 sample/sec
Global step:  1227 - [=======>----------------------]  26% - acc: 0.6953 - loss: 1.7715 - 17080.1 sample/sec
Global step:  1237 - [========>---------------------]  31% - acc: 0.7305 - loss: 1.7270 - 17931.9 sample/sec
Global step:  1247 - [==========>-------------------]  36% - acc: 0.7266 - loss: 1.7477 - 17237.0 sample/sec
Global step:  1257 - [===========>------------------]  41% - acc: 0.7539 - loss: 1.7049 - 16864.4 sample/sec
Global step:  1267 - [=============>----------------]  46% - acc: 0.7383 - loss: 1.7215 - 17459.8 sample/sec
Global step:  1277 - [==============>---------------]  51% - acc: 0.7617 - loss: 1.7079 - 17783.1 sample/sec
Global step:  1287 - [================>-------------]  56% - acc: 0.7656 - loss: 1.6946 - 17979.6 sample/sec
Global step:  1297 - [=================>------------]  61% - acc: 0.7461 - loss: 1.7129 - 17690.2 sample/sec
Global step:  1307 - [===================>----------]  66% - acc: 0.6758 - loss: 1.7875 - 17228.9 sample/sec
Global step:  1317 - [====================>---------]  71% - acc: 0.7266 - loss: 1.7278 - 17054.1 sample/sec
Global step:  1327 - [======================>-------]  77% - acc: 0.7500 - loss: 1.7184 - 17026.2 sample/sec
Global step:  1337 - [=======================>------]  82% - acc: 0.7109 - loss: 1.7461 - 16977.2 sample/sec
Global step:  1347 - [=========================>----]  87% - acc: 0.7305 - loss: 1.7370 - 17319.0 sample/sec
Global step:  1357 - [==========================>---]  92% - acc: 0.7344 - loss: 1.7362 - 16962.2 sample/sec
Global step:  1367 - [============================>-]  97% - acc: 0.7305 - loss: 1.7256 - 17250.0 sample/sec

Epoch 7 - train accuracy: 73.11% - test accuracy: 68.53% (6853/10000) - time: 00:00:03.12
This epoch receive better accuracy: 68.53 > 67.79. Saving session...
###########################################################################################################

Epoch: 8/10

Global step:  1373 - [>-----------------------------]   0% - acc: 0.7539 - loss: 1.7093 - 17470.6 sample/sec
Global step:  1383 - [=>----------------------------]   5% - acc: 0.7773 - loss: 1.6869 - 16656.5 sample/sec
Global step:  1393 - [==>---------------------------]  10% - acc: 0.7578 - loss: 1.7089 - 16686.7 sample/sec
Global step:  1403 - [====>-------------------------]  15% - acc: 0.7344 - loss: 1.7314 - 17990.1 sample/sec
Global step:  1413 - [=====>------------------------]  20% - acc: 0.6992 - loss: 1.7444 - 17169.7 sample/sec
Global step:  1423 - [=======>----------------------]  26% - acc: 0.6914 - loss: 1.7599 - 16695.0 sample/sec
Global step:  1433 - [========>---------------------]  31% - acc: 0.7188 - loss: 1.7445 - 17149.4 sample/sec
Global step:  1443 - [==========>-------------------]  36% - acc: 0.7305 - loss: 1.7275 - 16917.8 sample/sec
Global step:  1453 - [===========>------------------]  41% - acc: 0.7773 - loss: 1.6832 - 17104.1 sample/sec
Global step:  1463 - [=============>----------------]  46% - acc: 0.7500 - loss: 1.7130 - 17340.8 sample/sec
Global step:  1473 - [==============>---------------]  51% - acc: 0.7773 - loss: 1.6900 - 16777.7 sample/sec
Global step:  1483 - [================>-------------]  56% - acc: 0.7734 - loss: 1.6898 - 17347.0 sample/sec
Global step:  1493 - [=================>------------]  61% - acc: 0.7617 - loss: 1.7044 - 17778.4 sample/sec
Global step:  1503 - [===================>----------]  66% - acc: 0.6953 - loss: 1.7701 - 17066.0 sample/sec
Global step:  1513 - [====================>---------]  71% - acc: 0.7461 - loss: 1.7225 - 16867.4 sample/sec
Global step:  1523 - [======================>-------]  77% - acc: 0.7461 - loss: 1.7084 - 18154.1 sample/sec
Global step:  1533 - [=======================>------]  82% - acc: 0.7695 - loss: 1.6998 - 16702.6 sample/sec
Global step:  1543 - [=========================>----]  87% - acc: 0.7383 - loss: 1.7168 - 16767.3 sample/sec
Global step:  1553 - [==========================>---]  92% - acc: 0.7422 - loss: 1.7121 - 16745.8 sample/sec
Global step:  1563 - [============================>-]  97% - acc: 0.7812 - loss: 1.6859 - 18157.2 sample/sec

Epoch 8 - train accuracy: 74.61% - test accuracy: 71.10% (7110/10000) - time: 00:00:03.14
This epoch receive better accuracy: 71.10 > 68.53. Saving session...
###########################################################################################################

Epoch: 9/10

Global step:  1569 - [>-----------------------------]   0% - acc: 0.8203 - loss: 1.6693 - 17285.5 sample/sec
Global step:  1579 - [=>----------------------------]   5% - acc: 0.7578 - loss: 1.6947 - 17360.4 sample/sec
Global step:  1589 - [==>---------------------------]  10% - acc: 0.7852 - loss: 1.6840 - 17761.3 sample/sec
Global step:  1599 - [====>-------------------------]  15% - acc: 0.7930 - loss: 1.6854 - 17047.6 sample/sec
Global step:  1609 - [=====>------------------------]  20% - acc: 0.7539 - loss: 1.7072 - 17385.2 sample/sec
Global step:  1619 - [=======>----------------------]  26% - acc: 0.7383 - loss: 1.7265 - 17654.7 sample/sec
Global step:  1629 - [========>---------------------]  31% - acc: 0.7500 - loss: 1.7173 - 17799.9 sample/sec
Global step:  1639 - [==========>-------------------]  36% - acc: 0.7266 - loss: 1.7427 - 18013.7 sample/sec
Global step:  1649 - [===========>------------------]  41% - acc: 0.7930 - loss: 1.6800 - 16630.1 sample/sec
Global step:  1659 - [=============>----------------]  46% - acc: 0.7656 - loss: 1.6955 - 17085.8 sample/sec
Global step:  1669 - [==============>---------------]  51% - acc: 0.7695 - loss: 1.6911 - 17063.8 sample/sec
Global step:  1679 - [================>-------------]  56% - acc: 0.7695 - loss: 1.6938 - 17537.1 sample/sec
Global step:  1689 - [=================>------------]  61% - acc: 0.7539 - loss: 1.7067 - 17287.5 sample/sec
Global step:  1699 - [===================>----------]  66% - acc: 0.7148 - loss: 1.7515 - 17185.4 sample/sec
Global step:  1709 - [====================>---------]  71% - acc: 0.7734 - loss: 1.6871 - 17262.7 sample/sec
Global step:  1719 - [======================>-------]  77% - acc: 0.7852 - loss: 1.6695 - 17003.3 sample/sec
Global step:  1729 - [=======================>------]  82% - acc: 0.7539 - loss: 1.7142 - 17032.2 sample/sec
Global step:  1739 - [=========================>----]  87% - acc: 0.7539 - loss: 1.6997 - 16652.3 sample/sec
Global step:  1749 - [==========================>---]  92% - acc: 0.7500 - loss: 1.7116 - 17637.6 sample/sec
Global step:  1759 - [============================>-]  97% - acc: 0.7812 - loss: 1.6774 - 17548.8 sample/sec

Epoch 9 - train accuracy: 76.45% - test accuracy: 71.50% (7150/10000) - time: 00:00:03.14
This epoch receive better accuracy: 71.50 > 71.10. Saving session...
###########################################################################################################

Epoch: 10/10

Global step:  1765 - [>-----------------------------]   0% - acc: 0.7812 - loss: 1.6761 - 17611.0 sample/sec
Global step:  1775 - [=>----------------------------]   5% - acc: 0.7891 - loss: 1.6682 - 17352.6 sample/sec
Global step:  1785 - [==>---------------------------]  10% - acc: 0.7734 - loss: 1.6908 - 16725.7 sample/sec
Global step:  1795 - [====>-------------------------]  15% - acc: 0.7617 - loss: 1.6965 - 17682.3 sample/sec
Global step:  1805 - [=====>------------------------]  20% - acc: 0.7812 - loss: 1.6857 - 17006.3 sample/sec
Global step:  1815 - [=======>----------------------]  26% - acc: 0.7305 - loss: 1.7351 - 17645.1 sample/sec
Global step:  1825 - [========>---------------------]  31% - acc: 0.7812 - loss: 1.6702 - 17286.6 sample/sec
Global step:  1835 - [==========>-------------------]  36% - acc: 0.7695 - loss: 1.6912 - 17319.0 sample/sec
Global step:  1845 - [===========>------------------]  41% - acc: 0.8164 - loss: 1.6496 - 17946.2 sample/sec
Global step:  1855 - [=============>----------------]  46% - acc: 0.7734 - loss: 1.6865 - 17377.0 sample/sec
Global step:  1865 - [==============>---------------]  51% - acc: 0.7695 - loss: 1.6866 - 17286.4 sample/sec
Global step:  1875 - [================>-------------]  56% - acc: 0.7852 - loss: 1.6736 - 17126.4 sample/sec
Global step:  1885 - [=================>------------]  61% - acc: 0.7812 - loss: 1.6790 - 17047.6 sample/sec
Global step:  1895 - [===================>----------]  66% - acc: 0.7266 - loss: 1.7306 - 16749.2 sample/sec
Global step:  1905 - [====================>---------]  71% - acc: 0.8008 - loss: 1.6635 - 17318.7 sample/sec
Global step:  1915 - [======================>-------]  77% - acc: 0.8281 - loss: 1.6411 - 17390.5 sample/sec
Global step:  1925 - [=======================>------]  82% - acc: 0.8320 - loss: 1.6349 - 16974.0 sample/sec
Global step:  1935 - [=========================>----]  87% - acc: 0.7969 - loss: 1.6645 - 18026.7 sample/sec
Global step:  1945 - [==========================>---]  92% - acc: 0.7656 - loss: 1.6956 - 17895.4 sample/sec
Global step:  1955 - [============================>-]  97% - acc: 0.8164 - loss: 1.6479 - 17398.1 sample/sec

Epoch 10 - train accuracy: 78.30% - test accuracy: 71.93% (7193/10000) - time: 00:00:03.12
This epoch receive better accuracy: 71.93 > 71.50. Saving session...
###########################################################################################################
Best accuracy pre session: 71.93, time: 00:00:32.72
```


`predict_out.txt`
```
2019-03-14 16:18:39.291383: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2019-03-14 16:18:39.387408: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:897] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2019-03-14 16:18:39.388393: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1405] Found device 0 with properties: 
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:22:00.0
totalMemory: 10.91GiB freeMemory: 10.00GiB
2019-03-14 16:18:39.388428: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-03-14 16:18:39.583260: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-03-14 16:18:39.583296: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-03-14 16:18:39.583302: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-03-14 16:18:39.583533: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 9668 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:22:00.0, compute capability: 6.1)

Trying to restore last checkpoint ...
Restored checkpoint from: ./tensorboard/cifar-10-v1.0.0/-1960

Accuracy on Test-Set: 71.93% (7193 / 10000)
```