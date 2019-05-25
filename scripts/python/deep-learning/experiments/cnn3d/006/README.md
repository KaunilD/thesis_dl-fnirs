##### Experiment: 006


| hyp-params    | value                    |
| :------------ | ------------------------ |
| epochs        | 30                       |
| loss          | MultiLabelSoftMarginLoss |
| optimizer     | Adam                     |
| learning rate | 0.001                    |
| accuracy      | hamming_score            |

##### Model: 3D CNN

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv3d-1          [-1, 1, 5, 22, 1]             101
            Linear-2                    [-1, 9]             207
           Sigmoid-3                    [-1, 9]               0
================================================================
Total params: 308
Trainable params: 308
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.04
Forward/backward pass size (MB): 0.00
Params size (MB): 0.00
Estimated Total Size (MB): 0.04
----------------------------------------------------------------
```

##### Data

| param              | value                                                        |
| :----------------- | ------------------------------------------------------------ |
| timesteps          | 100                                                          |
| normalization      | none                                                         |
| source experiments | only session 1 from [mindfulness/benchmark_tasks/fNIRS_Data](https://github.com/lmhirshf/mindfulness/tree/master/benchmark_tasks/data/fNIRS_Data) |
| label type         | multilabel; default3                                         |
| label config       | [ wm_o, wm_l, wm_h, v_o, v_l, v_h, a_o, a_l, a_h, ewm_o, ewm_l, ewm_h ] |

##### Training

```
Epoch   Train Loss      Validation Loss Validation Acc
0       419.50088       27.31999        0.271
1       394.75207       25.84787        0.188
2       390.95656       25.30643        0.165
3       389.42745       25.18773        0.160
4       388.62196       25.17447        0.160
5       388.49124       25.17342        0.160
6       388.00463       25.17826        0.160
7       387.83292       25.17816        0.160
8       387.73361       25.17767        0.160
9       387.52299       25.17377        0.160
10      387.10318       25.18034        0.160
11      387.11284       25.18055        0.160
12      387.06545       25.18340        0.160
13      387.06613       25.18774        0.160
14      387.05359       25.18674        0.160
15      387.05199       25.18461        0.160
16      387.04801       25.18151        0.160
17      387.04382       25.18663        0.160
18      387.03476       25.18445        0.160
19      387.03233       25.17957        0.160
20      387.02101       25.14210        0.160
21      387.35184       25.17627        0.160
22      387.10178       25.17808        0.160
23      387.56046       25.17286        0.160
24      387.01774       25.16347        0.160
25      387.03842       25.17976        0.160
26      387.06982       25.18571        0.160
27      386.99649       25.16083        0.160
28      387.51319       25.18455        0.160
29      387.01582       25.18417        0.160
30      386.96848       25.18263        0.160
```

##### Observations

1. hamming score giving low values for sparse labels. try euclidean distance instead.