##### Experiment: 001


| hyp-params    | value                    |
| :------------ | ------------------------ |
| epochs        | 30                       |
| loss          | MultiLabelSoftMarginLoss |
| optimizer     | Adam                     |
| learning rate | 0.001                    |

##### Model: 3D CNN

``` 
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv3d-1          [-1, 5, 5, 22, 1]             105
            Conv3d-2          [-1, 2, 2, 11, 1]              12
            Conv3d-3           [-1, 1, 1, 5, 1]               3
            Linear-4                   [-1, 12]              72
           Sigmoid-5                   [-1, 12]               0
================================================================
Total params: 192
Trainable params: 192
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 0.00
Params size (MB): 0.00
Estimated Total Size (MB): 0.01
----------------------------------------------------------------
```

### Data:

| param              | value                                                        |
| :----------------- | ------------------------------------------------------------ |
| timesteps          | 20                                                           |
| normalization      | none                                                         |
| source experiments | only session 1 from [mindfulness/benchmark_tasks/fNIRS_Data](https://github.com/lmhirshf/mindfulness/tree/master/benchmark_tasks/data/fNIRS_Data) |
| label type         | multilabel; default4                                         |
| label config       | [ wm_o, wm_l, wm_h, v_o, v_l, v_h, a_o, a_l, a_h, ewm_o, ewm_l, ewm_h ] |

