## Experiment Results

## Task1 Traditional Classifiers

```
训练集样本数: 486
测试集样本数: 57
特征维度: 384

============================================================
所有方法性能对比总结:
============================================================
SVM (RBF核):        98.25%
SVM (线性核):       98.25%
SVM (多项式核):     96.49%
朴素贝叶斯:         98.25%
K近邻 (K=5):        96.49%
逻辑回归:           100.00%
============================================================
```

### Best Method Detailed Results (SVM - RBF Kernel)

```
训练集准确率: 100.00%
测试集准确率: 98.25%

各类别表现:
  类别 0 (Black_footed_Albatross):    6/6   100.00%
  类别 1 (Laysan_Albatross):          6/6   100.00%
  类别 2 (Sooty_Albatross):           6/6   100.00%
  类别 3 (Groove_billed_Ani):         6/6   100.00%
  类别 4 (Crested_Auklet):            4/5    80.00%  ⚠️
  类别 5 (Least_Auklet):              5/5   100.00%
  类别 6 (Parakeet_Auklet):           6/6   100.00%
  类别 7 (Rhinoceros_Auklet):         5/5   100.00%
  类别 8 (Brewer_Blackbird):          6/6   100.00%
  类别 9 (Red_winged_Blackbird):      6/6   100.00%

总计: 56/57 正确
```

## Task2 Deep Learning Models

Loading train data from /home/stu6/PriorTaskForML2025/data/train...
Loaded 10597 images from 200 classes.
Loading val data from /home/stu6/PriorTaskForML2025/data/val...
Loaded 1191 images from 200 classes.
Total parameters: 11,279,112
Trainable parameters: 11,279,112

### CNN

Final evaluation with best model:
  Validation Loss: 3.4150
  Validation Accuracy: 19.98%

### Resnet18

Final evaluation with best model:
  Validation Loss: 1.3540
  Validation Accuracy: 70.03%

### Resnet34

Final evaluation with best model:
  Validation Loss: xxx(正在跑实验)
  Validation Accuracy: xx%(正在跑实验，预计80%左右)