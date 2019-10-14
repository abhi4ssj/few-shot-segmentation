# 'Squeeze \& Excite' Guided Few-Shot Segmentation of Volumetric Images

The paper is published at the journal Medical Image Analysis. Link to arxiv: https://arxiv.org/abs/1902.01314.

This project contains the source code for training and evaluation for all the experiments of the aforementioned work. Root directory contains the neural network file 'few_shot_segmentaion.py' and the relevant 'solver.py' for the proposed method.

## Getting Started

### Pre-requisites

You need to have following in order for this library to work as expected
1. python >= 3.5
2. pip >= 19.0
3. pytorch >= 1.0.0
4. numpy >= 1.14.0
5. nn-common-modules >=1.0 (https://github.com/ai-med/nn-common-modules, A collection of commonly used code modules in deep learning. Follow this link to know more about installation and usage)
6. Squeeze and Excitation >=1.0 (https://github.com/ai-med/squeeze_and_excitation, Follow this link to know more about installation and usage)
7. nn-additional-losses >=1.0 (https://github.com/ai-med/nn-additional-losses, A collection of losses not part of pytorch standard library particularly useful for segmentation task. Follow this link to know more about installation and usage)

### Training your model

```
python run.py --mode=train --device=device_id
```

### Evaluating your model

```
python run.py --mode=eval
```

## Code Authors

* **Shayan Ahmad Siddiqui**  - [shayansiddiqui](https://github.com/shayansiddiqui)
* **Abhijit Guha Roy**  - [abhi4ssj](https://github.com/abhi4ssj)


## Help us improve
Let us know if you face any issues. You are always welcome to report new issues and bugs and also suggest further improvements. And if you like our work hit that start button on top. Enjoy :)

## 

If you use this code, please cite:

```
@article{roy2019squeeze,
  title={'Squeeze \& Excite'Guided Few-Shot Segmentation of Volumetric Images},
  author={Roy, Abhijit Guha and Siddiqui, Shayan and P{\"o}lsterl, Sebastian and Navab, Nassir and Wachinger, Christian},
  journal={Medical Image Analysis},
  year={2019}
}
```
