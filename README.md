## Exploring Feature Representation and Training strategies in Temporal Action Localization

This repo holds the codes and models for the temporal action localization framework presented on ICIP 2019.

**Exploring Feature Representation and Training strategies in Temporal Action Localization**
Tingting Xie, Xiaoshan Yang, Tianzhu Zhang, Changsheng Xu, Ioannis Patras, *ICIP 2019*

[[Arxiv Preprint]](https://arxiv.org/abs/1905.10608)

If you find this helps your research, please cite:
```
@article{xie2019exploring,
  title={Exploring Feature Representation and Training strategies in Temporal Action Localization},
  author={Xie, Tingting and Yang, Xiaoshan and Zhang, Tianzhu and Xu, Changsheng and Patras, Ioannis},
  journal={arXiv preprint arXiv:1905.10608},
  year={2019}
}
```
## Contents
---

* [Usage Guide](#usage-guide)
	* [Download data](#download-data)
	* [Get the code](#get-the-code)
	* [Proposals](#proposals)
	* [Train the network](#train-the-network)
	* [Use reference models for evaluation](#use-reference-models-for-evaluation)
	* [Postprocessing](#postprocessing)

* [Temporal Action Detection Performance on THUMOS14](#temporal-action-detection-performance-on-thumos14)

* [Other Info](#other_info)
	* [Related project](#related-project)
	* [Contact](#contact)

---

### Usage Guide

#### Download data

In this paper, [two-stream feature][anet-2016] was using to extract features in unit-level for [thumos14 dataset][thmos14]. The RGB feature could be downloaded here: [val set](https://drive.google.com/file/d/180YUoPvyaF2Z_T9KMKINLdDQCZEg60Jb/view?usp=sharing), [test set](https://drive.google.com/file/d/1x9Q78AZiAGqx4XB2zO3SEKp1htsATlnU/view?usp=sharing); the denseflow features can be downloaded here: [val set](https://drive.google.com/file/d/1-6dmY_Uy-H19HxvfK_wUFQCYHmlPzwFx/view?usp=sharing), [test set](https://drive.google.com/file/d/1Qm9lIJQFm5s6hDSB_2k1tj8q2tnabflJ/view?usp=sharing). Note that, val set is used for training, as the train set for THUMOS-14 does not contain untrimmed videos.

#### Get the code

The training and testing in the work is implemented in Tensorflow for ease of use. We need the following software mainly to run it.

- Python3
- Tensorflow1.14

GPUs are required for running this code. Usually 1 GPU and 3~4GB of the memory would ensure a smooth training experience.

Then clone this repo with git.

```bash
git clone git@github.com:June01/icip19-tad.git
```

Note: Before running the code, please remember to change the path of the features(named by```self.prefix```) in ```config.py```.

#### Proposals

The test action proposals are provided in ```props/test_proposals_from_TURN.txt```. If you want to generate your own proposals, please go to [TURN][turn] repository. Also, in this paper we report the performance according to different Average Number(AN) proposals, which are also provided in ```./props/```.

#### Train the network

In the original paper, we train the network with the following command.

```bash
python main.py --pool_level=k --fusion_type=fusion_type
```

```k``` is the granularity we used to divide each proposal into units. Mostly, we use```k=5``` by default. ```fusion_type``` represents the way we deal with two stream features, such as RGB, Flow, early fusion. As to the late fusion, please turn to [postprocessing](#postprocessing).

Note: All the results in the paper was reported on [THUMOS14 evaluation 2014][eval2014]. However, there is another one [THUMOS14 evaluation 2015][eval2015], which is not obviously stated on the website even though it should have been done years ago. (We figured out the differences between these two evaluation codes, please file an issue if any explanation about it needed.) Base on the new evaluation metric, we make some changes during training, you can train your own network with the following command. Also, the results on it could be found in the next section.

```bash
python main.py --pool_level=k --fusion_type=fusion_type --dropout=True --opm_type='adam_wd' --l1_loss=True
```

#### Use reference models for evaluation

We provide the pretrained reference models in tensorflow ```ckpt``` format, which is put in ```./model/``` in this repo.

First, you need to get the detection scores for all proposals by running:

```bash
python main.py --pool_level=k --fusion_type=fusion_type  --mode=test --cas_step=3 --test_model_path=MODEL_PATH
```

#### Postprocessing

Then, the result pickle file ```PKL_FILE``` could be found in ```./eval/test_results/```, and it will be used to compute the action class it belongs to and the corresponding offsets in folder ```./eval/```.

```bash
python gen_prop_outputs.py PKL_FILE_1 PKL_FILE_2 3
```
For rgb, flow and early fusion results, ```PKL_FILE_1``` and ```PKL_FILE_2``` should be set the same; while for late fusion, ```PKL_FILE_1``` should be set to be the rgb pkl file and ```PKL_FILE_2``` should be set to be the flow pkl file. After this step, you may get the ```FINAL_PKL_FILE```.

Finally, NMS is used to supppress the redundant proposals. The final predicted actions list will be save in ```./eval/after_postprocessing/```.

```bash
python postproc.py FINAL_PKL_FILE 0.5
```

### Temporal Action Detection Performance on THUMOS14

The mAP@0.5 performance of the baseline model we provide is ```44.85%``` under the [evaluation method 2014][eval2014]. Based on [evaluation method 2015][eval2015], we also report the important results on it as follows, which is also comparable with the state-of-the-art ```36.9%```.

#### Table 1: mAP@tIoU(%) with different k(cascade step = 3)

```
|      mAP@IoU (%)    |  0.3  |  0.4  |  0.5  |  0.6  |  0.7  |
---------------------------------------------------------------
| Ours(k=1)           | 46.69 | 40.48 | 31.23 | 19.95 | 9.78  |
| Ours(k=2)           | 50.20 | 43.67 | 34.31 | 23.77 | 10.83 |
| Ours(k=5)           | 51.66 | 46.56 | 36.83 | 25.39 | 12.69 |
| Ours(k=10)          | 52.49 | 46.58 | 37.37 | 24.54 | 12.43 |
---------------------------------------------------------------
```

#### Table 2: mAP@tIoU (%) with different fusion methods(k=5).
```
|      mAP@IoU (%)    |  0.3  |  0.4  |  0.5  |  0.6  |  0.7  |
---------------------------------------------------------------
| RGB                 | 39.07 | 33.67 | 23.55 | 13.15 | 5.70  |
| Flow                | 47.12 | 42.05 | 33.80 | 22.89 | 12.13 |
| Early Fusion        | 51.66 | 46.56 | 36.83 | 25.39 | 12.69 |
| Late Fusion         | 49.77 | 44.45 | 34.98 | 21.33 | 10.36 |
---------------------------------------------------------------
```
### Other Info

#### Related project
- [Anet-2016][anet-2016]: The two-stream based feature extractor used in this paper.
- [CBR][cbr]: The foundmental network we based on.
- [TURN-TAP][turn]: The first stage proposals generated from.


#### Contact

For any question, please file an issue or contact

```
Tingting Xie: t.xie@qmul.ac.uk
```

Also, I would like to thank Yu-le Li and Christos Tzelepis for his valuable suggestions and discussions both in this project and the paper.

[anet-2016]: https://github.com/yjxiong/anet2016-cuhk
[cbr]: https://github.com/jiyanggao/CBR
[turn]: https://github.com/jiyanggao/TURN-TAP
[thumos14]: https://www.crcv.ucf.edu/THUMOS14/home.html
[eval2014]: https://www.crcv.ucf.edu/THUMOS14/THUMOS14_Evaluation.pdf
[eval2015]: https://storage.googleapis.com/www.thumos.info/thumos15_zips/THUMOS14_evalkit_20150930.zip
