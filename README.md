## Exploring Feature Representation and Training strategies in Temporal Action Localization

This repo holds the codes and models for the SSN framework presented on ICIP 2019.

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
	* [Train the model](#train-the-model)
	* [Test the model](#test-the-model)
	* [Use reference models for evaluation](#use-reference-models-for-evaluation)

* [Temporal Action Detection Performance on THUMOS14](#temporal-action-detection-performance-on-thumos14)

* [Other Info](#other_-nfo)
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
git clone (#TODO: Add the repo address)
```

Note: Before running the code, please remember to change the path to the dataset ```self.prefix``` in ```config.py```.

#### Proposals

The test action proposals are provided in ```props/test_proposals_from_TURN.txt```. If you want to generate your own proposals, please go to [TURN][turn] repository. Also, in this paper we report the performance according to different Average Number(AN) proposals, and these proposals are also provided in ```./props/```.

#### Train the model

In the original paper, we use the model trained in the following way.

```bash
python main.py --pool_level=k --fusion_type=fusion_type
```

```k``` is the granularity we used to divide each proposal. Mostly, we use```k=5```. ```fusion_type``` represents the way we regard the two stream features, which includes 'early', 'rgb', 'flow'.(#TODO: Add 'late fusion code')

Also, we trained an improved version of this.(#TODO: remove the variances calculation in the code)

```bash
python main.py --pool_level=k --fusion_type=fusion_type --dropout=True --opm_type='adam_wd'
```

#### Use reference models for evaluation

We provide the pretrained reference models in tensorflow ```ckpt``` format, which is in ```./model/``` in this repo.

There are three steps to evaluate temporal action detection with our pretrained models. First, we will extract the detection scores for all proposals by running:

```bash
python main.py --pool_level=5 --fusion_type=fusion_type  --mode=test --cas_step=3 --test_model_path=MODEL_PATH
```

Then, the result pickle file ```PKL_FILE``` could be found in ```./eval/test_results/```, and it will be used to find the action class it belongs to and the corresponding offsets in folder ```./eval/```.

```bash
python gen_prop_outputs.py PKL_FILE 3/1(#TODO: Change it according to different versions)
```

Finally, NMS is used to supppress the redundant proposals.

```bash
python postproc.py PKL_FILE 0.5
```

### Temporal Action Detection Performance on THUMOS14

The mAP@0.5 performance of the baseline model we provide is ```44.85%``` under the [evaluation method 2014][eval2014]. While we find another [evaluation method 2015][eval2015], to make a fair comparison, we also report the important results on it as follows.

#### Table 1: mAP@tIoU(%) with different fixed-size feature representation methods

```
|      mAP@IoU (%)    |  0.3  |  0.4  |  0.5  |  0.6  |  0.7  |
|-----------------------------|-------|-------|---------------|
| STPP(L=2)           |       |       |       |       |       |
| STPP(L=3)           |       |       |       |       |       |
| BSP(2/4/2)          |       |       |       |       |       |
| BSP(4/8/4)          |       |       |       |       |       |
| BSP(8/16/8)         |       |       |       |       |       |
| Ours(k=1)           | 46.69 | 40.48 | 31.23 | 19.95 | 9.78  |
| Ours(k=2)           | 50.20 | 43.67 | 34.31 | 23.77 | 10.83 |
| Ours(k=5)           | 51.66 | 46.56 | 36.83 | 25.39 | 12.69 |
| Ours(k=10)          | 52.49 | 46.58 | 37.37 | 24.54 | 12.43 |
|-------------------------------------------------------------|
```
#### Table 2: mAP@tIoU (%) with different fusion methods.
```
|      mAP@IoU (%)    |  0.3  |  0.4  |  0.5  |  0.6  |  0.7  |
|-----------------------------|-------|-------|---------------|
| RGB                 |       |       |       |       |       |
| Flow                |       |       |       |       |       |
| Early Fusion        |       |       |       |       |       |
| Late Fusion         |       |       |       |       |       |
|-------------------------------------------------------------|
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

Also, I would like to thank Christos Tzelepis for his suggestions both in this project and the paper.

[anet-2016]: https://github.com/yjxiong/anet2016-cuhk
[cbr]: https://github.com/jiyanggao/CBR
[turn]: https://github.com/jiyanggao/TURN-TAP
[thumos14]: https://www.crcv.ucf.edu/THUMOS14/home.html
[eval2014]: [https://www.crcv.ucf.edu/THUMOS14/THUMOS14_Evaluation.pdf]
[eval2015]: [https://storage.googleapis.com/www.thumos.info/thumos15_zips/THUMOS14_evalkit_20150930.zip]
