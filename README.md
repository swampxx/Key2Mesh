# Key2Mesh: MoCap-to-Visual Domain Adaptation for Efficient Human Mesh Estimation from 2D Keypoints (CVPRW24)

Welcome! This is the official implementation of the CVPRW24 paper [Key2Mesh: MoCap-to-Visual Domain Adaptation for Efficient Human Mesh
Estimation from 2D Keypoints](https://arxiv.org/abs/2404.07094) by Bedirhan Uguz, Ozhan Suat, Batuhan Karagoz,
and [Emre Akbas](https://user.ceng.metu.edu.tr/~emre/).

For more details and to see some cool results, check out the [project page](https://key2mesh.github.io/).

## Installation

### Data

We follow VIBE's data preparation steps. Please refer to the [VIBE repository](https://github.com/mkocabas/VIBE) for
instructions on downloading the required data.

After downloading the data, please copy the necessary files to the `data` directory.
You will need to have the following structure:

```
 data
 ├── 3DPW_test.pt
 ├── J_regressor_h36m.npy
 └── SMPL_NEUTRAL.pkl
```

### Dependencies

* Install torch 1.12.1 (with CUDA 11.3 support)
* Install torchvision 0.13.1 (with CUDA 11.3 support)
* Then, run the following command to install the remaining dependencies: `pip install -r requirements.txt`

## Demo

_Coming soon..._

## Evaluation

To evaluate the model adapted to either the 3DPW or InstaVariety datasets, update the `run` parameter in
the `configs/eval_3dpw.yaml` file:

* For 3DPW, set it to `target-3dpw`.
* For InstaVariety, set it to `target-insta`.

Then, run the following command: `python eval_3dpw.py`

## Acknowledgements

This code is built on top of the following work:
[J. Song, X. Chen, and O. Hilliges, "Human Body Model Fitting by Learned Gradient Descent," in ECCV, 2020.](https://arxiv.org/abs/2008.08474).


## Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{uguz2024mocap,
  title={MoCap-to-Visual Domain Adaptation for Efficient Human Mesh Estimation from 2D Keypoints},
  author={Uguz, Bedirhan and Suat, Ozhan and Karagoz, Batuhan and Akbas, Emre},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1622--1632},
  year={2024}
}
```