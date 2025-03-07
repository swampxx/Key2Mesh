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
The `demo.py` demonstrates how to estimate and visualize SMPL body models from 2D keypoints. A sample dataset is
provided, which includes raw images and their corresponding OpenPose outputs, serving as a guide for the expected data
structure. You can run the demo using the provided example or use your own data by updating the configuration.

#### Running the Demo with Example Data:

Simply execute the following command: `python demo.py`

#### Using Your Own Data:

1. Extract 2D Keypoints:
    - Install and run [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) on your images to extract 2D
      keypoints, and save the results in JSON format.
2. Organize Your Data:
    - Structure your data similarly to the example:
        - Place your raw images in a folder (e.g., your_data/imgs).
        - Save the corresponding OpenPose outputs (JSON files) in a folder (e.g., your_data/openpose_outs).
3. Update Configuration:
    - Modify the input parameters in the configs/demo.yaml file to point to your dataset directory and set image
      size.
4. Finally, run the demo: `python demo.py`

## Evaluation

To evaluate the model adapted to either the 3DPW or InstaVariety datasets, update the `run` parameter in
the `configs/eval_3dpw.yaml` file:

* For 3DPW, set it to `target-3dpw`.
* For InstaVariety, set it to `target-insta`.

Then, run the following command: `python eval_3dpw.py`

## Acknowledgements

This code is built on top of the following work:
[J. Song, X. Chen, and O. Hilliges, "Human Body Model Fitting by Learned Gradient Descent," in ECCV, 2020.](https://arxiv.org/abs/2008.08474)

## Citation

If you find our work useful in your research, please consider citing:
```
@InProceedings{Uguz_2024_CVPR,
  author    = {Uguz, Bedirhan and Suat, Ozhan and Karagoz, Batuhan and Akbas, Emre},
  title     = {MoCap-to-Visual Domain Adaptation for Efficient Human Mesh Estimation from 2D Keypoints},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month     = {June},
  year      = {2024},
  pages     = {1622-1632}
}
```