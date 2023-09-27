<div align="center">
  
# Simulated Bias in Artifical Medical Images

</div>

Implementation for framework presented in our MICCAI 2023 paper: "A flexible framework for simulating and evaluating biases in deep learning-based medical image analysis"

<p align="center">
<img src="figures/paper1779_fig1.png?raw=true" width="750">
</p>


## Abstract 
Despite the remarkable advances in deep learning for medical image analysis, it has become evident that biases in datasets used for training such models pose considerable challenges for a clinical deployment, including fairness and domain generalization issues. Although the development of bias mitigation techniques has become ubiquitous, the nature of inherent and unknown biases in real-world medical image data prevents a comprehensive understanding of algorithmic bias when developing deep learning models and bias mitigation methods. To address this challenge, we propose a modular and customizable framework for bias simulation in synthetic but realistic medical imaging data. Our framework provides complete control and flexibility for simulating a range of bias scenarios that can lead to undesired model performance and shortcut learning. In this work, we demonstrate how this framework can be used to simulate morphological biases in neuroimaging data for disease classification with a convolutional neural network as a first feasibility analysis. Using this case example, we show how the proportion of bias in the disease class and proximity between disease and bias regions can affect model performance and explainability results. The proposed framework provides the opportunity to objectively and comprehensively study how biases in medical image data affect deep learning pipelines, which will facilitate a better understanding of how to responsibly develop models and bias mitigation methods for clinical use.

## Usage

### Generative models for sampling effects
We use PCA models fit to velocity fields (Log-Euclidian framework) derived from 50 T1-weighted brain MRIs ([IXI](https://brain-development.org/ixi-dataset/) dataset) nonlinearly registered to the [SRI24 atlas](https://www.nitrc.org/projects/sri24) as generative models for sampling global subject effects as well as local disease and bias effects. The code used for generating these models is in: 
```bash
├── pca
│   ├── pca_isv_velo_ixi.py
│   ├── pca_velo_ixi.py
│   ├── save_roi_masks.py
│   └── subspacemodels.py
```
* `pca_isv_velo_ixi.py` fits the PCA model for the full subject morphology.
* `pca_velo_ixi.py` fits the PCA models for the localized region morphology.

Our PCA models for global subject morphology as well as each region defined by the LPBA40 labels are available [here](link).

### Dataset generation
Subject, disease, and bias effects are represented by morphological variation introduced to a template image (we use the SRI24 atlas). Generating the synthetic datasets requires the following files: 
```bash
├── generate_data
│   ├── define_stratified_distributions.py
│   ├── generate_data.py
│   └── utils.py
```
* `defined_stratified_distributions.py` generates parameters for the train, validation, and test splits in which disease classes, biase groups, and splits are stratified by subject and disease effect magnitude values to be sampled from the respective PCA models. This controls for "bias" in the underlying distributions of subject + disease effects in each disease class and bias group that deep learning models can pick up on. The number of samples and bounds of the sampling distribution are defined in this code.
* `generate_data.py` generates the data for each split in `.nii.gz` format. Disease and bias regions are defined in this code.

For a fully controlled evaluation of bias in deep learning pipelines, it is recommended to generate subject- and disease effect-paired datasets of images with and without the addition of bias effects (simply comment out the addition of bias effects in `generate_data.py`). Evaluating these with- and without-bias "counterfactuals" with the exact same model training scheme (initialization seeds, GPU state, splits, batch order) and measuring bias group performance disparities relative to the without-bias "baseline" ensures the most rigorous evaluation, as models may exhibit small (<5%) levels of baseline disparity (e.g., true and false positive rates) even with effect stratification. 

A sample dataset (used in our upcoming journal paper) is available for download [here](link).

### Model pipeline 
We train models in Keras/Tensorflow but the generated datasets can be used with any deep learning library. Our model pipeline code is in:
```bash
├── model
│   ├── avg_saliency_maps.py
│   ├── datagenerator_3d.py
│   ├── evaluate_3d.py
│   ├── model_3d.py
│   └── xai_3d.py
```

### Citation
If you find this framework, code, or paper useful to your research, please cite the [paper](link):

```
@inproceedings{stanley2023framework,
    address = {Cham},
    series = {Lecture {Notes} in {Computer} {Science}},
    author = {Stanley, Emma A.M. and Wilms, Matthias and Forkert, Nils D.},
    title = {A flexible framework for simulating and evaluating biases in deep learning-based medical image analysis},
    year = {2023},
    booktitle = {Medical {Image} {Computing} and {Computer} {Assisted} {Intervention} – {MICCAI} 2023},
    publisher = {Springer Nature Switzerland},
    }
```
```
E.A.M. Stanley, M. Wilms, N.D. Forkert (2023) A flexible framework for simulating and evaluating biases in deep learning-based medical image analysis. In: Proceedings of MICCAI 2023. 
```

### Environment 
Our dataset generation code used:
* Python 3.10.6
* simpleitk v. 2.1.1.1

Our code for Keras model training used: 
* Python 3.10.6
* simpleitk v. 2.1.1.1
* tensorflow-gpu 2.10.0
* cudnn 8.4.1.50
* cudatoolkit 11.7.0

And NVIDIA GeForce RTX 3090 GPU.

Full environment in `requirements.txt`.

