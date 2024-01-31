<div align="center">
  
# Simulated Bias in Artificial Medical Images

</div>

<p align="center">
<img src="figures/paper1779_fig1.png?raw=true" width="600">
</p>


Implementation for framework presented in our MICCAI 2023 paper: "[A flexible framework for simulating and evaluating biases in deep learning-based medical image analysis](https://link.springer.com/chapter/10.1007/978-3-031-43895-0_46)" and utilized in our preprint "[Towards objective and systematic evaluation of bias in medical imaging AI](https://arxiv.org/abs/2311.02115)".

Our code here is based on our initial feasibility study of spatially localized morphological bias effects in structural neuroimaging datasets. However, the crux of the SimBA framework is the **systematic augmentation of a template image with disease (target) effects, bias effects, and subject effects**. This simple procedure can be expanded to other organ templates and imaging modalities. 

If you find the SimBA 🦁 framework, code, or paper useful to your research, please cite us!
```
@inproceedings{stanley_framework_2023,
	address = {Cham},
	series = {Lecture {Notes} in {Computer} {Science}},
	author = {Stanley, Emma A. M. and Wilms, Matthias and Forkert, Nils D.},
	title = {A {Flexible} {Framework} for {Simulating} and {Evaluating} {Biases} in {Deep} {Learning}-{Based} {Medical} {Image} {Analysis}},
	booktitle = {Medical {Image} {Computing} and {Computer} {Assisted} {Intervention} – {MICCAI} 2023},
	publisher = {Springer Nature Switzerland},
	editor = {Greenspan, Hayit and Madabhushi, Anant and Mousavi, Parvin and Salcudean, Septimiu and Duncan, James and Syeda-Mahmood, Tanveer and Taylor, Russell},
	year = {2023},
	pages = {489--499}}
```
```
Stanley, E.A.M., Wilms, M., Forkert, N.D. (2023). A Flexible Framework for Simulating and Evaluating Biases in Deep Learning-Based Medical Image Analysis. In: Greenspan, H., et al. Medical Image Computing and Computer Assisted Intervention – MICCAI 2023. MICCAI 2023. Lecture Notes in Computer Science, vol 14221. Springer, Cham. https://doi.org/10.1007/978-3-031-43895-0_46
```

### Abstract 
Despite the remarkable advances in deep learning for medical image analysis, it has become evident that biases in datasets used for training such models pose considerable challenges for a clinical deployment, including fairness and domain generalization issues. Although the development of bias mitigation techniques has become ubiquitous, the nature of inherent and unknown biases in real-world medical image data prevents a comprehensive understanding of algorithmic bias when developing deep learning models and bias mitigation methods. To address this challenge, we propose a modular and customizable framework for bias simulation in synthetic but realistic medical imaging data. Our framework provides complete control and flexibility for simulating a range of bias scenarios that can lead to undesired model performance and shortcut learning. In this work, we demonstrate how this framework can be used to simulate morphological biases in neuroimaging data for disease classification with a convolutional neural network as a first feasibility analysis. Using this case example, we show how the proportion of bias in the disease class and proximity between disease and bias regions can affect model performance and explainability results. The proposed framework provides the opportunity to objectively and comprehensively study how biases in medical image data affect deep learning pipelines, which will facilitate a better understanding of how to responsibly develop models and bias mitigation methods for clinical use.


## Generating Data

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

Our PCA models for global subject morphology as well as each region defined by the LPBA40 labels are available [here](https://drive.google.com/file/d/1LQeOA1mrNZm9KbWZrj5-8ZGJpMjiAzFD/view?usp=sharing).

Example shell script for generating PCA models in `example_pca.sh`.

### Dataset generation
Subject, disease, and bias effects are represented by morphological variation introduced to a template image (we use the SRI24 atlas). Generating the synthetic datasets requires the following files: 
```bash
├── generate_data
│   ├── define_stratified_distributions.py
│   ├── generate_data.py
│   └── utils.py
```
* `defined_stratified_distributions.py` generates parameters for the train, validation, and test splits in which disease classes, bias groups, and splits are stratified by subject and disease effect magnitude values to be sampled from the respective PCA models. This controls for "bias" in the underlying distributions of subject + disease effects in each disease class and bias group that deep learning models can pick up on. The number of samples and bounds of the sampling distributions for disease and subject effects are defined in this code.
* `generate_data.py` generates the data for each pre-defined split in `.nii.gz` format. Disease and bias regions are defined in this code.

⭐ For a fully controlled evaluation of bias in deep learning pipelines, it is recommended to generate subject- and disease effect-paired datasets of images with and without the addition of bias effects (simply comment out the addition of bias effects in `generate_data.py`). **Evaluating these with- and without-bias "counterfactuals" with the exact same model training scheme (initialization seeds, GPU state, splits, batch order) and measuring bias group performance disparities relative to the without-bias "baseline" ensures the most rigorous evaluation**, as models may exhibit small (<5%) levels of baseline disparity (i.e., true and false positive rates) even with effect stratification. Sample "counterfactual" datasets (used in our [full paper](https://arxiv.org/abs/2311.02115)) are available for download [here](https://mega.nz/folder/wKNVTSqZ#4OgMoOnEFyk32CunjV-XIg).

Example shell script for generating data in `example_generate_data.sh`.


## Evaluating Deep Learning Pipelines

### Model pipeline 
We train models in Keras/Tensorflow but the generated datasets can be used with any deep learning library. Our model pipeline code is in:
```bash
├── model
│   ├── avg_saliency_maps.py
│   ├── datagenerator_3d.py
│   ├── evaluate_3d.py
│   ├── model_3d.py
│   ├── xai_3d.py
│   └── saliency_rois.py
```
Example shell script for running our model pipeline in `example_model_pipeline.sh`.

### Bias mitigation 
We evaluate the following bias mitigation strategies in the [full paper](https://arxiv.org/abs/2311.02115): [reweighing](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=595538decb1228a9fbeb0a1df3581c64dea95dd7), [unlearning](https://www.sciencedirect.com/science/article/pii/S1053811920311745), and [bias group models](https://link.springer.com/chapter/10.1007/978-3-030-87199-4_39). 
Our main code for each of these strategies is in: 
```bash
├── bias_mitigation
│   ├── bias_mit_utils.py
│   ├── datagenerator_3d_reweigh.py
│   ├── model_3d_reweigh.py
│   ├── datagenerator_3d_unlearning.py
│   ├── unlearning.py
│   ├── split_encoder_prediction.py
│   ├── evaluate_3d_unlearning.py
│   └── model_3d_protectedgroupmodel.py
```
* **Reweighing** follows the same standard model pipeline as [above](#model-pipeline), with `model_3d.py` replaced with `model_3d_reweigh.py` (which loads `datagenerator_3d_reweigh.py`).
* **Unlearning** requires models pre-trained on the main task (disease classification) and the bias prediction task. The pre-trained feature encoder, disease prediction head, and bias prediction head are then split as done in `split_encoder_prediction.py`. Then, `unlearning.py` (loads `datagenerator_3d_unlearning.py`) performs the adversarial training for unlearning the bias attribute while keeping disease clasification performance high. The encoder and prediction heads are stitched back together for evaluation as done in `evaluate_3d_unlearning.py`. 
* **Bias group models** begins with pre-training the model for a few epochs on the full dataset, and then running `model_3d_protectedgroupmodel.py` on each of the bias groups separately.

## Environment 
Our dataset generation code used:
* Python 3.10.6
* simpleitk 2.1.1.1
* antspyx 0.3.4
* pandas 1.5.0
* numpy 1.23.3
* scikit-learn 1.1.2
* scipy 1.9.1

Our code for the Keras model pipeline used: 
* Python 3.10.6
* pandas 1.5.0
* numpy 1.23.3
* scikit-learn 1.1.2
* simpleitk 2.1.1.1
* tensorflow-gpu 2.10.0
* cudnn 8.4.1.50
* cudatoolkit 11.7.0

GPU: NVIDIA GeForce RTX 3090

Full environment in `requirements.txt`.


## Resources
* A summary of this framework was presented at the [Fairness of AI in Medical Imaging (FAIMI)](https://faimi-workshop.github.io/2023-online/) 2023 online symposium! 
* Questions? Open an issue or send an [email](mailto:emma.stanley@ucalgary.ca?subject=SimBA).

