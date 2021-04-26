# Classification and Gradient-based Localization of Chest Radiographs

## Contents
- [Introduction](#introduction)
- [Overview](#overview)
- [Steps](#steps)
- [Results](#results)
- [Conclusions](#conclusions)
- [References](#references)

## Introduction
> A team of radiologists from New Orleans studied the usefulness of Chest Radiographs for diagnosing COVID-19 compared to the reverse-transcription polymerase chain reaction (RT-PCR) and found out they could aid rapid diagnosis, especially in areas with limited testing facilities [[1]](https://pubs.rsna.org/doi/10.1148/ryct.2020200280 "A Characteristic Chest Radiographic Pattern in the Setting of the COVID-19 Pandemic").<br>
> Another study found out that the radiographs of different viral cases of pneumonia are comparative, and they overlap with other infectious and inflammatory lung diseases, making it hard for radiologists to recognize COVID‐19 from other viral pneumonia cases [[2]](https://pubs.rsna.org/doi/10.1148/rg.2018170048 "Radiographic and CT Features of Viral Pneumonia").<br>
> This project aims to make the former study a reality while dealing with the intricacies in the latter, with the help of Deep Learning.<br>

## Overview
> The project uses the COVID-19 Radiography Database [[3]](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database) as it's dataset.
> It has a total of `21165` Chest X-Rays (CXRs) belonging to 4 different classes (`COVID-19`, `Lung Opacity`, `Normal` and `Viral Pneumonia`).<br>
> Three top scoring CNN architectures, __VGG-16__, __ResNet-18__ and __DenseNet-121__, trained on the ImageNet Dataset [[4]](http://image-net.org/), were chosen for __fine-tuning__ on the dataset.<br>
> The results obtained from the different architectures were then evaluted and compared.<br>
> Finally, with the help of __Gradient weighted Class Activation Maps__ (Grad-CAM) [[5]](https://arxiv.org/abs/1610.02391 "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization") the affected areas in CXRs were localized.<br>

* ___Note:___ The dataset and the trained models can be found in [here](https://drive.google.com/drive/folders/14L8wd-d2a3lvgqQtwV-y53Gsnn6Ud2-w?usp=sharing).<br>

## Steps
> 1. [Dataset Exploration](./1_data_exploration.ipynb "1_data_exploration.ipynb")
> 2. [Split the dataset](./split_dataset.py "split_dataset.py")
>    |Type|COVID-19|Lung Opacity|Normal|Viral Pneumonia|Total|
>    |:-|-:|-:|-:|-:|-:|
>    |Train|3496|5892|10072|1225|20685|
>    |Val|60|60|60|60|240|
>    |Test|60|60|60|60|240|
> 3. [Fine-tune VGG-16, ResNet-18 and DenseNet-121](./2_finetune_models.ipynb "2_finetune_models.ipynb")
>    1. [Define Transformations](./utils.py#L15-L41)
>    2. [Handle imbalanced dataset with Weighted Random Sampling (Over-sampling)](#)
>    3. [Prepare the Pre-trained models](./networks.py "networks.py")
>    4. [Fine-tune step with Early-stopping](./utils.py#L91-L159)
>       - |Hyper-parameters||
>         |:-|-:|
>         |Learning rate|`0.00003`|
>         |Batch Size|`32`|
>         |Number of Epochs|`25`|
>       - |Loss Function|Optimizer|
>         |:-:|:-:|
>         |`Categorical Cross Entropy`|`Adam`|
>    5. [Plot running losses & accuracies](./plot_utils.py#L8-L42)
>       |Model|Summary Plot|
>       |:-:|:-:|
>       |VGG-16|![vgg_plot](./outputs/summary_plots/vgg.png)|
>       |ResNet-18|![res_plot](./outputs/summary_plots/resnet.png)|
>       |DenseNet-121|![dense_plot](./outputs/summary_plots/densenet.png)|
> 4. [Results Evaluation](./3_evaluate_results.ipynb "3_evaluate_results.ipynb")
>    1. [Plot confusion matrices](./plot_utils.py#L45-L69)
>    2. [Compute test-set Accuracy, Precision, Recall & F1-score](./utils.py#L72-L88)
>    3. [Localize using Grad-CAM](./grad_cam.py)
<br>

## Results

<table>
<tr>
<th></th>
<th>VGG-16</th>
<th>ResNet-18</th>
<th>DenseNet-121</th>
</tr>
<tr>
<td>

|__Pathology__|
|:-|
|COVID-19|
|Lung Opacity|
|Normal|
|Viral Pneumonia|

</td>
<td>

|Accuracy|Precision|Recall|F1-Score|
|-:|-:|-:|-:|
|0.9956|0.9833|1.0000|0.9916|
|0.9582|0.8833|0.9464|0.9138|
|0.9622|0.9667|0.8923|0.9280|
|0.9913|0.9833|0.9833|0.9833|
            
</td>
<td>

|Accuracy|Precision|Recall|F1-Score|
|-:|-:|-:|-:|
|0.9871|0.9667|0.9830|0.9748|
|0.9664|0.8667|1.0000|0.9286|
|0.9664|1.0000|0.8823|0.9375|
|0.9957|1.0000|0.9836|0.9917|
            
</td>
<td>

|Accuracy|Precision|Recall|F1-Score|
|-:|-:|-:|-:|
|0.9957|0.9833|1.0000|0.9916|
|0.9623|0.9167|0.9322|0.9244|
|0.9623|0.9500|0.9047|0.9268|
|0.9957|0.9833|1.0000|0.9916|
            
</td>
</tr>
<tr>
<td>

|TL;DR|
|:-|
|Train set|
|Test set|

</td>
<td>

|Total Correct Predictions|Total Accuracy|
|-:|-:|
|20362|98.44%|
|229|__95.42%__|

</td>
<td>

|Total Correct Predictions|Total Accuracy|
|-:|-:|
|20639|99.78%|
|230|__95.83%__|

</td>
<td>

|Total Correct Predictions|Total Accuracy|
|-:|-:|
|20540|99.30%|
|230|__95.83%__|

</td>
</tr>
<tr>
<td>Confusion Matrices</td>
<td>

![vgg_confmat](./assets/vgg_confmat.png)

</td>
<td>

![res_confmat](./assets/res_confmat.png)

</td>
<td>

![dense_confmat](./assets/dense_confmat.png)

</td>
</tr>
</table>
<br>

- __Localization with Gradient-based Class Activation Maps__
> |![original](./assets/original.jpg)|![vgg_cam](./assets/vgg_cam.jpg)|![res_cam](./assets/res_cam.jpg)|![dense_cam](./assets/dense_cam.jpg)|
> |:-:|:-:|:-:|:-:|
> |<sup>_COVID-19 infected CXR_</sup>|<sup>_VGG-16_</sup>|<sup>_ResNet-18_</sup>|<sup>_DenseNet-121_</sup>|

<br>

## Conclusions
> - DenseNet-121 having only `7.98 Million` parameters did relatively better than VGG-16 and ResNet-18, with `138 Million` and `11.17 Million` parameters respectively.<br>
> - Increase in model's parameter count doesn’t necessarily acheive better results, but increase in residual connections might.<br>
> - Oversampling helped in dealing with imbalanced data to a great extent.<br>
> - Fine-tuning helped substantially by dealing with the comparatively small dataset and speeding up the training process.<br>
> - GradCAM aided in localizing the areas in CXRs that decides a model's predictions.<br>
> - The models did a good job distinguishing various infectious and inflammatory lung diseases, which is rather hard manually, as mentioned earlier.<br>

## References
> [1] David L. Smith, John-Paul Grenier, Catherine Batte, and Bradley Spieler. __A Characteristic Chest Radiographic Pattern in the Setting of the COVID-19 Pandemic.__ Radiology: Cardiothoracic Imaging 2020 2:5.<br>
> [2] Hyun Jung Koo, Soyeoun Lim, Jooae Choe, Sang-Ho Choi, Heungsup Sung, and Kyung-Hyun Do. __Radiographic and CT Features of Viral Pneumonia.__ RadioGraphics 2018 38:3, 719-739.<br>
> [3] Tawsifur Rahman, Muhammad Chowdhury, Amith Khandakar. __COVID-19 Radiography Database.__ Kaggle.<br>
> [4] Deng, J. et al., 2009. __Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition.__ pp. 248–255.<br>
> [5] Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra. __Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.__ arXiv:1610.02391v4.