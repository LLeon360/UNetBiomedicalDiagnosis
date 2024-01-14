# Sample Results
<details>

<summary><b>Expand to view images</b></summary>

#### U-Net Ex:
<img width="800" alt="UNet(5E16F) Results" src="https://user-images.githubusercontent.com/103869590/202833418-8b14db95-2513-47ed-a7a0-4cd2bd049e7f.PNG">

#### GradCAM Heatmap Ex:
<img width="430" alt="DenseNet201_Last_Img150" src="https://user-images.githubusercontent.com/103869590/179291908-def12ab5-6b3e-498d-9436-d2a57828effc.png">

</details>

# Research Paper 

## Presented at AAAI (Association for the Advancement of AI) Fall Series Symposium 2022:

## Presentation Video:
https://youtu.be/5adO69BNP0M

### AAAI KGML Symposium Agenda Here: (Scroll to Session 5 Paper 3, Presentation at 2:36)
https://sites.google.com/vt.edu/kgml-aaai-22 

## Paper Link: 
Jaiden Xuan Schraut, Leon Liu, Jonathan Gong & Yiqiao Yin (2022), "A multi-output network with U-net enhanced class activation map and robust classification performance for medical imaging analysis"
https://link.springer.com/article/10.1007/s44163-022-00045-1

Paper Presented at AAAI:
Leon Liu, Yiqiao Yin (2022), "Towards Explainable AI on Chest X-ray Diagnosis using Image Segmentation and CAM Visualization"
https://drive.google.com/open?id=1aLdN9p4008pGKvzhkdzUcY4FD-sjQFWV

## Abstract
As of today, there are many very capable state of the art AI image classifications techniques, namely in convolutional neural nets like VGG, ResNet, DenseNet, Xception, etc. However in the field of digital pathology, high accuracy classification is not enough, these black box models produce only a final confidence for classification but no insight into its decision making. It is here that visualization techniques and computer vision can be applied to augment these models for increased interpretability to build more transparent and trustworthy systems. 

My research proposes a pipeline specifically for lung disease classification and explainability by applying techniques of U-Net image segmentation and Grad-CAM heatmaps to communicate a neural net's decision-making and provide visualizations for a doctor to analyze and make final judgements in practical clinical use.

# UNetBiomedicalDiagnosis
Contains a module for functions that build and evaluate a U-Net Model which is a modified auto-encoder image-to-image architecture that includes skip connections from encoder to decoder layers with matching input dimensions to retain features from the encoded image. The model is designed for image segmentation and produces a mask based off x-ray data. The model is used with lung x-rays to segment out lungs, brain scans to segment out brain tumors, and pictures of a room to segment out a human figure. With the COVID-QU-Ex chest x-ray dataset for lung segmentation, it is able to achieve 98% accuracy with a wide range of 3,4,5 encoding block of Conv2D and MaxPooling and BatchNormalization + 1 latent layer without pooling and the mirrored number of decoding layers that do Conv2DTranspose for deconvolution and UpScaling to reverse MaxPooling making 7,9,11 layers with doubling filters/kernels in each encoding block from 16,32,64,80 that double with each block like in a traditional CNN and then half with each decoding block.

## Colab Notebook:
### Unet Training: 
https://colab.research.google.com/drive/1q8k_JOL2-EkIIpuX2x3EcqEAtwWfuBoB#scrollTo=sV328lJpNOMD 

### Unet Result + CNN + GradCAM Heatmap: 
https://colab.research.google.com/drive/1hGpYr7w37-9qHvYaFKWLQKPytmldCT0S#scrollTo=uWeiAf8ybw-j
