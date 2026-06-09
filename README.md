# Adversarial Optical‑Prior Transfer with Multiresolution Anisotropic Decoding for Coherence-Structured SAR Segmentation


🌟 DDSD-GANet: Adversarial Optical-Prior Transfer with Multiresolution Anisotropic Decoding for Coherence-Structured SAR Segmentation
This project implements the Dual-Driven Spectral-Directional Cross-Modal Generative Adversarial Network (DDSD-GANet) proposed in the paper "Adversarial Optical-Prior Transfer with Multiresolution Anisotropic Decoding for Coherence-Structured SAR Segmentation".


DDSD-GANet is an advanced framework for Synthetic Aperture Radar (SAR) image semantic segmentation. It is designed to address the challenges inherent in SAR imagery, specifically multiplicative speckle noise, absence of optical-domain semantic priors (geometric semantic deficiency), and intrinsic scale inconsistency. By incorporating adversarial cross-modal learning and an innovative decoder structure , DDSD-GANet achieves significant performance improvements in the structured segmentation of SAR targets, such as buildings, vessels, and sea-land boundaries.


## Detailed image of Dual-Driven Spectral-Directional Cross-Modal Generative Adversarial Network

<img width="761" height="387" alt="image-DDSD-GANet" src="https://github.com/user-attachments/assets/cdb0dbad-62cc-4b99-a29c-aed3a67661ce" />


The Dual-Driven Spectral-Directional Cross-Modal Generative Adversarial Network is proposed to address the challenges of coherence-induced noise, scale inconsistency, and semantic deficiency in SAR imagery.Subfigures: (a) overall framework; (b) segmentation encoder; (c) segmentation decoder with MACU and CAG; (d) ResNet encoder; (e–f) SAR-to-optical GAN encoder/decoder; (g) Up Block for progressive upsampling; (h) OCF for cross-branch fusion and gating.

## Detailed image of Multiresolution Anisotropic Coherence Unit
<img width="541" height="351" alt="image-MACU2" src="https://github.com/user-attachments/assets/c4a4a228-f19f-4da6-9659-5463bb72a533" />



The proposed Multiresolution Anisotropic Coherence Unit couples AAPB with MRPB to capture directional structures while suppressing speckle noise.

## Detailed image of Coherent-aware Gated Skip

<img width="783" height="383" alt="image-CAG" src="https://github.com/user-attachments/assets/b8442cb8-de1e-4410-9807-b58dbca184ba" />


The proposed Coherent-aware Gated Skip employs a dual-driven gating mechanism to suppress spurious patterns in shallow features and, under global semantic guidance, selectively pass true boundary information.

## Datasets

SARBuD：The dataset is available through \url{https://github.com/CAESAR-Radi/SARBuD}

HRSID：The dataset is available through \url{https://github.com/chaozhong2010/HRSID}

Sea-land Segmentation：The dataset is available through \url{https://radars.ac.cn/web/data/getData?dataType=sea-landsegmentation}

## Implementation Details

For fair comparison, all methods share the same data preprocessing and training settings. Input images are normalized and resized to a fixed resolution, then augmented with random cropping, intensity jitter, and simple affine transformations to increase data diversity. We train all networks for 100 epochs with a batch size of 2 using the AdamW optimizer with a learning rate of 1e-4. Experiments are implemented in PyTorch on a workstation equipped with NVIDIA RTX 3090 GPUs.

## Segmentation Performance Comparisons

In evaluating network performance, a suite of distinct metrics was utilized, including the 95th Percentile Hausdorff Distance, Dice Coefficient, Intersection over Union, Cohen’s Kappa, and Matthews Correlation Coefficient.

<img width="808" height="405" alt="image" src="https://github.com/user-attachments/assets/5a098f48-113a-42f6-8576-15d563cc9fa2" />

<img width="805" height="404" alt="image" src="https://github.com/user-attachments/assets/221b662f-a2af-4fae-a860-086eb9d2acfd" />

<img width="809" height="422" alt="image" src="https://github.com/user-attachments/assets/604ffe21-c539-4a62-b83d-5f6865bc98f0" />

<img width="1208" height="1049" alt="image" src="https://github.com/user-attachments/assets/c2038a48-3cac-4b29-b176-b344dda57f0d" />







