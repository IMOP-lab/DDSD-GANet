# Adversarial Optical‑Prior Transfer with Multiresolution Anisotropic Decoding for Coherence-Structured SAR Segmentation


🌟 DDSD-GANet: Adversarial Optical-Prior Transfer with Multiresolution Anisotropic Decoding for Coherence-Structured SAR Segmentation
This project implements the Dual-Driven Spectral-Directional Cross-Modal Generative Adversarial Network (DDSD-GANet) proposed in the paper "Adversarial Optical-Prior Transfer with Multiresolution Anisotropic Decoding for Coherence-Structured SAR Segmentation".


DDSD-GANet is an advanced framework for Synthetic Aperture Radar (SAR) image semantic segmentation. It is designed to address the challenges inherent in SAR imagery, specifically multiplicative speckle noise, absence of optical-domain semantic priors (geometric semantic deficiency), and intrinsic scale inconsistency. By incorporating adversarial cross-modal learning and an innovative decoder structure , DDSD-GANet achieves significant performance improvements in the structured segmentation of SAR targets, such as buildings, vessels, and sea-land boundaries.


## Detailed image of Dual-Driven Spectral-Directional Cross-Modal Generative Adversarial Network

<img width="1741" height="806" alt="image" src="https://github.com/user-attachments/assets/65f37380-6ec7-4425-8168-8b810ddead96" />

The Dual-Driven Spectral-Directional Cross-Modal Generative Adversarial Network is proposed to address the challenges of coherence-induced noise, scale inconsistency, and semantic deficiency in SAR imagery.Subfigures: (a) overall framework; (b) segmentation encoder; (c) segmentation decoder with MACU and CAG; (d) ResNet encoder; (e–f) SAR-to-optical GAN encoder/decoder; (g) Up Block for progressive upsampling; (h) OCF for cross-branch fusion and gating.

## Detailed image of Multiresolution Anisotropic Coherence Unit

<img width="1251" height="809" alt="image" src="https://github.com/user-attachments/assets/6267064a-4606-4d55-a81b-fe92039ba67e" />

The proposed Multiresolution Anisotropic Coherence Unit couples AAPB with MRPB to capture directional structures while suppressing speckle noise.

## Detailed image of Coherent-aware Gated Skip

<img width="1137" height="553" alt="image" src="https://github.com/user-attachments/assets/6d6e5919-1e6e-4684-a7ee-198b06027167" />

The proposed Coherent-aware Gated Skip employs a dual-driven gating mechanism to suppress spurious patterns in shallow features and, under global semantic guidance, selectively pass true boundary information.

## Datasets

SARBuD：The dataset is available through \url{https://github.com/CAESAR-Radi/SARBuD}
HRSID：The dataset is available through \url{https://github.com/chaozhong2010/HRSID}
Sea-land Segmentation：The dataset is available through \url{https://radars.ac.cn/web/data/getData?dataType=sea-landsegmentation}

