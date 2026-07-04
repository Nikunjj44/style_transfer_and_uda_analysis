# Input-Space Unsupervised Domain Adaptation on Office-Home (Art → Real World)

This project aims to benchmark five input-space Unsupervised Domain Adaptation (UDA) methods that address domain shift by aligning source and target images before classification. The components covered in this repository include image style transfer with Spatial and Spectral CycleGAN, and UDA classification with Source-only, Spatial CycleGAN, Spectral CycleGAN, CyCADA, and FDA — all evaluated on the Office-Home Art → Real World domain pair.

Deep learning models achieve remarkable performance when training and test data share the same distribution, but in practice collecting labelled data for every target deployment scenario is expensive and often infeasible. Unsupervised Domain Adaptation (UDA) bridges this gap by leveraging labelled source data and unlabelled target data to learn domain-invariant representations. Input-space UDA approaches address domain shift by translating source images to appear visually similar to target images before training a classifier — if the source images look like they came from the target domain, a model trained on them should generalise better to real target data. The key challenge is performing this translation in a way that preserves the semantic content of the image while changing only its low-level style characteristics such as texture, colour distribution, and background.

## Project Setup

This project operates within the following technical framework:

1. **Task 1 Backbone (Image Style Transfer):** A **ResNet-based CycleGAN generator with 9 residual blocks** operating on 256 x 256 images is used. A **PatchGAN discriminator with 4 convolution layers** provides patch-level real/fake predictions, and a buffer of size 50 stores previously generated images for training stability.
2. **Task 2 Backbone (UDA Classification):** A **ResNet-50 classifier initialized from ImageNet pretrained weights** is used. The final 1000-class FC layer is replaced with a 65-class layer matching the Office-Home dataset. For CyCADA, the ResNet-50 backbone is wrapped as a feature extractor with a separate classification head and discriminator network.
3. **Training Setup:** Spatial CycleGAN is trained for 100 epochs, while Spectral CycleGAN is trained for 200 epochs with an additional Fast Fourier Transform (FFT) decomposition step before applying the generators. For CyCADA, feature-level alignment is added via a domain discriminator attached to the ResNet-50 embedding, with a gradient reversal layer (GRL) and adversarial loss weight of 0.2.
4. **Evaluation Protocol:** All methods are evaluated on classification accuracy, Macro-F1, and Weighted-F1 to ensure fair performance assessment across all 65 classes.

## Tech Stack Used

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![CycleGAN](https://img.shields.io/badge/CycleGAN-Image%20Translation-blue?style=for-the-badge)
![ResNet](https://img.shields.io/badge/ResNet--50-ImageNet%20Pretrained-orange?style=for-the-badge)
![Domain Adaptation](https://img.shields.io/badge/Domain%20Adaptation-UDA-green?style=for-the-badge)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

## Methodology

1. **Dataset**  
   The **Office-Home** dataset consists of 65 object categories captured in either art forms like paintings (Art dataset with 2,427 images) or in real natural settings (Real World dataset with 4,357 images). The Art domain is used as the labelled source and the Real World domain is used as the unlabelled target. This domain pair captures the ability of a method to bridge the semantic gap between artistic renderings and real-world objects. Since the Art domain already has a relatively small visual gap to the Real World domain, the transformations applied by the five methods tend to be subtle rather than dramatic — making this domain pair a useful test case for assessing whether style transfer methods can provide meaningful benefit when the underlying domain gap is modest.

2. **Task 1: Image Style Transfer**  
   The goal of this task is to reduce the visual difference between the source and target domains by transforming the appearance of images. Two complementary approaches are implemented. **Spatial CycleGAN** works directly on images in the pixel space and learns to convert source images into the style of the target domain, changing features like colour, texture, and overall appearance while trying to keep the object content the same. **Spectral CycleGAN** takes a different approach by working in the frequency domain — instead of modifying the full image, it mainly adjusts the low-frequency components (which control global appearance such as brightness and tone) while preserving high-frequency structural details like edges and shapes. This frequency-domain constraint makes the transformation more stable and helps maintain important object information.

3. **Task 2: Input-Space UDA**  
   This task evaluates five UDA methods on the classification task. **Source-only** serves as the baseline that directly applies a source-trained model to target data without any adaptation. **Spatial CycleGAN** and **Spectral CycleGAN** use the translated images from Task 1 as training data. **CyCADA** (Cycle-Consistent Adversarial Domain Adaptation) extends cycle-consistency to both the pixel and feature levels by adding a domain discriminator to the ResNet-50 embedding, with a gradient reversal layer forcing the classifier to learn domain-invariant features. **FDA** (Fourier Domain Adaptation) aligns source and target domains by directly swapping the low-frequency amplitude of the source image's spectrum with that of a target image, bypassing the need for adversarial training entirely.

4. **Task 1 Qualitative Observations (Art → Real World)**  
   As training progresses for **Spatial CycleGAN**, the model learns to neutralize the lighting and palettes of the images. At early stages, the lighting becomes more natural and colours appear more realistic, and as training progresses the textures become smoother and the edges more sharply defined. By epoch 100, the colour palette and saturation of the translated images replicate that of real world images while remaining spatially identical to the source art. For **Spectral CycleGAN**, the translated low-frequency image is merged with the preserved high-frequency image to produce an adaptive sharp image — this pipeline preserves sharp edges and fine details in high frequencies while shifting toward the Real World style in low frequencies. The combination produces a style-adapted image with sharp features intact.

5. **Task 2 Results and Analysis (Art → Real World)**  
   **CyCADA achieves the highest accuracy at 81.20%** with a Macro-F1 of 79.53, outperforming Source-only by +1.95% — the largest improvement among all methods. Spectral CycleGAN and Spatial CycleGAN follow closely at 79.89% and 79.69% respectively, both only marginally improving over Source-only (+0.64% and +0.44%). This near-parity with Source-only suggests that for the Art → Real World pair specifically, the domain gap is relatively small, meaning image-level translation methods provide limited additional benefit. The most striking result is **FDA at 74.39%, which underperforms Source-only by -4.87%**. This degradation confirms that FDA's blind spectral swapping introduces sufficient semantic distortion to harm classifier performance rather than help it, highlighting the importance of semantic-aware adaptation methods over blind spectral swapping in complex object datasets.

6. **Qualitative Method Comparison**  
   **Spatial CycleGAN** applies transformations across the entire pixel space, with objects like the bike and scissors showing minimal visible change while the alarm clock exhibits slight darkening and contrast increase. While structural integrity is maintained, the unconstrained nature of transformation means the model cannot distinguish between background style and object identity, risking semantic distortion. **Spectral CycleGAN** restricts transformations to low-frequency bands, targeting fine-grained properties like overall illumination and broad colour cast — the alarm clock's background shifts to a darker tone while the clock face brightens, reflecting a global illumination adjustment rather than object-level interference. **CyCADA** produces visibly the cleanest results across all samples — the alarm clock retains its dark background with no halo artifacts, the bike preserves its natural background colours, and the chair's fabric texture is entirely intact. This is because CyCADA's semantic constraint ensures object identity is not changed during translation, at the trade-off of significantly higher computational complexity. **FDA** shows halo-like glowing effects at the boundary of the frequency swap, with a tradeoff observed between β values (too small provides insufficient alignment; too large disregards semantic identity entirely, with β = 0.01 being suitable for this dataset).

7. **Worst-Performing Classes Analysis**  
   Since the Art → Real World pair consists of 65 classes, analysis of the worst-performing classes revealed three primary failure reasons: **intra-class visual diversity** (in classes like Toys and Notebooks), **inter-class visual similarity** (like TV/Monitor and Folder/Notebook/Clipboard), and general domain gap issues. Throughout these challenging classes, **CyCADA's semantic constraints proved to be the most beneficial**, providing the most consistent results, whereas FDA performed significantly worse in the majority of these weak classes. A noticeable observation was that while CyCADA is best overall, spectral methods can still be beneficial for certain object classes where global colour is the primary domain difference (such as TV or Marker), suggesting a case for hybrid or class-aware method selection.

## Key Results

### Task 2: UDA Classification Results (Art → Real World)

| Method | Accuracy (%) | Macro-F1 | Weighted-F1 |
|--------|-------------:|---------:|------------:|
| Source-Only | 79.25 | 77.61 | 78.99 |
| Spatial CycleGAN | 79.69 | 77.85 | 79.31 |
| Spectral CycleGAN | 79.89 | 78.40 | 79.55 |
| FDA | 74.39 | 72.52 | 73.73 |
| **CyCADA** | **81.20** | **79.53** | **80.80** |

## Future Scope

The results on Art → Real World suggest that method complexity does not guarantee better outcomes — the near-parity of CycleGAN methods with Source-only indicates that when the domain gap is modest, image-level translation offers limited additional benefit and semantic-aware feature-level alignment (as in CyCADA) provides the strongest gains. Future work could explore hybrid approaches that combine CyCADA's semantic consistency with the class-aware spectral adjustments that benefit specific object categories, potentially yielding a more robust adaptation strategy for datasets with varying intra-class and inter-class visual challenges.

## Academic Context
Completed during my Graduate studies at **Nanyang Technological University (NTU), Singapore**.

* **Course:** Advanced Computer Vision
* **Semester:** AY 2025/2026, Semester 2
* **Grade Achieved** A+

## Usage
**Note for current/future NTU students:** While this repository is public, please ensure you adhere to NTU's Academic Integrity Policy. This is intended as a reference for my personal portfolio; using this code for your own graded assignments is strictly prohibited by the University.
