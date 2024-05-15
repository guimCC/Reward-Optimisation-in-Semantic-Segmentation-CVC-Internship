# Reward Optimization in Semantic Segmentation - CVC Internship

**Institution:** [Computer Vision Center (CVC)](https://www.cvc.uab.es/)
**Internship Period:** January 24 - June 24

## Repo Table of Contents
1. [Implementation Details](#Implementation)
2. [Project Overview Slides](#Resources/rewardOptimisation.md)
3. [Project Paper](#Todo)
4. [Experiments](#Todo) <!-- Raw experiment data and insights -->
5. [Utilities](#Implementation/Utilities/)

## Project Overview

This project aims to implement reward optimization in the domain of semantic segmentation within computer vision. The primary objective is to leverage this technique as a supplementary tool for domain adaptation to enhance the model's metric scores.

The theoretical approach is inspired by the paper ["Tuning Computer Vision Models with Task Rewards"](https://arxiv.org/pdf/2302.08242), which outlines a general methodology applicable to similar tasks.

The implementation is conducted within the [MMSegmentation framework](https://mmsegmentation.readthedocs.io/en/latest/), a platform known for its depth in runtime modifications and widespread use in the scientific community.

## Objectives

## Technologies Used

This project was primarily developed using PyTorch, a leading deep learning framework that facilitates building complex neural network architectures. PyTorch's dynamic computation graph enabled flexible and efficient model development and experimentation.

### Key Libraries
- **NumPy**: Heavily used for high-performance scientific computing and data analysis, particularly for manipulating large arrays and matrices of numeric data.
- **Pandas**: Employed for data manipulation and analysis, particularly useful during the data exploration phases for handling and processing data in a tabular form.

### Software and Tools
- **MMSegmentation**: An open source semantic segmentation toolbox based on PyTorch, utilized for its robust model structures and segmentation utilities.
- **Git**: Used for version control, allowing effective tracking of changes and collaboration across various phases of the project.
- **Conda**: Employed as a package and environment management system, which helped maintain consistency across development environments.

### Hardware and Infrastructure
- **CUDA**: Leveraged for GPU acceleration to facilitate efficient training of deep learning models, essential for handling complex computations and large datasets.
- **Distributed Training**: Implemented across multiple GPUs on remote servers, enhancing the training speed and scalability of the model development process.

These technologies combined to create a robust development environment that supported the advanced computational needs of the project, from model training to data analysis.


## Utilities

[Utilities](#Implementation/Utilities) contains several useful tools that have been used in this project.

### Frequency and Weight Computation

- [dataset_frequency_computation](Implementation/Utilities/dataset_frequency_computation.py)

  Computes the frequency of each class in a dataset, given its annotations root directory.

- [dataset_weight_computation](Implementation/Utilities/dataset_weight_computation.py)

  Using the frequencies, it computes the class weights with the following formulas:

  - $C_c$: Class counts
  - $T_c$: Total counts
  - $C_w$: Class weights

  $$C_w = \frac{1}{C_c + \text{CLS\_SMOOTH} \cdot T_c} \ \ \ \ \ \ \ \ C_w = \frac{N \cdot C_w}{\sum{C_w}} $$

### Experimentation Visualization

After performing a **MMSegmentation** training run under an example working directory `workdir_path`, the `workdir_path/id/vis_data/scalars.json` can be used to explore the experiment's results.

- [metrics_cleaner](Implementation/Utilities/Cleaner.ipynb)

  Given a `scalars.json` file, creates two CSV files, `loss.csv` and `metrics.csv`, which hold the relevant information for exploration in a treatable format.

- [experiment_exploration](Implementation/Utilities/ExperimentExploration.ipynb)

  Loads experiment scalars from `loss.csv` and `metrics.csv` and performs several plots. Some utilities include:
  - Plotting the **mIoU** at each validation step.
  - Plotting the **mIoU trend** competing with the current model.
  - Plotting the **regression line** of the **mIoU** values to see the trend.
  - Plotting the **smoothed mIoU** values, similar to what **TensorBoard** does.
