# Reward Optimisation in Semantic Segmentation - CVC Internship
**Institution:** [Computer Visioin Center (CVC)](https://www.cvc.uab.es/)
**Internship Period:** January 24 - June 24

## Repo Table of Contents
1. [Implementation Details](Implementation)
2. [Project Overview Slides](#Resources/rewardOptimisation.md)
3. [Project Paper](#Todo)
4. [Experiments](#Todo) <!-- Raw experiment data and insights-->
5. [Utilities](Implementation/Utilities/)

## Project Overview

This project aims to implement reward optimization in the semantic segmentation domain of computer vision. The primary objective is to leverage this technique as a supplementary tool for Domain Adaptation to enhance the model's metric scores.

The theoretical approach is inspired by the paper ["Tuning Computer Vision Models with Task Rewards"](https://arxiv.org/pdf/2302.08242), which outlines a general methodology applicable to similar tasks.

The implementation is conducted within the [MMSegmentation framework](https://mmsegmentation.readthedocs.io/en/latest/), a platform known for its depth in runtime modifications and widespread use in the scientific community.


## Objectives

## Technologies Used

## Utilities

[Utilities](Implementation/Utilities) contains some useful utilities that have been used on the project.

### Frequency and weight computation

- [dataset_frequency_computation](Implementation/Utilities/dataset_frequency_computation.py)

Computes the frequency of each class in a dataset given its annotations root directory.

- [dataset_weight_computation](Implementation/Utilities/dataset_weight_computation.py)

Using the frequencies, it computes the class weights using the following formula:

$C_c$: Class counts --- $T_c$: Total counts --- $C_w$: Class weights

$$C_w = \frac{1}{C_c + \text{CLS\_SMOOTH} \cdot C_w} \ \ \ \ \ \ \ \ C_w = \frac{N \cdot C_w}{\sum{C_w}} $$

### Experimentation visualisation

# TODO: wrap experiment functionallity with functions to be able to "tune" the type of output plot
