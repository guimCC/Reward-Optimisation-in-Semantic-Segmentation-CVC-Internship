---
marp: true
---

# Reward optimisation
## Rosa Sensat intership at CVC Research Journal

---

# Semantic segmentation
- Definition with examples (photo VS ground truth)
- Real (cityscapes, mapillary, easyportrait, ...) vs Synthetic datasets (GTA, ...)
- Domain adaptation (General)
- Domain adaptation: From Synthetic to Real datasets -> Transfer learning
- Cityscapes as the main problem to solve

---
# Reinforcement learning
- Brief introduction to RL ?

---

# Optimisation of non differentiable functions
- mIoU as the metric funcion
- Monte Carlo Gradient Estimation
- New Loss function for pretrained models
- Reduction of variance

---

# Implementation details
- Semantic segmentation model structure (Encoder, Decode - Heads)
- Reward computation
- New loss function

---

# Experimentation
- Overview of the different degrees of freedom (Scheduler, LR, batch-size, ...)
- General implementation results ? -> Reduction of variance
- Learning rate and sheduler results
- Batch size and baseline computation
- Model structure (Auxiliary head)

---

# Results and future work
- Achievements on different datasets
- Future work