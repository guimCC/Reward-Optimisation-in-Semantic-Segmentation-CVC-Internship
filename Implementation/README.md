# Reward Optimisation implementation in MMSegmentation
This section of the repository contains modifications necessary for implementing Reward Optimization in MMSegmentation.

## Decode Head Using Reward Optimisation
- Add `Code/reward_decode_head.py` to `mmsegmentation/mmseg/models/decode_heads/`.
- For each **decode head implementation** you plan to use, such as **ASPPHead**, **FCNHead**, etc.:
  - Create a copy of `mmsegmentation/mmseg/models/decode_heads/<HeadName>_head.py` and rename it to `mmsegmentation/mmseg/models/decode_heads/reward_<HeadName>_head.py`.
  - Import using `from .reward_decode_head.py import RewardDecodeHead`.
  - Modify the **class declaration** to:
  ```python
    @MODELS.register_module()
    class Reward<HeadName>Head(RewardDecodeHead):
        ...
    ```
  - In `mmsegmentation/mmseg/models/decode_heads/__init__.py`, import the new module:
    - Add `from .reward_<HeadName>_head import Reward<HeadName>Head`.
    - Add `'Reward<HeadName>Head'` to the `__all__` list.

## Custom Loss Function and Reward Computation
- **Reward Computation**
  - To optimise the **mIoU** metric, add `Code/iou_metric_loss.py` to `mmsegmentation/mmseg/models/losses/`.
  - Register the new loss in `mmsegmentation/mmseg/models/losses/__init__.py`:
    - `from .iou_metric_loss.py import IoUMetricLoss`.
    - Add `'IoUMetricLoss'` to `__all__`.
- **New loss function**
  - To incorporate a reward factor into the **cross entropy** function, add `Code/cross_entropy_reward_loss.py` to `mmsegmentation/mmseg/models/losses/`.
  - Register the new loss in `mmsegmentation/mmseg/models/losses/__init__.py`:
    - `from .iou_metric_loss.py import CrossEntropyRewardLoss`.
    - Add `'CrossEntropyRewardLoss'` to `__all__`.

## Configurations Using Reward Optimisation
When setting up an **experimentation** or **training** using **reward optimisation**, consider the following:
- **Load a pretrained model**: According to the referenced paper, it is optimal to fine-tune a model that has been pretrained using the **Cross Entropy** loss function. To load a `.pkg` weights file for fine-tuning, add the following to the **model's config file**:
  ```python
  model = dict(
    type='EncoderDecoder',
    init_cfg=dict(type='Pretrained', checkpoint='<path_to_.pkg_file'),
    ...
  ```
- **Use New Decode Head and Loss Function**: For a model using the original **Decode Head** and pretrained with the **CrossEntropy** loss:
  ```python
  ...
  decode_head=dict(
    type='<HeadName>Head',
    ...
    loss_decode=dict(
      type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
  ...
  ```
  Replace with the new modules:
  ```python
  ...
  decode_head=dict(
    type='Reward<HeadName>Head',
    ...
    loss_decode=dict(
      type='CrossEntropyRewardLoss', use_sigmoid=False, loss_weight=1.0)),
  ...
  ```
- **Fine tunning and param sheduler**: Adjust the scheduler for fine-tuning
  - **Learning rate of optimiser**: Start with the learning rate at wich the pretraining concluded
  - **Learning policy**: Considering the modifications to the **loss function**, a decaying learning rate is generally beneficial to stabilize and improve model performance.