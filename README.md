# RewardOptimisation_in_MMSegmentation
This repository contains the necessary changes that have to be done to MMSegmentation in order to perform Reward Optimisation.

This project is inspired by the work done on: **insert paper link** 

## Decode head using reward optimisation
- Add `Code/reward_decode_head.py` to `mmsegmentation/mmseg/models/decode_heads/`
- For each **decode head implementation** you are willing to use in your code, such as **ASPPHead**, **FCNHead**, ...
  - Create a copy of `mmsegmentation/mmseg/models/decode_heads/<HeadName>_head.py` called `mmsegmentation/mmseg/models/decode_heads/reward_<HeadName>_head.py`
  - Import `from .reward_decode_head.py import RewardDecodeHead`
  - Change the **class declaration** to:
  ```python
    @MODELS.register_module()
    class Reward<HeadName>Head(RewardDecodeHead):
        ...
    ```
  - At `mmsegmentation/mmseg/models/decode_heads/__init__.py` import the new module like so:
    - Add `from .reward_<HeadName>_head import Reward<HeadName>Head`
    - Add `'Reward<HeadName>Head'` to the `__all__` list.

## Custom loss function and reward computation
- **Reward Computation**
  - In order to perform the **reward computation** to optimise for the **mIoU** metric function, add `Code/iou_metric_loss.py` to `mmsegmentation/mmseg/models/losses/`
  - Register the **new loss** at `mmsegmentation/mmseg/models/losses/__init__.py`
    - `from .iou_metric_loss.py import IoUMetricLoss`
    - Add `'IoUMetricLoss'` to `__all__`
- **New loss function**
  - To apply the new **loss function** wich multiplies the **cross entropy** function by a reward factor, add `Code/cross_entropy_reward_loss.py` to `mmsegmentation/mmseg/models/losses/`
  - Register the **new loss** at `mmsegmentation/mmseg/models/losses/__init__.py`
    - `from .iou_metric_loss.py import CrossEntropyRewardLoss`
    - Add `'CrossEntropyRewardLoss'` to `__all__`

## Configs using reward optimisation
When defining an **experimentation** or a **training run** for a model where it's desidered to use the **reward optimisation** technique. There are some things that have to be taken into consideration.
- **Load a pretrained module**: Following the indications from the paper, this technique gives the best performance to **fine tune** a model that has already been **pretrained** with the **Cross Entropy** loss function. In order to load a `.pkg` weights file to perform the fine tunning, add:
  ```python
  model = dict(
    type='EncoderDecoder',
    init_cfg=dict(type='Pretrained', checkpoint='<path_to_.pkg_file'),
    ...
  ```
  At the **config file** holding the **model settings**
- 