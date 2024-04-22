# RewardOptimisation_in_MMSegmentation
This repository contains the necessary changes that have to be done to MMSegmentation in order to perform Reward Optimisation

## Decode head using reward optimisation
- `Code/reward_decode_head.py` to `mmsegmentation/mmseg/models/decode_heads/`. 
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
- d
- d