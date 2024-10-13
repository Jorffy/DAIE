# DAIE

This is the official implementation of the paper "Dual-Level Adaptive Incongruity-Enhanced Model for Multimodal Sarcasm Detection", which is accepted by Neurocomputing. 
(https://doi.org/10.1016/j.neucom.2024.128689)


## Model Architecture

<div align=center>
<img src="DAIEmodel.png" width="85%" height="85%" />
</div>
The framework of dual-level adaptive incongruity-enhanced model (DAIE).

## Environment Requirements
The experiments were conducted on a single GeForce RTX 3090 GPU with 24GB memory. 
* Python 3.7.2
* PyTorch 1.8.0+cu111
* CUDA 11.2

To run the code, you need to install the requirements:
``` 
pip install -r requirements.txt
```

## Data Download
We evaluate our model using a publicly available multimodal sarcasm detection dataset. For the orginial dataset, see as https://github.com/headacheboy/data-of-multimodal-sarcasm-detection.

To run our code and for a fair comparison, we adhere to the preprocessing methods outlined in previous work. Please replace paths of datasets in *DATA_PATH* and *IMG_PATH* of `main.py` using your paths.

## Run Code

At last,  you can run the below code:

```shell
bash run.sh
```

## Papers for the Project & How to Cite

If you use or extend our work, please cite the paper as follows:
```
@article{wu2024dual,
  title={Dual-level adaptive incongruity-enhanced model for multimodal sarcasm detection},
  author={Wu, Qiaofeng and Fang, Wenlong and Zhong, Weiyu and Li, Fenghuan and Xue, Yun and Chen, Bo},
  journal={Neurocomputing},
  pages={128689},
  year={2024},
  publisher={Elsevier}
}
```