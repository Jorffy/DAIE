# DAIE

Code for "Dual-Level Adaptive Incongruity-Enhanced Model for Multimodal Sarcasm Detection".


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
