# Sound Field Reconstruction using Diffuse Models

## Brief

The code in this repository is mainly inspired by [**Palette: Image-to-Image Diffusion Models**](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models).
After you prepared own data, you need to modify the corresponding configure file to point to your data.

### Training/Resume Training

1. Set `resume_state` of configure file to the directory of previous checkpoint. Take the following as an example, this directory contains training states and saved model:

```yaml
"path": { //set every part file path
	"resume_state": "experiments/training_sfr/checkpoint/100" 
},
```

2. Run the script:

```python
python run.py -p train -c config/sfr.json
```

### Test

1. Modify the configure file to point to your data following the steps in **Data Prepare** part.
2. Set your model path following the steps in **Resume Training** part.
3. Run the script:
```python
python run.py -p test -c config/sfr.json
```

## Main changes wrt original Palette implementation

* `dataset.py` - creation of sound field dataset, starting from frequency responses of rooms
* `mask.py` - random masking of the sound fields, based on the numebr of available mics
* `metric.py` - added nmse metric