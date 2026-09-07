# Surf Pose Estimation with detectron2

Using the detectron2 pose estimation model from the [official tutorial page](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5) to capture my pose while surfing.

![img](img.JPG)

## Running locally (Apple Silicon)

The notebook was originally written for Colab (CUDA 10.1 + torch 1.6). It now
runs locally on CPU. Setup:

```bash
conda env create -f environment.yml
conda activate surf-detectron2

git clone https://github.com/facebookresearch/detectron2.git ~/Projects/detectron2
CC=clang CXX=clang++ ARCHFLAGS="-arch arm64" \
  pip install -e ~/Projects/detectron2 --no-build-isolation

python -m ipykernel install --user \
  --name surf-detectron2 --display-name "Surf Detectron2"
```

Then open the notebook and select the **Surf Detectron2** kernel.

Inference is pinned to CPU (`cfg.MODEL.DEVICE = "cpu"`) — keypoint R-CNN uses
ops that MPS does not implement. A single image takes a few seconds.
