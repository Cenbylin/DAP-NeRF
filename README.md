# DAP-NeRF

> __Dynamic Appearance Particle Neural Radiance Field__  
> Ancheng Lin, Yusheng Xiang, Jun Li, Mukesh Prasad\
> __[Paper (arxiv)](https://arxiv.org/abs/2310.07916)__

![block](./imgs/render_traj.gif)

DAP-NeRF introduces particle-based representation to model the motions of visual elements in a dynamic 3D scene. DAP-NeRF consists of superposition of a static field and a dynamic field. The dynamic field is quantised as a collection of *appearance particles*, which carries the visual information of a small dynamic element in the scene and is equipped with a motion model. All components, including the static field, the visual features and motion models of the particles, are learned from monocular videos without any prior geometric knowledge of the scene.

## Notes
**TODO:** Clean codes thoroughly.

**10/6/2024:** Codes realeased.

## Requirements
```shell
pip install -r requirements.txt
```

## Data Preparation
```
├── data
│   ├── d_nerf
│   │   ├── mutant
│   │   ├── standup 
│   │   ├── ... 
│   ├── vrig
│   ├── NHR
│   ├── ...
```
**For synthetic dataset (d_nerf):**  
The dataset provided in [D-NeRF](https://github.com/albertpumarola/D-NeRF) is used. You can download the dataset from [dropbox](https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?dl=0).

**For HyperNeRF dataset (vrig):**  
The dataset provided in [HyperNeRF](https://github.com/google/hypernerf) is used. You can download scenes from [Hypernerf Dataset](https://github.com/google/hypernerf/releases/tag/v0.1) and organize them as [Nerfies](https://github.com/google/nerfies#datasets).

**For NHR dataset:**  
The dataset provided in [NHR](https://wuminye.github.io/NHR/datasets.html) is used. The processed scenes can be downloaded from [here](https://drive.google.com/drive/folders/16-sNCc28G8M6MfE6n79XlusZkE9K9Je_?usp=sharing).

## Training & Evaluation
For Synthetic or HyperNeRF scenes, run 
```shell
python run_dap.py --config configs/(dnerf|vrig)/xxx.py --render_test --render_video --eval_psnr --eval_ssim --eval_lpips_vgg
``` 

For NHR scenes, run 
```shell
python run_nhr_dap.py --config configs/nhr/xxx.py --render_test --render_video --eval_psnr --eval_ssim --eval_lpips_vgg
```

## Acknowledgements
This repository is partially based on [TiNeuVox](https://github.com/hustvl/TiNeuVox) and [nerfacc](https://github.com/nerfstudio-project/nerfacc). Thanks for their awesome works.