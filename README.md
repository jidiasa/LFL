# Reliable Perpetual Scene Generation via Agentic Validation and Refinement

## with a Local Feedback Loop (LFL)

This repository contains the official implementation of **“Reliable Perpetual Scene Generation via Agentic Validation and Refinement with Local Feedback Loop.”**  
We introduce a large-language-model (LLM) agent that performs **global back-tracking corrections** across **all keyframes** in long-range scene synthesis.  
The `RSG/` directory houses the core framework and the global refinement logic.  
If you already have a pipeline that **first generates keyframes and then produces full scenes**, you can integrate LFL by **over-riding only the `generate` and `modify` methods**.

An end-to-end example based on **WonderJourney + LFL** is provided in `example/`.

---

## Getting Started

### Clone the repo

```bash
git clone https://github.com/jidiasa/LFL.git
cd LFL/example
```

### Create and activate a Conda environment

```bash
conda create -n lfl_wonderjourney python=3.9 -y
conda activate lfl_wonderjourney
```

### 3. Install PyTorch & TorchVision (CUDA 12.1 wheels)

```bash
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
```

### Install PyTorch3D (for differentiable rendering)

```bash
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install -c pytorch3d pytorch3d
```

### Install the remaining Python dependencies

```bash
pip install -r requirements.txt
```

### 6. Set your OpenAI API key

```bash
export OPENAI_API_KEY='your_api_key_here'
```

### Download MiDaS weights (for depth estimation)

```bash
wget https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt
```

### Run the demo (village scene)

```bash
python RSG.py --example_config config/village.yaml
```

---

## Example: WonderJourney + LFL

We provide a ready-to-run configuration that plugs LFL into WonderJourney’s keyframe-to-scene pipeline.  
Simply execute the command above and inspect the generated scene assets in `outputs/`.

---

## Results

Our LFL-enhanced model applies **Gaussian scene optimization** on top of WonderJourney.  
While convergence is slower, the final quality surpasses WonderWorld on key perceptual metrics.  
See the `results/` folder for qualitative comparisons and quantitative scores.  

---
