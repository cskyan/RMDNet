# RMDNet
RMDNet (RNA-aware Multi-branch Dung Beetle Optimization Network) is a deep learning framework designed for accurate prediction of RNA–protein binding sites. It integrates multi-scale sequence features from CNN, CNN-Transformer, and ResNet branches, and incorporates RNA secondary structure information using graph neural networks (GNN) with DiffPool. To improve prediction robustness, RMDNet applies an improved Dung Beetle Optimization (IDBO) algorithm to dynamically assign fusion weights across branches during inference.

📌 Features: Multi-branch architecture, structural graph integration, adaptive fusion, motif extraction, case study visualization.
# Data
📁 Dataset support: RBP-24, RBP-31, RBPsuite2.0

All datasets used in this study are publicly available. The RBP-24 dataset can be accessed at: http://www.bioinf.uni-freiburg.de/Software/GraphProt. 

The RBP-31 dataset is available at: https://github.com/mstrazar/iONMF. 

The RBPsuite2 dataset is available from Zenodo at: https://zenodo.org/records/14949530.
#  Environment & Dependencies
🛠️This project was developed and tested under the following environment:

- Python 3.7 or 3.8
- PyTorch 1.10+ (recommended)
- torch-geometric 2.0+ (with compatible PyG dependencies)
- scikit-learn >= 0.24
- tqdm
- numpy
- pickle (standard)
- argparse (standard)
- CUDA (for GPU acceleration, optional but recommended)

📦 Optional tools

RNAfold (from ViennaRNA package) is required to generate RNA secondary structures.
You can install it via:

<pre> sudo apt install vienna-rna </pre>

or download from: https://www.tbi.univie.ac.at/RNA/
# Usage
🧠 Model Training & Prediction

Use main.py to train the model and predict RNA–protein binding sites for a specific RBP:
<pre> python main.py </pre>

🎯 Motif Extraction

Use motif.py to extract interpretable sequence motifs from the first-layer CNN kernels after training:
<pre> python motif.py </pre>
This will generate and save sequence logos representing learned motifs for the specified RBP.

The parameters in the code are adjustable, with some specified in the DBO.py. Please adjust according to your supported environment.

