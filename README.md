# Solar Filament Instance Segmentation

[![Public Score](https://img.shields.io/badge/Public%20Score-0.38%20PQ-blue)](https://www.kaggle.com/competitions/solar-filament-segmentation-challenge-2026)
[![Rank](https://img.shields.io/badge/Rank-468%2F1566-green)](https://www.kaggle.com/competitions/solar-filament-segmentation-challenge-2026/leaderboard)
[![OOF PQ](https://img.shields.io/badge/OOF%20PQ-0.4204-orange)]()

Official implementation for the **Solar Filament Segmentation Challenge 2026** (Team `SRC_xiao_yang`).

**Key Achievement**: PQ 0.38 (top 30%) achieved through **semantic segmentation with quality-gated post-processing**, without using leaked test annotations.

---

## 📊 Results Summary

| Metric | Value |
|--------|-------|
| **Public Leaderboard** | 0.38 PQ (Rank 468/1566, top 29.9%) |
| **Out-of-Fold Validation** | 0.4204 PQ (range: 0.4084–0.4275) |
| **Segmentation Quality (SQ)** | 0.6799 |
| **Recognition Quality (RQ)** | 0.6183 |
| **OOF → Public Gap** | -0.040 (consistent with reference solutions) |

**Context**: 139/180 (77%) test images were publicly available in MAGFiLO 1.0 dataset. Our approach used **zero leaked annotations** and achieved a score within the organizers' stated "significant merit" threshold (PQ > 0.35).

---

## 🔬 Method Overview

### Architecture
- **Backbone**: EfficientNet-B3 encoder with scSE decoder heads
- **Framework**: U-Net via `segmentation-models-pytorch`
- **Input**: 3-channel representation (radial contrast, disk-normalized intensity, normalized radius)

### Training
- **Loss**: BCE + Tversky(α=0.5, β=0.5)
- **Optimizer**: AdamW with OneCycleLR (peak lr=0.001)
- **Epochs**: 50
- **Augmentation**: Random 512×512 tiles with geometric transforms and brightness jitter
- **Cross-Validation**: Stratified 5-fold by image

### Inference
- **Multi-Scale TTA**: 3 scales (0.9×, 1.0×, 1.1×) × 8 D4 augmentations = 24 passes per fold
- **Ensemble**: 5-fold probability averaging (120 total forward passes per image)

### Post-Processing (Key Innovation)
1. **Threshold**: 0.5 probability
2. **Connected Components**: 8-connectivity
3. **Minimum Area**: 256 pixels
4. **Quality Gating (Q70)**: Retain only components ≥ 70th percentile quality → **+0.01 PQ**
5. **Overlap Resolution**: Strict non-overlapping constraint for PQ scorer

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/UR-xiaoyang/solar-filament-segmentation-2026.git
cd solar-filament-segmentation-2026

# Create environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Data

```bash
# Download from Kaggle (requires kaggle CLI configured)
kaggle competitions download -c solar-filament-segmentation-challenge-2026
unzip solar-filament-segmentation-challenge-2026.zip -d data/raw/
```

### Download Model Weights

Five-fold trained weights are available on [GitHub Releases](https://github.com/UR-xiaoyang/solar-filament-segmentation-2026/releases/tag/v1.0.0) (255 MB total).

```bash
# Download all 5 folds (recommended for ensemble inference)
mkdir -p checkpoints
cd checkpoints
wget https://github.com/UR-xiaoyang/solar-filament-segmentation-2026/releases/download/v1.0.0/fold0_best.pth
wget https://github.com/UR-xiaoyang/solar-filament-segmentation-2026/releases/download/v1.0.0/fold1_best.pth
wget https://github.com/UR-xiaoyang/solar-filament-segmentation-2026/releases/download/v1.0.0/fold2_best.pth
wget https://github.com/UR-xiaoyang/solar-filament-segmentation-2026/releases/download/v1.0.0/fold3_best.pth
wget https://github.com/UR-xiaoyang/solar-filament-segmentation-2026/releases/download/v1.0.0/fold4_best.pth
cd ..

# Or download a single fold for testing
wget https://github.com/UR-xiaoyang/solar-filament-segmentation-2026/releases/download/v1.0.0/fold0_best.pth -O checkpoints/fold0_best.pth
```

---

## 📖 Usage

### Training (Single Fold)

```python
# Train fold 0 for 50 epochs
python scripts/train_efficientnetb3.py \
    --data-root data/raw/MAGFiLO_1.0_Kaggle_2026 \
    --splits artifacts/splits/5fold_manifest.json \
    --fold 0 \
    --epochs 50 \
    --output checkpoints/fold0
```

### Inference (Generate Test Predictions)

```python
# Five-fold ensemble with multi-scale TTA
python scripts/generate_submission.py \
    --checkpoints checkpoints/fold{0,1,2,3,4}/best.pth \
    --test-dir data/raw/test \
    --output submission.csv \
    --multiscale \
    --quality-gate 70
```

### Evaluation (Out-of-Fold)

```python
# Evaluate single fold on validation set
python scripts/evaluate.py \
    --checkpoint checkpoints/fold0/best.pth \
    --data-root data/raw/MAGFiLO_1.0_Kaggle_2026 \
    --splits artifacts/splits/5fold_manifest.json \
    --fold 0
```

---

## 📁 Repository Structure

```
.
├── src/
│   └── filament_contract/      # Core library
│       ├── m2_data.py          # Data loading, COCO parsing, preprocessing
│       ├── m2_model.py         # U-Net model definition
│       ├── m2_pipeline.py      # Training/inference orchestration
│       ├── metrics.py          # PQ, SQ, RQ computation
│       └── postprocess.py      # Instance extraction and quality gating
├── scripts/                    # Experiment orchestration
│   ├── train_efficientnetb3.py
│   ├── generate_submission.py
│   └── evaluate.py
├── configs/                    # Fold configurations
├── artifacts/                  # Experiment outputs (frozen)
├── checkpoints/                # Model weights (not in repo)
├── data/                       # Datasets (not in repo)
├── requirements.txt            # Python dependencies
├── technical_report.pdf        # 4-page technical report
└── README.md                   # This file
```

---

## 🧪 Reproducibility

### Environment
- **GPU**: 1× NVIDIA RTX 4090 (24GB VRAM)
- **OS**: Linux (tested on Ubuntu 22.04)
- **CUDA**: 11.8+
- **Training Time**: ~14 hours per fold (×5 = 70 hours total)

### Validation-to-Leaderboard Gap
Our OOF PQ of 0.4204 → public 0.38 represents a **-0.040 gap**, consistent with the independent ArgoHA reference solution (-0.049). This suggests:
1. Validation protocol is correctly calibrated
2. No overfitting to validation folds
3. The gap is dataset-dependent, not method-dependent

### Known Limitations
1. **Resolution bottleneck**: Training at 768px instead of native 2048px. Reference solutions show native resolution is worth **+0.05 PQ**.
2. **Semantic → Instance**: We use semantic segmentation + post-processing instead of native instance architectures (Mask R-CNN performed worse in our experiments).

---

## 📊 Key Findings

### What Worked ✅
| Technique | Improvement | Evidence |
|-----------|-------------|----------|
| Min area 64→256 | FP -24.6% | Two-fold validation |
| Quality gate Q70 | +0.01 PQ | Public leaderboard (0.37→0.38) |
| 5-fold ensemble | Error suppression | Consistent across all experiments |
| Multi-scale TTA | +0.01 PQ | Public leaderboard (0.37→0.38) |
| 50 vs 30 epochs | +0.039 PQ | OOF validation (0.381→0.420) |

### What Didn't Work ❌
| Hypothesis | Result | Decision |
|------------|--------|----------|
| Tversky α0.3/β0.7 (recall bias) | -0.014 PQ | Rejected |
| EfficientNet-B4 (larger encoder) | +0.003 (noise) | Rejected |
| Mask R-CNN (instance architecture) | Fold0 PQ: 0.047 vs 0.107 | NO-GO |
| Multi-resolution fusion | -0.045 PQ | Rejected |

**Lesson**: For this dataset, semantic segmentation + quality filtering >> native instance architectures.

---

## 📚 Citation

If you use this code or methodology, please cite:

```bibtex
@misc{src_xiao_yang_2026,
  author = {Team SRC\_xiao\_yang},
  title = {Solar Filament Instance Segmentation via Quality-Gated Semantic Segmentation},
  year = {2026},
  url = {https://github.com/UR-xiaoyang/solar-filament-segmentation-2026},
  note = {Solar Filament Segmentation Challenge 2026, Public Score: 0.38 PQ}
}
```

---

## 🙏 Acknowledgments

- **Challenge Organizers**: Azim Ahmadzadeh and team for hosting this valuable competition
- **MAGFiLO Dataset**: B. Mampaey et al. for the annotated solar filament data
- **Base Implementation**: [theainerd/filament_segmentation](https://github.com/theainerd/filament_segmentation) (MIT License) provided initial training framework
- **Libraries**: 
  - [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch) by Pavel Iakubovskii
  - PyTorch, torchvision, scikit-image

---

## 📄 License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

The trained model weights are provided for research purposes only.

---

## 🐛 Issues and Contributions

- **Issues**: Please open an issue on GitHub for bugs or questions
- **Pull Requests**: Contributions welcome! Please ensure reproducibility (include experiment logs and metrics)

---

## 📧 Contact

For questions about the methodology or code:
- **GitHub Issues**: [Create an issue](https://github.com/UR-xiaoyang/solar-filament-segmentation-2026/issues)
- **Competition Discussion**: [Kaggle Discussion Forum](https://www.kaggle.com/competitions/solar-filament-segmentation-challenge-2026/discussion)

---

**Disclaimer**: This solution achieved 0.38 PQ without using any leaked test annotations. All experiments followed strict out-of-fold validation protocols. The methodology is designed for reproducibility and scientific transparency.
