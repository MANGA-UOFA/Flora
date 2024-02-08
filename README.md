## 💐 Flora: Low-Rank Adapters Are Secretly Gradient Compressors

This is the official repository for the paper [Flora: Low-Rank Adapters Are Secretly Gradient Compressors](https://arxiv.org/abs/2402.03293). This repository contains the code for the experiments in the paper.

### Environment Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The current repository only supports JAX. We will add PyTorch support in the future.

### Replication

To replicate major experiments in the paper, run the following commands:

```bash
sh replicate.sh
```

You can also run individual experiments by selecting the corresponding script in the file `replicate.sh`.

### Citation

```bibtex
@article{hao2024flora,
  title={Flora: Low-Rank Adapters Are Secretly Gradient Compressors},
  author={Hao, Yongchang and Cao, Yanshuai and Mou, Lili},
  journal={arXiv preprint arXiv:2402.03293},
  year={2024}
}
```
