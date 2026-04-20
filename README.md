# Reliable and Trustworthy AI Assignment 2

## Environment

```bash
conda create -n rtai-a2 python=3.10
conda activate rtai-a2
pip install -r requirements.txt
```

or

```bash
cd Reliable-and-Trustworthy-AI_2

./setup_and_run.sh
```

The PyTorch packages in `requirements.txt` use CUDA 11.8 wheels.

## DeepXplore

The original DeepXplore repository should be placed next to this repository:

```text
assignment2/
├── deepxplore/
└── Reliable-and-Trustworthy-AI_2/
```

I used the original DeepXplore implementation as the reference design and
implemented a PyTorch CIFAR-10/ResNet50 version in `test.py`. The implementation
keeps the two main DeepXplore ideas: differential behavior between models and
neuron coverage.

## Models

Two CIFAR-10 ResNet50 models are included under `models/`.

They were trained with different random seeds:

```bash
python train_models.py \
  --epochs 5 \
  --train-limit 10000 \
  --test-limit 2000 \
  --batch-size 256 \
  --num-workers 2 \
  --seeds 1 2 \
  --device cuda:0
```

## Run

```bash
./setup_and_run.sh
```

The run writes:

- `results/summary.json`
- `results/disagreement_01.png`
- `results/disagreement_02.png`
- `results/disagreement_03.png`
- `results/disagreement_04.png`
- `results/disagreement_05.png`

## Current Results

- Disagreement-inducing inputs: 30 / 30
- Model A neuron coverage at threshold 0.75: 0.343
- Model B neuron coverage at threshold 0.75: 0.347
- Average neuron coverage at threshold 0.75: 0.345

Coverage by threshold:

| Threshold | Model A | Model B | Average |
| --- | ---: | ---: | ---: |
| 0.20 | 1.000 | 1.000 | 1.000 |
| 0.50 | 0.966 | 0.966 | 0.966 |
| 0.75 | 0.343 | 0.347 | 0.345 |
| 0.90 | 0.085 | 0.092 | 0.089 |

Many disagreements appeared in visually similar or ambiguous CIFAR-10 groups,
especially animal classes and vehicle classes. Some original seed images already
caused disagreement, while others required small input changes from the
gradient-based search.
