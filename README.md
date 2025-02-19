ADD TO PYTHONPATH

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Environment

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install -r requirements.txt
```

### Run

```bash
accelerate launch scripts/train_composition.py
```
