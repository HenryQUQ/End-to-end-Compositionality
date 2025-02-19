ADD TO PYTHONPATH

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install -r requirements.txt
```

### Run

```bash
accelerate launch scripts/train_composition.py
```
