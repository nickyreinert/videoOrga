
## 1. Environment Setup

### 1.1 Install PyEnv and Python 3.11.8

1. Install build dependencies in WSL:

```bash
sudo apt update
sudo apt install -y build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev libncurses5-dev libncursesw5-dev \
libffi-dev liblzma-dev tk-dev
```

2. Install PyEnv:

```bash
curl https://pyenv.run | bash
```

3. Download Python 3.11.8 tarball from official source and copy to WSL:

```
https://www.python.org/ftp/python/3.11.8/Python-3.11.8.tgz
```

```bash
cp /mnt/c/Users/Nicky/Downloads/Python-3.11.8.tgz ~/
mkdir -p ~/.pyenv/cache
cp ~/Python-3.11.8.tgz ~/.pyenv/cache/
```

4. Install Python 3.11.8 via PyEnv:

```bash
pyenv install 3.11.8
pyenv global 3.11.8
```

### 1.2 Create / Activate Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 1.3 Install PyTorch with CUDA Support

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Verify GPU access:

```bash
python3 -c "import torch; print(torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected output:

```
12.6 True NVIDIA GeForce RTX 3070
```

### 1.4 Install Project Dependencies

```bash
pip install -r requirements.txt
```
