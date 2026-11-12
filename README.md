# PyTorch_CEDR_backend

Tested with PyTorch >=1.13<=2.5

# Installation
- Create a pip environment and source it
```bash
python3 -m venv <desired_name>
source desired_name/bin/activate
```

- Install desired PyTorch version and pybind, this tutorial will use 2.5
- You can find installation instructions [here](https://pytorch.org/get-started/locally/).
```bash
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cpu
pip3 install pybind11
```

- Clone CEDR backend repository
```bash
git clone git@github.com:UA-RCL/CEDR-PyTorch-Backend.git # ssh
# or 
git clone https://github.com/UA-RCL/CEDR-PyTorch-Backend.git # https
```

- Install the library
```bash
cd PyTorch_CEDR_backend
python3 setup.py install
pip3 install .
```

- Install your model’s requirements using pip

# Running With CEDR

- Clone hcw-2024 branch of CEDR
```bash
cd ../
git clone -b pytorch git@github.com:UA-RCL/CEDR.git # ssh
# or
git clone -b pytorch https://github.com/UA-RCL/CEDR.git # https
```

- Build CEDR
```bash
cd CEDR
mkdir build
cd build
cmake ..
make
```

- Build the PyTorch application for CEDR in the sourced environment
```bash
cd ../applications/APIApps/pytorch_backend/
make blocking
cp pytorch_app-x86.so ../../../build/
```

- Copy test case to build folder and go to build folder
- **Submitted python script should be named as main.py**
```bash
cp torch_cedr_test.py ../../../build/main.py
cd ../../../build/
```

- Run with CEDR
```bash
./cedr -c ./daemon_config.json -l NONE &
./sub_dag -a ./pytorch_app-x86.so -n 1 -p 0
```

- Output should be
```bash
Input shape: torch.Size([1, 1, 28, 28])
Output shape: torch.Size([1, 10])
tensor([[ 0.0238,  0.0264,  0.0252,  0.0656,  0.0072, -0.0572,  0.0727, -0.0219,
          0.0348,  0.0464]])
```
