# CEDR-PyTorch-Backend

A PyTorch C++ extension that replaces PyTorch's CPU implementations of selected operators with versions that run through the [CEDR](https://github.com/UA-RCL/CEDR) runtime. Once the extension is imported, matching operators in an unmodified PyTorch model are dispatched to CEDR runtime and schedule to hardware accelerators. Everything else runs on PyTorch's default CPU path. This approach allows end-users to flexibly and seamlessly deploy PyTorch models on FPGA based SoC platforms without any hardware expertise.

# Supported operations

| Operator | Dispatched to CEDR when | Otherwise |
|---|---|---|
| `aten::conv2d` | 4D `float32` weight, square kernel, stride 1, padding 1, dilation 1 | Falls back to PyTorch's convolution |
| `aten::linear` | `float32` input, batch size 1 | — |

# Requirements

- Linux (tested on Ubuntu 22.04)
- Python 3 (tested with 3.10)
- GCC with OpenMP support (tested with GCC 11.4)
- CMake (to build CEDR)
- PyTorch (tested with 2.14.1)

# Installation
- Create a virtual environment and activate it
```bash
python3 -m venv <desired_name>
source <desired_name>/bin/activate
```

- Install desired PyTorch version and pybind, tested with 2.14.1
- You can find installation instructions [here](https://pytorch.org/get-started/locally/).
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
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
cd CEDR-PyTorch-Backend
pip3 install --no-build-isolation .
```

- Install your model's requirements using pip

# Running With CEDR

- Clone pytorch branch of CEDR
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
cd ../applications/APIApps/pytorch/
make blocking
cp pytorch_app-x86.so ../../../build/
```

- Copy the test case ([`torch_cedr_test.py`](https://github.com/UA-RCL/CEDR/blob/pytorch/applications/APIApps/pytorch/torch_cedr_test.py)) to build folder and go to build folder
- **Submitted python script should be named as main.py**
```bash
cp torch_cedr_test.py ../../../build/main.py
cd ../../../build/
```

- To use the backend in your own script, import it after `torch`. Importing it is what registers the CEDR operators:
```python
import torch
import cedr_backend
```

- The backend loads `./libdash-rt/libdash-rt.so` relative to the current working directory, so run from the CEDR `build` folder

- Run with CEDR
```bash
cp ../daemon_config.json ./
./cedr -c ./daemon_config.json -l NONE &
./sub_dag -a ./pytorch_app-x86.so -n 1 -p 0
```

- Output should be
```text
Input shape: torch.Size([1, 1, 28, 28])
Output shape: torch.Size([1, 10])
tensor([[ 0.0238,  0.0264,  0.0252,  0.0656,  0.0072, -0.0572,  0.0727, -0.0219,
          0.0348,  0.0464]])
```

# Adding a New Operation

Adding an operation has two parts: an API call in CEDR that does the work, and a wrapper in this backend that routes the PyTorch operator to it. The steps below use ReLU as an example.

## 1. Add the API call to CEDR

Follow the CEDR tutorial: [Introducing a New API Call](https://github.com/UA-RCL/CEDR/blob/tutorial/CEDR_tutorial.md#6-introducing-a-new-api-call). This example assumes the new API call has the following signature:

```c
void DASH_RELU_flt(dash_re_flt_type* input, dash_re_flt_type* output, size_t size);
```

## 2. Declare the API call in this backend

Add the same signature to [`include/dash.h`](include/dash.h), inside the `extern "C"` block:

```c
void DASH_RELU_flt(dash_re_flt_type* input, dash_re_flt_type* output, size_t size);
```

## 3. Route the PyTorch operator to it

In [`cedr_backend.cpp`](cedr_backend.cpp), add a wrapper inside `namespace at { namespace native {` that takes the same arguments as the PyTorch operator (`aten::relu(Tensor self) -> Tensor`) and calls the new API. Inputs that CEDR doesn't handle should fall back to PyTorch. Don't call `at::relu` for the fallback: it would dispatch back to this wrapper and recurse.

If the data has to be reshaped before or after the CEDR call, put that code in a helper in [`cedr_ops.cpp`](cedr_ops.cpp) and declare it in [`include/cedr_ops.h`](include/cedr_ops.h). `conv2d` and `linear` work this way. ReLU needs no reshaping, so the wrapper calls CEDR directly.

```cpp
Tensor cedr_relu(const Tensor& self) {
    // Fall back to PyTorch for non-float32 inputs
    if (self.scalar_type() != at::kFloat) {
        return at::clamp_min(self, 0);
    }

    Tensor input = self.contiguous();
    Tensor output = at::empty_like(input);
    DASH_RELU_flt((dash_re_flt_type *)input.data_ptr<float>(), (dash_re_flt_type *)output.data_ptr<float>(), input.numel());
    return output;
}
```

Then register it at the bottom of the file, next to `conv2d` and `linear`:

```cpp
TORCH_LIBRARY_IMPL(aten, CPU, m) {
    m.impl("relu", at::native::cedr_relu);
}
```

The operator name and its signature are listed in PyTorch's [`native_functions.yaml`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml). This registration covers `torch.relu`, `F.relu` and `nn.ReLU()`. The in-place variants (`inplace=True`, `relu_`) are a separate operator, `relu_`.

## 4. Rebuild and run

Rebuild CEDR with the new API call, then reinstall this backend in your environment:

```bash
pip3 install --no-build-isolation .
```

Then run your model with CEDR as described in [Running With CEDR](#running-with-cedr).

# Citation

If you use this work, please cite:

```bibtex
@inproceedings{suluhan2023pytorch,
  title={PyTorch and CEDR: Enabling Deployment of Machine Learning Models on Heterogeneous Computing Systems},
  author={Suluhan, H Umut and Gener, Serhan and Fusco, Alexander and Ugurdag, H Fatih and Akoglu, Ali},
  booktitle={2023 20th ACS/IEEE International Conference on Computer Systems and Applications (AICCSA)},
  pages={1--8},
  year={2023},
  organization={IEEE}
}
```
