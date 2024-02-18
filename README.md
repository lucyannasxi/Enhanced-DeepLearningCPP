# Enhanced Deep Learning Library in C++

This is an expanded version of an original toy deep learning library, now written in C++ and also supports CUDA for efficient calculations.

## Getting Started

To start with this, clone this new repository.

```
git clone https://github.com/lucyannasxi/Enhanced-DeepLearningCPP.git
```

### Prerequisites

- CUDA 9.0
- g++ 7.3.0

Additional prerequisites for development:
- clang-format 6.0.0
- clang-tidy 6.0.0

### Installing

To compile just run make from the main directory:
```
make -j4
```
This will compile everything: library, tests, and samples.

## Running the tests

```
./build/test
```

You can also run a debug version:
```
./build/test_debug
```

## Samples

Source code for samples can be found in the **samples** directory. To run them you have to add the compiled library to paths:
```
export LD_LIBRARY_PATH=/path/to/repository/build:$LD_LIBRARY_PATH
```

Example for running the mnist sample:
```
./build/sample_mnist
```

## Contributors

* **lucyannasxi**<br><br>## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details