***Benchmarking Fast Fourier Transform Parallelization Algorithms using OpenMP and CUDA***

This repository serves for the final report for MATH-GA 2012 High Performance Computing in Spring 2022 instructed by Professor Benjamin Peherstorfer, NYU CIMS.

To run, just: 
```
make all
./fft_openmp
```

Detailed description of files:

**fft_openmp.cpp**: OpenMP implementation of FFT

**fft_serial.cpp**: Serial version of FFT

**analyze_result.py**: Analyze the results (running time/MFLOPS) for OpenMP/serial trails

**fft_stockham.cu**: CUDA Version of FFT

**option_pricing_with_fft.cpp**: FFT Application in option pricing
