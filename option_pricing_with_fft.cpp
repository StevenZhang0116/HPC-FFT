#include <cmath>
#include <cstdlib>
#include <iostream>
#include <complex>
#include "fft_openmp.cpp"
using namespace std;

template <typename T>
T normalCDF(T value)
{
    return 0.5 * erfc(-value * M_SQRT1_2);
}

template <typename T>
complex<T> characteristic_function_of_normal_pdf(complex<T> u, T S_0, T mu, T sigma, T t)
{
    T x = -0.5 * sigma * sigma * t * u * u;
    T y = u * (log(S_0) + (mu - 0.5 * sigma * sigma) * t);
    complex<T> z = x + 1i * y;
    return exp<T>(z);
}

template <typename T>
complex<T> fourier_transform_of_c_T(T y, T a, T S_0, T mu, T sigma, T t)
{
    complex<T> u = y - 1j * (1 + a);
    complex<T> denominator = a + a * a + (2 * a + 1) * 1j * y - y * y;
    complex<T> Phi = characteristic_function_of_normal_pdf(u, S_0, mu, sigma, t);
    return Phi / denominator;
}

template <typename T>
complex<T> *_fft(T eta, long N, T a, T S_0, T mu, T sigma, T t)
{
    T b = M_PI / eta - log(S_0);

    int arr[N];
    for (long i = 0; i < N / 2; i++)
    {
        arr[2 * i] = 2;
        arr[2 * i + 1] = 4;
    }
    arr[0] = 1;

    complex<T> final_terms[N];
    for (long i = 0; i < N; i++)
    {
        final_terms[i] = eta * exp<T>(1j * b * eta * i) * fourier_transform_of_c_T(eta * i, a, S_0, mu, sigma, t) * arr[i] / 3.0;
    }

    T w[N];
    T x[2 * N];
    T y[2 * N];
    T result[N];
    for (long i = 0; i < N; i++)
    {
        x[2 * i] = final_terms[i].x;
        x[2 * i + 1] = final_terms[i].y;
    }


    cfft2 (N, x, y, w, 1);
    for (long i = 0; i < N; i++)
    {
        result[i] = y[2 * i];
    }

    return result;
}

template <typename T>
complex<T> *option_valuation_with_fft(T eta, long N, T a, T S_0, T mu, T sigma, T t)
{
    T lam = M_PI * 2 / N / eta;
    T log_S_0 = log(S_0);
    T k;
    T exp_term = exp(-mu * t);
    complex<T> *arr = _fft(eta, N, a, S_0, mu, sigma, t);
    complex<T> result[N];
    for (long i = 0; i < N; i++)
    {
        k = log_S_0 + lam * -N / 2 + i;
        result[i] = exp_term * exp(-a * k) / M_PI * arr[i];
    }
    return result;
}

template <typename T>
T call_BlackScholes(T S_0, T *K_arr, T tau, T r, T div, T sigma, long N)
{
    T d1, d2;
    T result[N];
    T sigma_sqrt_tau = sigma * sqrt(tau);
    T term = ((r - div) + sigma * sigma / 2.) * tau / sigma_sqrt_tau;
    T coeff1 = S_0 * exp(-div * tau);
    T coeff2 = K * exp(-r * tau);
    for (long i = 0; i < N; i++)
    {
        d1 = log(S_0 / K_arr[i]) / sigma_sqrt_tau + term;
        d2 = d1 - sigma_sqrt_tau;
        result[i] = coeff1 * normalCDF(d1) - coeff2 * normalCDF(d2);
    }
    return result;
}
