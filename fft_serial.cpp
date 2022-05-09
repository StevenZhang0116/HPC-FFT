// Reference: Wesley Petersen and Peter Arbenz. “Introduction to parallel computing. A practical
// guide with examples in C”. In: (Jan. 2004)

// Complex binary radix (n = 2^m) FFT in Serial Version.
// Simply delete the openmp part of code from fft_openmp.cpp; rests are very similar

# include <cmath>
# include <cstdlib>
# include <iostream>
# include <iomanip>
# include <ctime>
# include <omp.h>
# include <fstream>
# include <string>
# include <vector>
# include <iterator>
using namespace std;

// step function, use the logic/parameters presented on p151 in Petersen's book
void step (int n, int mj, double a[], double b[], double c[], double d[], double w[], double sgn){
  double ambr, ambu;
  int ja, jb, jc, jd, jw;
  double wjw[2];
  int mj2 = 2 * mj;
  int lj = n / mj2;

  for (int j = 0; j < lj; j++){
    jw = j * mj;
    ja = jw;
    jb = ja;
    jc = j * mj2;
    jd = jc;

    wjw[0] = w[jw*2]; 
    wjw[1] = w[jw*2+1];

    if (sgn < 0.0) wjw[1] = -wjw[1];

    for (int k = 0; k < mj; k++){
      c[(jc+k)*2] = a[(ja+k)*2] + b[(jb+k)*2];
      c[(jc+k)*2+1] = a[(ja+k)*2+1] + b[(jb+k)*2+1];

      ambr = a[(ja+k)*2] - b[(jb+k)*2];
      ambu = a[(ja+k)*2+1] - b[(jb+k)*2+1];

      d[(jd+k)*2]   = wjw[0] * ambr - wjw[1] * ambu;
      d[(jd+k)*2+1] = wjw[1] * ambr + wjw[0] * ambu;
    }
  }
  return;
}

// copy function
void ccopy (int n, double x[], double y[]){
  for (int i = 0; i < n; i++){
    y[i*2] = x[i*2];
    y[i*2+1] = x[i*2+1];
   }
  return;
}

// performs complex FFT, use the logic/parameters presented on p128 in Petersen's book
void cfft2 (int n, double x[], double y[], double w[], double sgn){
  int m = (int)(log((double) n) / log(1.99));
  int mj = 1;
  int tgle = 1;

  step(n, mj, &x[0], &x[n], &y[0], &y[mj*2], w, sgn);

  if(n == 2) return;

  for(int j = 0; j < m - 2; j++){
    mj *= 2;
    if(tgle){
      step(n, mj, &y[0], &y[n], &x[0], &x[mj*2], w, sgn);
      tgle = 0;
    }
    else{
      step(n, mj, &x[0], &x[n], &y[0], &y[mj*2], w, sgn);
      tgle = 1;
    }
  }

  if(tgle) ccopy(n, y, x); 
  mj = n/2;
  step(n, mj, &x[0], &x[n], &y[0], &y[mj*2], w, sgn);

  return;
}

// Sets up sine and cosine table for complex FFT
void sincosine (int n, double w[]){
  double arg;
  const double pi = 3.14159265;
  double aw = 2.0*pi/((double)n);

  for (int i = 0; i < int(n/2); i++){
    arg = aw * ((double)i);
    w[i*2+0] = cos(arg);
    w[i*2+1] = sin(arg);
  }
  return;
}

// Generate uniformly distributed pseudorandom numbers
// Reference: https://stackoverflow.com/questions/18131612/how-to-generate-random-double-numbers-with-high-precision-in-c
double randomizer (double *seed){
  double d2 = 0.288212312e10;
  double t = (double) *seed;
  t = fmod(18712.0 * t, d2);
  *seed = (double) t;
  double value = (double) ((t - 1.0)/(d2 - 1.0));
  return value;
}

double cpu_time(void){
    double value = (double) clock() / (double) CLOCKS_PER_SEC;
    return value;
}


// main function
int main (){
  int ind; int indmax = 26;
  int n = 1;
  int nits = 10000;
  static double seed = rand() % 100 + 100;
  double *w; double wtime1; double wtime2; double wtime; 
  double *x; double *y; double *z;
  double z0; double z1;
  double sgn;

  int thread_num = 1;

  string dataname = "serial-data-"+to_string(thread_num)+".txt";
  cout << dataname << endl;

  ofstream myfile(dataname);

  cout << "  ====SERIAL VERSION OF FFT====" << endl;
  cout << "  Number of processors available = " << omp_get_num_procs () << "\n";
  cout << "  Number of threads = " << omp_get_max_threads () << "\n";
  omp_set_num_threads(thread_num);
  cout << "  Use " << thread_num << " threads" << endl;
  cout << "             N      NITS    Error         Time          Time/Call     MFLOPS\n";

  vector<double> vec_n;
  vector<double> vec_time;
  vector<double> vec_mflop;

  for(ind = 1; ind <= indmax; ind++){
    n *= 2;
    w = new double[n];
    x = new double[2 * n];
    y = new double[2 * n];
    z = new double[2 * n];

    int firstind = 1;

    for (int icase = 0; icase < 2; icase++){
      if (firstind){
        for (int i = 0; i < 2 * n; i += 2){
          z0 = randomizer(&seed); z1 = randomizer(&seed);
          x[i] = z0; z[i] = z0;
          x[i+1] = z1; z[i+1] = z1;
        }
      } 
      else{
        for (int i = 0; i < 2 * n; i += 2){
          z0 = 0.0; z1 = 0.0;
          x[i] = z0; z[i] = z0;
          x[i+1] = z1; z[i+1] = z1;
        }
      }

      sincosine(n, w);

      if (firstind){
        sgn = +1.0;
        cfft2 (n, x, y, w, sgn);
        sgn = -1.0;
        cfft2 (n, y, x, w, sgn);

        double fnm1 = 1.0 / (double) n;
        double error = 0.0;
        for (int i = 0; i < 2 * n; i += 2){
          error = error + pow (z[i] - fnm1 * x[i], 2) + pow (z[i+1] - fnm1 * x[i+1], 2);
        }
        error = sqrt(fnm1 * error);
        cout << "  " << setw(12) << n << "  " << setw(8) << nits << "  " << setw(12) << error;
        firstind = 0;
        vec_n.push_back(n);
      }
      else{
        wtime1 = omp_get_wtime();
        for (int it = 0; it < nits; it++){
          sgn = +1.0;
          cfft2 (n, x, y, w, sgn);
          sgn = -1.0;
          cfft2 (n, y, x, w, sgn);
        }
        wtime2 = omp_get_wtime();
        wtime = wtime2 - wtime1;
        double flops = (double) 2 * (double) nits * ((double) 5 * (double) n * (double) ind);
        double mflops = flops / 1.0E+06 / wtime;

        cout << "  " << setw(12) << wtime << "  " << setw(12) << wtime / (double)(2 * nits) << setw(10) << mflops << "\n";
        vec_time.push_back(wtime);
        vec_mflop.push_back(mflops);
      }
    }

    if((ind % 4) == 0) nits = nits / 10;
    if(nits < 1) nits = 1; 

    delete [] w;
    delete [] x;
    delete [] y;
    delete [] z;
  }

  for(auto i = 0; i < vec_n.size(); ++i){
    myfile << vec_n[i] << ";" << vec_time[i] << ";" << vec_mflop[i] << endl;
  }

  return 0;
}