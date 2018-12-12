#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
      fprintf(stderr, "GPUassert: %s %s %dn", cudaGetErrorString(code), file, line);
      if (abort) { exit(code); }
    }
}

__global__
void calcF(int nAtoms, long nDiff, 
  double *diffPos3D0, double *diffPos3D1, double *diffPos3D2,
  double *atomOccu, int *atomType,
  double *atomPos3D0, double *atomPos3D1, double *atomPos3D2,
  double *atomDW3D0, double *atomDW3D1, double *atomDW3D2,
  double *atomicF, size_t atomicFpitch,
  double *outFRe, double *outFIm)
{
  long totalThreadsX = (blockDim.x * gridDim.x);
  long totalThreadsY = (blockDim.y * gridDim.y);
  int calcManyX = ceil((double)nDiff / totalThreadsX);
  int calcManyY = ceil((double)nAtoms / totalThreadsY);

  for (int threadRunX = 0; threadRunX < calcManyX; ++threadRunX)
  {
    for (int threadRunY = 0; threadRunY < calcManyY; ++threadRunY)
    {
      long diffIndex = threadIdx.x + (blockDim.x * blockIdx.x) + threadRunX * totalThreadsX;
      long atomIndex = threadIdx.y + (blockDim.y * blockIdx.y) + threadRunY * totalThreadsY;

      if (atomIndex < nAtoms && diffIndex < nDiff)
      {
        double dotProduct = 0.0;
        double atomicFExpFact = 0.0;
        double* thisAtomicF = (double*)(atomType[atomIndex] * atomicFpitch + (char*)atomicF);
        dotProduct += atomPos3D0[atomIndex] * diffPos3D0[diffIndex];
        dotProduct += atomPos3D1[atomIndex] * diffPos3D1[diffIndex];
        dotProduct += atomPos3D2[atomIndex] * diffPos3D2[diffIndex];
        atomicFExpFact += diffPos3D0[diffIndex] * diffPos3D0[diffIndex];
        atomicFExpFact += diffPos3D1[diffIndex] * diffPos3D1[diffIndex];
        atomicFExpFact += diffPos3D2[diffIndex] * diffPos3D2[diffIndex];

        // Lambda needs to be included here if diffPos3D is not in Q space!
        atomicFExpFact = atomicFExpFact / (16. * M_PI * M_PI);
        // not sure if & is needed
        // might be possible to speed up by summing up block wise and than sum blocks
        double atomicFRe = 1.0;
        double atomicFIm = 0.0;
        // double* thisBaseFIm = (double*)((char*)baseFIm + diffIndex * baseFImpitch);
        // atomicAdd(&outFRe[diffIndex], (thisBaseFRe[atomType[atomIndex]] * atomOccu[atomIndex] * cos(dotProduct)) );
        double cosDot = cos(dotProduct);
        double sinDot = sin(dotProduct);
        double resultFRe = atomOccu[atomIndex] *((atomicFRe * cosDot)+(atomicFIm * sinDot));
        double resultFIm = atomOccu[atomIndex] *((atomicFRe * sinDot)+(atomicFIm * cosDot));
        atomicAdd(&outFRe[diffIndex], resultFRe );
        atomicAdd(&outFIm[diffIndex], resultFIm );
      }
    }
  }
}

int main(int argc, char** argv) 
{
  long nAtoms = atol(argv[1]); // along thread.y
  long nDiff = atol(argv[2]); // along thread.x
  int nAtomTypes = 3; // Fe, O, Ir

  // determine compute structure
  int nTx = 64;
  int nTy = 8;
  if (nDiff < nTx){
    nTx = nDiff;
  }
  if (nAtoms < nTy) { 
    nTy = nAtoms;
  }
  int nBx = floor(nDiff / nTx);
  int nBy = floor(nAtoms / nTy);
  int maxBlocks = 500000;
  if (nBx > maxBlocks) {
    nBx = maxBlocks;
  }
  dim3 threadsPerBlock(nTx, nTy);
  dim3 numBlocks(nBx, nBy);

  std::cout << "#Atoms: " << nAtoms << ", #Diffractions: " << nDiff << ", #AtomTypes: " << nAtomTypes << std::endl;
  std::cout << "Blocks: " << nBx << " x " << nBy << " Threads: " << nTx << " x " << nTy << std::endl;
  float memGuess = (nDiff*(3+4)*sizeof(double)+nAtoms*4*sizeof(double)+nAtoms*1*sizeof(int));
  std::cout << "MemUsage: " << memGuess*1e-6 << "MB" << std::endl;

  // create GPU pointer
  double  *atomicF;
  int     *atomType = new int[nAtoms];
  double  *atomOccu = new double[nAtoms];
  double  *atomPos3D0 = new double[nAtoms];
  double  *atomPos3D1 = new double[nAtoms];
  double  *atomPos3D2 = new double[nAtoms];
  double  *atomDW3D0 = new double[nAtoms];
  double  *atomDW3D1 = new double[nAtoms];
  double  *atomDW3D2 = new double[nAtoms];
  double  *diffPos3D0 = new double[nDiff];
  double  *diffPos3D1 = new double[nDiff];
  double  *diffPos3D2 = new double[nDiff];
  double  *diffFstaticRe = new double[nDiff];
  double  *diffFstaticIm = new double[nDiff];
  double  *diffFunitRe = new double[nDiff];
  double  *diffFunitIm = new double[nDiff];
  // create Host data structures
  double  atomicFValues[nAtomTypes][9];
  size_t  atomicFpitch;
  // alloc GPU mem
  gpuErrchk(cudaMallocPitch(&atomicF, &atomicFpitch, 9 * sizeof(double), nAtomTypes));
  gpuErrchk(cudaMallocManaged(&atomType, nAtoms*sizeof(int)));
  gpuErrchk(cudaMallocManaged(&atomOccu, nAtoms*sizeof(double)));
  gpuErrchk(cudaMallocManaged(&atomPos3D0, nAtoms*sizeof(double)));
  gpuErrchk(cudaMallocManaged(&atomPos3D1, nAtoms*sizeof(double)));
  gpuErrchk(cudaMallocManaged(&atomPos3D2, nAtoms*sizeof(double)));
  gpuErrchk(cudaMallocManaged(&atomDW3D0, nAtoms*sizeof(double)));
  gpuErrchk(cudaMallocManaged(&atomDW3D1, nAtoms*sizeof(double)));
  gpuErrchk(cudaMallocManaged(&atomDW3D2, nAtoms*sizeof(double)));
  gpuErrchk(cudaMallocManaged(&diffPos3D0, nDiff*sizeof(double)));
  gpuErrchk(cudaMallocManaged(&diffPos3D1, nDiff*sizeof(double)));
  gpuErrchk(cudaMallocManaged(&diffPos3D2, nDiff*sizeof(double)));
  gpuErrchk(cudaMallocManaged(&diffFstaticRe, nDiff*sizeof(double)));
  gpuErrchk(cudaMallocManaged(&diffFstaticIm, nDiff*sizeof(double)));
  gpuErrchk(cudaMallocManaged(&diffFunitRe, nDiff*sizeof(double)));
  gpuErrchk(cudaMallocManaged(&diffFunitIm, nDiff*sizeof(double)));

  // Generate some test data:
  std::cout << "Init data " << std::endl;
  int newAtomType[] = {0,1,1,0,0,1,1,0};
  double newAtomOccu[] = {0.9, 1.0, 0.9, 1.0,
                       0.9, 1.0, 0.9, 1.0};
  double newAtomPos3D[8][3] = {{0.0,0.5,0.0},{0.5,0.5,0.0},
                           {0.0,0.0,0.0},{0.5,0.0,0.0},
                           {0.25,0.5,0.5},{0.75,0.5,0.5},
                           {0.25,0.0,0.5},{0.75,0.0,0.5}};
  for (int index = 0; index < nAtoms; ++index)
  {
    atomType[index] = newAtomType[index];
    atomOccu[index] = newAtomOccu[index];
    atomPos3D0[index] = newAtomPos3D[index][0];
    atomPos3D1[index] = newAtomPos3D[index][1];
    atomPos3D2[index] = newAtomPos3D[index][2];
  };
  for(int index = 0; index < nDiff; ++index)
  {
    diffPos3D0[index] = floor((double)index / 10.);
    diffPos3D1[index] = ceil((double)index / 20.);
    diffPos3D2[index] = (0.1 * (index % 10));

    diffFstaticRe[index] = 0.0;
    diffFstaticIm[index] = 0.0;
    diffFunitRe[index] = 0.0;
    diffFunitIm[index] = 0.0;
  };

  // copy
  std::cout << "Copy memory " << std::endl;
  gpuErrchk(cudaMemcpy2D(atomicF, atomicFpitch, atomicFValues, 9*sizeof(double), 9*sizeof(double), nAtomTypes, cudaMemcpyHostToDevice));

  std::cout << "Start calcF " << std::endl;
  // Run 
  calcF<<<numBlocks, threadsPerBlock>>>(nAtoms, nDiff, 
    diffPos3D0, diffPos3D1, diffPos3D2,
    atomOccu, atomType,
    atomPos3D0, atomPos3D1, atomPos3D2,
    atomDW3D0, atomDW3D1, atomDW3D2,
    atomicF, atomicFpitch,
    diffFunitRe, diffFunitIm);

  // Wait for GPU to finish before accessing on host
  gpuErrchk(cudaDeviceSynchronize());

  // Test
  float output = 0.0;
  std::cout << std::fixed;
  std::cout.precision(2);
  for(int index = 0; index < 100; ++index)
  {
    output = (pow(diffFunitRe[index], 2) + pow(diffFunitIm[index], 2));
    std::cout << output << " ";
    if ((index+1) % 10 == 0) {
      std::cout << std::endl;
    }
  }
  std::cout << std::endl;

  // Free memory
  cudaFree(atomicF);
  
  std::cout << "Fin " << std::endl;
  return 0;
}