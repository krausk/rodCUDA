#include<stdio.h>
#include <iostream>
#include <math.h>

// includes CUDA
#include<cuda.h>
#include <cuda_runtime.h>
#include<device_launch_parameters.h>

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
  double *diffPos3D, size_t diffPos3Dpitch,
  double *atomPos3D, size_t atomPos3Dpitch,
  // double *atomDW3D, size_t atomDW3Dpitch,
  double *atomOccu, int *atomType,
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
        double* thisAtomPos3D = (double*)((char*)atomPos3D + atomIndex * atomPos3Dpitch);
        double* thisDiffPos3D = (double*)(diffIndex * diffPos3Dpitch + (char*)diffPos3D);
        double* thisAtomicF = (double*)(atomType[atomIndex] * atomicFpitch + (char*)atomicF);
        for (int direction = 0; direction < 3; ++direction)
        {
          dotProduct += thisAtomPos3D[direction] * thisDiffPos3D[direction];
          atomicFExpFact += thisDiffPos3D[direction] * thisDiffPos3D[direction];
        }
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

  int nTx = 64;
  int nTy = 8;
  if (nDiff < nTx)
  {
    nTx = nDiff;
  }
  if (nAtoms < nTy)
  { 
    nTy = nAtoms;
  }
  dim3 threadsPerBlock(nTx, nTy);
  int nBx = floor(nDiff / threadsPerBlock.x);
  int nBy = floor(nAtoms / threadsPerBlock.y);
  if (nBx > 20000)
  {
    nBx = 20000;
  }
  dim3 numBlocks(nBx, nBy);

  std::cout << "#Atoms: " << nAtoms << ", #Diffractions: " << nDiff << ", #AtomTypes: " << nAtomTypes << std::endl;
  std::cout << "Blocks: " << nBx << " x " << nBy << " Threads: " << nTx << " x " << nTy << std::endl;
  std::cout << "MemUsage: " << (nDiff*(3+4)*sizeof(double)+nAtoms*4*sizeof(double)+nAtoms*1*sizeof(int)) << "B" << std::endl;

  long nDiffSizeDouble = nDiff*sizeof(double);

  int     *atomType;
  int     atomTypeValues[nAtoms];
  gpuErrchk(cudaMalloc(&atomType, nAtoms*sizeof(int)));
  double  *atomOccu;
  double  atomOccuValues[nAtoms];
  gpuErrchk(cudaMalloc(&atomOccu, nAtoms*sizeof(double)));
  double  *atomPos3D;
  double  atomPos3DValues[nAtoms][3];
  size_t  atomPos3Dpitch;
  gpuErrchk(cudaMallocPitch(&atomPos3D, &atomPos3Dpitch, 3 * sizeof(double), nAtoms));
  // TMP Not implemented yet!
  // double  *atomDW3D;
  // double  atomDW3DValues[nAtoms][3] = new double[nAtoms][3];
  // size_t  atomDW3Dpitch;
  // cudaMallocPitch(&atomDW3D, &atomDW3Dpitch, 3 * sizeof(double), nAtoms);
  double  *diffPos3D;
  double  diffPos3DValues[nDiff][3];
  size_t  diffPos3Dpitch;
  gpuErrchk(cudaMallocPitch(&diffPos3D, &diffPos3Dpitch, 3 * sizeof(double), nDiff));
  double  *atomicF;
  double  atomicFValues[nAtomTypes][9];
  size_t  atomicFpitch;
  gpuErrchk(cudaMallocPitch(&atomicF, &atomicFpitch, 9 * sizeof(double), nAtomTypes));
  double  *diffFstaticRe;
  double  diffFstaticReValues[nDiff];
  gpuErrchk(cudaMallocManaged(&diffFstaticRe, nDiffSizeDouble));
  double  *diffFstaticIm;
  double  diffFstaticImValues[nDiff];
  gpuErrchk(cudaMallocManaged(&diffFstaticIm, nDiffSizeDouble));
  double  *diffFunitRe;
  double  diffFunitReValues[nDiff];
  gpuErrchk(cudaMalloc(&diffFunitRe, nDiffSizeDouble));
  double  *diffFunitIm;
  double  diffFunitImValues[nDiff];
  gpuErrchk(cudaMalloc(&diffFunitIm, nDiffSizeDouble));

  // Generate some test data:
  std::cout << "Init data " << std::endl;
  int newvals1[] = {0,1,1,0,0,1,1,0};
  std::copy(newvals1, newvals1 + sizeof(newvals1) / sizeof(newvals1[0]), atomTypeValues);
  double newvals2[8][3] = {{0.0,0.5,0.0},{0.5,0.5,0.0},{0.0,0.0,0.0},{0.5,0.0,0.0},
                           {0.25,0.5,0.5},{0.75,0.5,0.5},{0.25,0.0,0.5},{0.75,0.0,0.5}};
  for (int index = 0; index < nAtoms; ++index)
  {
    atomPos3DValues[index][0] = newvals2[index][0];
    atomPos3DValues[index][1] = newvals2[index][1];
    atomPos3DValues[index][2] = newvals2[index][2];
  };
  double newvals3[] = {0.9, 1.0, 0.9, 1.0,
                       0.9, 1.0, 0.9, 1.0};
  std::copy(newvals3, newvals3+sizeof(newvals3)/sizeof(newvals3[0]), atomOccuValues);
  for(int index = 0; index < nDiff; ++index)
  {
    diffPos3DValues[index][0] = floor((double)index / 10.);
    diffPos3DValues[index][1] = ceil((double)index / 20.);
    diffPos3DValues[index][2] = (0.1 * (index % 10));

    diffFstaticReValues[index] = 0.0;
    diffFstaticImValues[index] = 0.0;
    diffFunitReValues[index] = 0.0;
    diffFunitImValues[index] = 0.0;
  };

  // copy
  std::cout << "Copy memory " << std::endl;
  gpuErrchk(cudaMemcpy(atomType, atomTypeValues, nAtoms*sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(atomOccu, atomOccuValues, nAtoms*sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy2D(atomicF, atomicFpitch, atomicFValues, 9*sizeof(double), 9*sizeof(double), nAtomTypes, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy2D(atomPos3D, atomPos3Dpitch, atomPos3DValues, 3*sizeof(double), 3*sizeof(double), nAtoms, cudaMemcpyHostToDevice));
  // cudaMemcpy2D(atomDW3D, atomDW3Dpitch, atomDW3DValues, 3*sizeof(double), 3*sizeof(double), nAtoms, cudaMemcpyHostToDevice);
  gpuErrchk(cudaMemcpy2D(diffPos3D, diffPos3Dpitch, diffPos3DValues, 3*sizeof(double), 3*sizeof(double), nDiff, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(diffFstaticRe, diffFstaticReValues, nDiffSizeDouble, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(diffFstaticIm, diffFstaticImValues, nDiffSizeDouble, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(diffFunitRe, diffFunitReValues, nDiffSizeDouble, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(diffFunitIm, diffFunitImValues, nDiffSizeDouble, cudaMemcpyHostToDevice));
  
  std::cout << "Start calcF " << std::endl;
  // Run 
  calcF<<<numBlocks, threadsPerBlock>>>(nAtoms, nDiff, 
    diffPos3D, diffPos3Dpitch,
    atomPos3D, atomPos3Dpitch,
    // atomDW3D, atomDW3Dpitch,
    atomOccu, atomType,
    atomicF, atomicFpitch,
    diffFunitRe, diffFunitIm);

  // Wait for GPU to finish before accessing on host
  gpuErrchk(cudaDeviceSynchronize());

  gpuErrchk(cudaMemcpy(diffFunitReValues, diffFunitRe, nDiffSizeDouble, cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(diffFunitImValues, diffFunitIm, nDiffSizeDouble, cudaMemcpyDeviceToHost));
  // Test
  float output = 0.0;
  std::cout << std::fixed;
  std::cout.precision(2);
  for(int index = 0; index < 100; ++index)
  {
    output = (pow(diffFunitReValues[index], 2) + pow(diffFunitImValues[index], 2));
    std::cout << output << " ";
    if ((index+1) % 10 == 0) 
    {
      std::cout << std::endl;
    }
  }
  std::cout << std::endl;
  std::cout << "Fin " << std::endl;

  // Free memory
  cudaFree(atomType);
  cudaFree(atomPos3D);
  cudaFree(atomOccu);
  cudaFree(diffPos3D);
  cudaFree(atomicF);
  cudaFree(diffFstaticRe);
  cudaFree(diffFstaticIm);
  cudaFree(diffFunitRe);
  cudaFree(diffFunitIm);
  
  return 0;
}