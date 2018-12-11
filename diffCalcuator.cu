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
  double *baseFRe, size_t baseFRepitch,
  double *baseFIm, size_t baseFImpitch,
  double *outFRe, double *outFIm)
{
  uint diffIndex = threadIdx.x + (blockDim.x * blockIdx.x);
  uint atomIndex = threadIdx.y + (blockDim.y * blockIdx.y);
  // uint atomIndex = threadIdx.x + (blockDim.x * blockIdx.x);
  // uint diffIndex = threadIdx.y + (blockDim.y * blockIdx.y);

  
  if (atomIndex < nAtoms && diffIndex < nDiff)
  {
    double dotProduct = 0.0;
    double* thisAtomPos3D = (double*)((char*)atomPos3D + atomIndex * atomPos3Dpitch);
    double* thisDiffPos3D = (double*)((char*)diffPos3D + diffIndex * atomPos3Dpitch);
    for (int direction = 0; direction < 3; ++direction)
    {
      dotProduct += thisAtomPos3D[direction] * thisDiffPos3D[direction];
    }
    // not sure if & is needed
    // might be possible to speed up by summing up block wise and than sum blocks
    double* thisBaseFRe = (double*)((char*)baseFRe + atomType[atomIndex] * baseFRepitch);
    atomicAdd(&outFRe[diffIndex], (thisBaseFRe[diffIndex] * atomOccu[atomIndex] * cos(dotProduct)) );
    double* thisBaseFIm = (double*)((char*)baseFIm + atomType[atomIndex] * baseFImpitch);
    atomicAdd(&outFIm[diffIndex], (thisBaseFIm[diffIndex] * atomOccu[atomIndex] * sin(dotProduct)) );
  }
}



int main(int argc, char** argv) 
{
  long nAtoms = atol(argv[1]); // along thread.y
  long nDiff = atol(argv[2]); // along thread.x
  int nAtomTypes = 3; // Fe, O, Ir

  std::cout << 
    "#Atoms: " << nAtoms << 
    ", #Diffractions: " << nDiff <<
    ", #AtomTypes: " << nAtomTypes <<
    std::endl;
  std::cout << "MemUsage: " << nDiff*(nAtomTypes+4)*sizeof(double)*2*1e-6 << "MB" << std::endl;

  // int nTx = 32;
  int nTx = 64;
  int nTy = 8;
  if (nDiff < 32)
  {
    nTx = nDiff;
  }
  if (nAtoms < 8)
  { 
    nTy = nAtoms;
  }
  dim3 threadsPerBlock(nTx, nTy);
  int nBx = ceil(nDiff / threadsPerBlock.x);
  int nBy = ceil(nAtoms / threadsPerBlock.y);
  dim3 numBlocks(nBx, nBy);
  std::cout << "Blocks: " << nBx << " x " << nBy << " Threads: " << nTx << " x " << nTy << std::endl;

  int     *atomType = new int[nAtoms];
  cudaMallocManaged(&atomType, nAtoms*sizeof(int));
  double  *atomPos3D;
  double  atomPos3DValues[nAtoms][3];
  size_t  atomPos3Dpitch;
  cudaMallocPitch(&atomPos3D, &atomPos3Dpitch, 3 * sizeof(double), nAtoms);
  double  *atomOccu = new double[nAtoms];
  cudaMallocManaged(&atomOccu, nAtoms*sizeof(double));
  // TMP Not implemented yet!
  // double  *atomDW3D;
  // double  atomDW3DValues[nAtoms][3] = new double[nAtoms][3];
  // size_t  atomDW3Dpitch;
  // cudaMallocPitch(&atomDW3D, &atomDW3Dpitch, 3 * sizeof(double), nAtoms);
  double  *diffPos3D;
  double  diffPos3DValues[nDiff][3];
  size_t  diffPos3Dpitch;
  cudaMallocPitch(&diffPos3D, &diffPos3Dpitch, 3 * sizeof(double), nDiff);
  double  *atomicF;
  double  atomicFValues[nDiff][9];
  size_t  atomicFpitch;
  cudaMallocPitch(&atomicF, &atomicFpitch, 9 * sizeof(double), nDiff);
  // This might take up a lot of RAM:
  double  *diffFatomicRe;
  double  diffFatomicReValues[nAtomTypes][nDiff];
  size_t  diffFatomicRepitch;
  cudaMallocPitch(&diffFatomicRe, &diffFatomicRepitch, nDiff * sizeof(double), nAtomTypes);
  double  *diffFatomicIm;
  double  diffFatomicImValues[nAtomTypes][nDiff];
  size_t  diffFatomicImpitch;
  cudaMallocPitch(&diffFatomicIm, &diffFatomicImpitch, nDiff * sizeof(double), nAtomTypes);
  // static is mainly bulk:
  double  *diffFstaticRe = new double[nDiff];
  cudaMallocManaged(&diffFstaticRe, nDiff*sizeof(double));
  double  *diffFstaticIm = new double[nDiff];
  cudaMallocManaged(&diffFstaticIm, nDiff*sizeof(double));
  double  *diffFunitRe = new double[nDiff];
  cudaMallocManaged(&diffFunitRe, nDiff*sizeof(double));
  double  *diffFunitIm = new double[nDiff];
  cudaMallocManaged(&diffFunitIm, nDiff*sizeof(double));

  // Generate some test data:
  int newvals1[] = {0,1,1,0,0,1,1,0};
  std::copy(newvals1, newvals1 + sizeof(newvals1) / sizeof(newvals1[0]), atomType);
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
  std::copy(newvals3, newvals3+sizeof(newvals3)/sizeof(newvals3[0]), atomOccu);
  for(int index = 0; index < nDiff; ++index)
  {
    diffPos3DValues[index][0] = floor((double)index / 10.);
    diffPos3DValues[index][1] = ceil((double)index / 20.);
    diffPos3DValues[index][2] = (0.1 * (index % 10));
    // std::copy(newvals4, newvals4+sizeof(newvals4)/sizeof(newvals4[0]), diffPos3D[index]);
    diffFstaticRe[index] = 0;
    diffFstaticIm[index] = 0;
    diffFunitRe[index] = 0;
    diffFunitIm[index] = 0;
  };
  for(int aIndex = 0; aIndex < nAtoms; ++aIndex)
  {
    for(int qIndex = 0; qIndex < nDiff; ++qIndex)
    {
      diffFatomicReValues[aIndex][qIndex] = 1.0;
      diffFatomicImValues[aIndex][qIndex] = 1.0;
    };
  };

  // copy
  gpuErrchk(cudaMemcpy2D(atomPos3D, atomPos3Dpitch, atomPos3DValues, 3*sizeof(double), 3*sizeof(double), nAtoms, cudaMemcpyHostToDevice));
  // cudaMemcpy2D(atomDW3D, atomDW3Dpitch, atomDW3DValues, 3*sizeof(double), 3*sizeof(double), nAtoms, cudaMemcpyHostToDevice);
  cudaMemcpy2D(diffPos3D, diffPos3Dpitch, diffPos3DValues, 3*sizeof(double), 3*sizeof(double), nDiff, cudaMemcpyHostToDevice);
  cudaMemcpy2D(atomicF, atomicFpitch, atomicFValues, 9*sizeof(double), 9*sizeof(double), nDiff, cudaMemcpyHostToDevice);
  cudaMemcpy2D(diffFatomicRe, diffFatomicRepitch, diffFatomicReValues, nDiff*sizeof(double), nDiff*sizeof(double), nAtomTypes, cudaMemcpyHostToDevice);
  cudaMemcpy2D(diffFatomicIm, diffFatomicImpitch, diffFatomicImValues, nDiff*sizeof(double), nDiff*sizeof(double), nAtomTypes, cudaMemcpyHostToDevice);
  
  std::cout << "Start calcF " << std::endl;
  // Run 
  calcF<<<numBlocks, threadsPerBlock>>>(nAtoms, nDiff, 
    diffPos3D, diffPos3Dpitch,
    atomPos3D, atomPos3Dpitch,
    // atomDW3D, atomDW3Dpitch,
    atomOccu, atomType,
    diffFatomicRe, diffFatomicRepitch,
    diffFatomicIm, diffFatomicImpitch,
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
  cudaFree(diffFatomicRe);
  cudaFree(diffFatomicIm);
  cudaFree(diffFstaticRe);
  cudaFree(diffFstaticIm);
  cudaFree(diffFunitRe);
  cudaFree(diffFunitIm);
  
  return 0;
}