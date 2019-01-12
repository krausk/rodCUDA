#include<stdio.h>
#include<iostream>
#include<math.h>
#include<vector>

// includes CUDA
#include<cuda.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

// Functions
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
  double *atomicF,
  double *outFRe, double *outFIm)
{
  long totalThreadsX = (blockDim.x * gridDim.x);
  long totalThreadsY = (blockDim.y * gridDim.y);
  int calcManyX = ceil((double)nDiff / totalThreadsX);
  int calcManyY = ceil((double)nAtoms / totalThreadsY);

  double pipi16 = (16. * M_PI * M_PI);

  for (int threadRunY = 0; threadRunY < calcManyY; ++threadRunY)
  {
    long atomIndex = threadIdx.y + (blockDim.y * blockIdx.y) + threadRunY * totalThreadsY;

    for (int threadRunX = 0; threadRunX < calcManyX; ++threadRunX)
    {
      long diffIndex = threadIdx.x + (blockDim.x * blockIdx.x) + threadRunX * totalThreadsX;
      
      if (atomIndex < nAtoms && diffIndex < nDiff)
      {
        // atomic form factor
        double diffDot = 0.0;
        diffDot += diffPos3D0[diffIndex] * diffPos3D0[diffIndex];
        diffDot += diffPos3D1[diffIndex] * diffPos3D1[diffIndex];
        diffDot += diffPos3D2[diffIndex] * diffPos3D2[diffIndex];
        double atomicFExpFact = 0.0;
        atomicFExpFact = diffDot / pipi16; // Lambda needs to be included here if diffPos3D is not in Q space!
        double atomicFRe = 1.0;
        for (uint index = 0; index < 4; ++index)
        {
          atomicFRe += (atomicF[atomType[atomIndex] * 9 + index] *
               exp(-atomicF[atomType[atomIndex] * 9 + 4 + index] * atomicFExpFact));
        }
        atomicFRe += atomicF[atomType[atomIndex] * 9 + 9];

        // debye-waller factors 
        double debyWaller = 1.0;
        if (atomDW3D0[atomIndex] < 0) { // if DW0 < 0 => inplane and out-of-plane
          if (atomDW3D1[atomIndex] < 0) { // if DW1 < 0 and => iso
            if (atomDW3D2[atomIndex] < 0) { // if DW2 < 0 no DW 
              debyWaller = 1.0;
            }
            else { //  iso debye-waller
              debyWaller *= exp(-atomDW3D0[atomIndex] * diffDot / pipi16);
            }
          }
          else { //  inplane and out-of-plane debye-waller
            debyWaller *= exp(-atomDW3D1[atomIndex] *
                              (diffPos3D0[diffIndex] * diffPos3D0[diffIndex] +
                               diffPos3D1[diffIndex] * diffPos3D1[diffIndex]) /
                              pipi16);
            debyWaller *= exp(-atomDW3D2[atomIndex] * diffPos3D2[diffIndex] *
                              diffPos3D2[diffIndex] / pipi16);
          }
        } 
        else { //  all direction debye-waller
          debyWaller *= exp(-atomDW3D0[atomIndex] * diffPos3D0[diffIndex] *
                            diffPos3D0[diffIndex] / pipi16);
          debyWaller *= exp(-atomDW3D1[atomIndex] * diffPos3D1[diffIndex] *
                            diffPos3D1[diffIndex] / pipi16);
          debyWaller *= exp(-atomDW3D2[atomIndex] * diffPos3D2[diffIndex] *
                            diffPos3D2[diffIndex] / pipi16);
        }

        // might be possible to speed up by summing up block wise and than sum blocks
        double dotProduct = 0.0;
        dotProduct += atomPos3D0[atomIndex] * diffPos3D0[diffIndex];
        dotProduct += atomPos3D1[atomIndex] * diffPos3D1[diffIndex];
        dotProduct += atomPos3D2[atomIndex] * diffPos3D2[diffIndex];
        // double* thisBaseFIm = (double*)((char*)baseFIm + diffIndex * baseFImpitch);
        // atomicAdd(&outFRe[diffIndex], (thisBaseFRe[atomType[atomIndex]] * atomOccu[atomIndex] * cos(dotProduct)) );
        double cosDot = cos(dotProduct);
        double sinDot = sin(dotProduct);
        double resultFRe = atomOccu[atomIndex] * (atomicFRe * cosDot);
        double resultFIm = atomOccu[atomIndex] * (atomicFRe * sinDot);
        atomicAdd(&outFRe[diffIndex], resultFRe );
        atomicAdd(&outFIm[diffIndex], resultFIm );
      }
    }
  }
}


class DiffRun
{
  public:
    long nAtoms; // along thread.y
    long nDiff; // along thread.x
    int nAtomTypes; // number of atomicF

    double  *atomicF;
    int     *atomType;
    double  *atomOccu;
    double  *atomPos3D0;
    double  *atomPos3D1;
    double  *atomPos3D2;
    double  *atomDW3D0;
    double  *atomDW3D1;
    double  *atomDW3D2;
    double  *diffPos3D0;
    double  *diffPos3D1;
    double  *diffPos3D2;
    double  *diffFstaticRe;
    double  *diffFstaticIm;
    double  *diffFunitRe;
    double  *diffFunitIm;

    bool finishedRun = false;
    void run(void);

    // determine compute structure
    int nTx = 64;
    int nTy = 8;
    int maxBlocks = 500000;
};
void DiffRun::run(void){
  if (nDiff < nTx){
    nTx = nDiff;
  }
  if (nAtoms < nTy) { 
    nTy = nAtoms;
  }
  int nBx = floor(nDiff / nTx);
  int nBy = floor(nAtoms / nTy);
  if (nBx > maxBlocks) {
    nBx = maxBlocks;
  }
  std::cout << "#Atoms: " << nAtoms << ", #Diffractions: " << nDiff << ", #AtomTypes: " << nAtomTypes << std::endl;
  std::cout << "Blocks: " << nBx << " x " << nBy << " Threads: " << nTx << " x " << nTy << std::endl;
  float memGuess = (nDiff*(3+4)*sizeof(double)+nAtoms*4*sizeof(double)+nAtoms*1*sizeof(int));
  std::cout << "MemUsage: " << memGuess*1e-6 << " MB" << std::endl;

  dim3 threadsPerBlock(nTx, nTy);
  dim3 numBlocks(nBx, nBy);

  gpuErrchk(cudaMallocManaged(&atomicF, nAtomTypes*9*sizeof(double)));
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

  calcF<<<numBlocks, threadsPerBlock>>>(
    nAtoms, nDiff, 
    diffPos3D0, diffPos3D1, diffPos3D2,
    atomOccu, atomType,
    atomPos3D0, atomPos3D1, atomPos3D2,
    atomDW3D0, atomDW3D1, atomDW3D2,
    atomicF,
    diffFunitRe, diffFunitIm);

  // Wait for GPU to finish before accessing on host
  gpuErrchk(cudaDeviceSynchronize());
  finishedRun = true;
};


class DiffRunController 
{
  public:
    std::vector<DiffRun*> diffRunsAll;
    DiffRun *diffRunCurrent;

    void addRun(DiffRun *diffRunAdd);
    DiffRunController(DiffRun* diffRunInit);
    void runAll(void);
};
DiffRunController::DiffRunController(DiffRun* diffRunInit) {
  addRun(diffRunInit);
}
void DiffRunController::addRun(DiffRun* diffRunAdd) {
  diffRunAdd = diffRunAdd;
  diffRunsAll.push_back(diffRunAdd);
}
void DiffRunController::runAll(void) {
  for (std::vector<DiffRun*>::iterator diffRunThis = diffRunsAll.begin();
      diffRunThis != diffRunsAll.end(); ++diffRunThis) {
    diffRunCurrent = *diffRunThis;
    diffRunCurrent->run();

    // Test
    float output = 0.0;
    std::cout << std::fixed;
    std::cout.precision(2);
    for(int index = 0; index < 100; ++index)
    {
      output = (pow(diffRunCurrent->diffFunitRe[index], 2) + pow(diffRunCurrent->diffFunitIm[index], 2));
      std::cout << output << " ";
      if ((index+1) % 10 == 0) {
        std::cout << std::endl;
      }
    }
    std::cout << std::endl;
  }
}


int main(int argc, char** argv) 
{
  long nAtoms = atol(argv[1]); // along thread.y
  long nDiff = atol(argv[2]); // along thread.x
  int nAtomTypes = 3; // Fe, O, Ir

  // create GPU pointer
  double  *atomicF = new double[nAtomTypes * 9];
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

  // Generate some test data:
  std::cout << "Init data " << std::endl;
  int newAtomType[] = {0, 1, 1, 0, 0, 1, 1, 0};
  double newAtomOccu[] = {0.9, 1.0, 0.9, 1.0,
                          0.9, 1.0, 0.9, 1.0};
  double newAtomPos3D[8][3] = {{0.0,0.5,0.0}, {0.5,0.5,0.0},
                               {0.0,0.0,0.0}, {0.5,0.0,0.0},
                               {0.25,0.5,0.5},{0.75,0.5,0.5},
                               {0.25,0.0,0.5},{0.75,0.0,0.5}};
  double newAtomDW3D[8][3] = {{-1.0,0.0,0.0}, {-1.0,-20.0,-40.0},
                              {-10.0,-20.0,0.0}, {0.0,0.0,-0.0},
                              {0.0,0.0,0.0}, {0.0,0.0,0.0},
                              {0.0,0.0,0.0}, {0.0,0.0,0.0}};
  for (int index = 0; index < nAtoms; ++index)
  {
    atomType[index] = newAtomType[index];
    atomOccu[index] = newAtomOccu[index];
    atomPos3D0[index] = newAtomPos3D[index][0];
    atomPos3D1[index] = newAtomPos3D[index][1];
    atomPos3D2[index] = newAtomPos3D[index][2];
    atomDW3D0[index] = newAtomDW3D[index][0];
    atomDW3D1[index] = newAtomDW3D[index][1];
    atomDW3D2[index] = newAtomDW3D[index][2];
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

  // Run 
  std::cout << "Create Runs " << std::endl;
  
  DiffRun* diffRunFirst =  new DiffRun();
  diffRunFirst->nAtoms = nAtoms;
  diffRunFirst->nDiff = nDiff;
  diffRunFirst->nAtomTypes = nAtomTypes;
  diffRunFirst->atomicF = atomicF;
  diffRunFirst->atomType = atomType;
  diffRunFirst->atomOccu = atomOccu;
  diffRunFirst->atomPos3D0 = atomPos3D0;
  diffRunFirst->atomPos3D1 = atomPos3D1;
  diffRunFirst->atomPos3D2 = atomPos3D2;
  diffRunFirst->atomDW3D0 = atomDW3D0;
  diffRunFirst->atomDW3D1 = atomDW3D1;
  diffRunFirst->atomDW3D2 = atomDW3D2;
  diffRunFirst->diffPos3D0 = diffPos3D0;
  diffRunFirst->diffPos3D1 = diffPos3D1;
  diffRunFirst->diffPos3D2 = diffPos3D2;
  diffRunFirst->diffFstaticRe = diffFstaticRe;
  diffRunFirst->diffFstaticIm = diffFstaticIm;
  diffRunFirst->diffFunitRe = diffFunitRe;
  diffRunFirst->diffFunitIm = diffFunitIm;


  DiffRunController* diffRunController =  new DiffRunController(diffRunFirst);
  diffRunController->addRun(diffRunFirst);

  std::cout << "Start calcF " << std::endl;
  diffRunController->runAll();


  // // Test
  // DiffRun* diffRunCurrent = diffRunController->diffRunCurrent;
  // float output = 0.0;
  // std::cout << std::fixed;
  // std::cout.precision(2);
  // for(int index = 0; index < 100; ++index)
  // {
  //   output = (pow(diffRunCurrent->diffFunitRe[index], 2) + pow(diffRunCurrent->diffFunitIm[index], 2));
  //   std::cout << output << " ";
  //   if ((index+1) % 10 == 0) {
  //     std::cout << std::endl;
  //   }
  // }
  // std::cout << std::endl;
  
  std::cout << "Fin " << std::endl;
  return 0;
}
