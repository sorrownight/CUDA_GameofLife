#include <chrono>
#include <ctime>
#include <functional>
#include <string>
#include <iostream>

const unsigned int THREAD_1D = 21;
const unsigned int BLOCK_AREA = THREAD_1D * THREAD_1D;
const unsigned int BLOCK_1D = 2;
const unsigned int DIM = (BLOCK_AREA * BLOCK_1D);
// Reserving 4 rows/cols of 0s on the border to eliminate edge cases

const unsigned int GEN_COUNT = 100;

__device__ static unsigned int getRow()
{
    return threadIdx.y + blockDim.y * DIM;
}

__device__ static unsigned int getCol()
{
    return threadIdx.x + blockDim.x * DIM;
}

__global__ void nextGen(bool (*gens)[DIM][DIM], int curGen)
{
    const unsigned int row = getRow();
    const unsigned int col = getCol();

    // We want some threads working on the reserved rows/cols of zero's too! (For performance)
    if (row == 0 || col == 0 || row == DIM - 1 || col == DIM - 1) gens[curGen][row][col] = false;
    else {
        const unsigned int neighborCount = gens[curGen-1][row - 1][col    ] // South
                                         + gens[curGen-1][row + 1][col    ] // North
                                         + gens[curGen-1][row    ][col - 1] // West
                                         + gens[curGen-1][row    ][col + 1] // East
                                         + gens[curGen-1][row - 1][col - 1] // South West
                                         + gens[curGen-1][row + 1][col - 1] // North West
                                         + gens[curGen-1][row - 1][col + 1] // South East
                                         + gens[curGen-1][row + 1][col + 1] // North East
                                         ;

        gens[curGen][row][col] = neighborCount == 3 || (neighborCount == 2 && gens[curGen-1][row][col]);
    }
}

bool*** firstIterLife(const bool initialGrid[DIM][DIM])
{
    dim3 dimGrid(BLOCK_1D, BLOCK_1D, 1);
    dim3 dimBlock(THREAD_1D, THREAD_1D, 1);

    bool (*d_gens)[DIM][DIM];
    cudaMalloc((void**) &d_gens, GEN_COUNT * DIM * DIM);
    cudaMemcpy(d_gens[0], initialGrid, DIM * DIM, cudaMemcpyHostToDevice);

    for (int gen = 1; gen < GEN_COUNT; gen++) {
        nextGen<<<dimGrid, dimBlock>>>(d_gens, gen);
    }
    bool (*result)[DIM][DIM] = new bool[GEN_COUNT][DIM][DIM];
    cudaMemcpy(result, d_gens, GEN_COUNT * DIM * DIM, cudaMemcpyDeviceToHost);

    return reinterpret_cast<bool ***>(result);
}

bool*** serialLife(const bool initialGrid[DIM][DIM])
{
    bool (*result)[DIM][DIM] = new bool[GEN_COUNT][DIM][DIM];
    memcpy(result[0], initialGrid, DIM * DIM);

    for (int curGen = 1; curGen < GEN_COUNT; curGen++) {
        for (int row = 0; row < DIM; row++) {
            for (int col = 0; col < DIM; col ++) {
                if (row == 0 || col == 0 || row == DIM - 1 || col == DIM - 1) result[curGen][row][col] = false;
                else {
                    const unsigned int neighborCount =     result[curGen - 1][row - 1][col    ] // South
                                                         + result[curGen-1][row + 1][col    ] // North
                                                         + result[curGen-1][row    ][col - 1] // West
                                                         + result[curGen-1][row    ][col + 1] // East
                                                         + result[curGen-1][row - 1][col - 1] // South West
                                                         + result[curGen-1][row + 1][col - 1] // North West
                                                         + result[curGen-1][row - 1][col + 1] // South East
                                                         + result[curGen-1][row + 1][col + 1] // North East
                    ;
                    result[curGen][row][col] = neighborCount == 3 || (neighborCount == 2 && result[curGen-1][row][col]);
                }
            }
        }
    }

    return reinterpret_cast<bool ***>(result);
}

bool*** measureTime(const std::function<bool***(const bool[DIM][DIM])>& func,
                 const bool initialGrid[DIM][DIM])
{
    using namespace std::chrono;

    auto start = system_clock::now();
    bool*** result = func(initialGrid);
    auto end = system_clock::now();

    std::cout << "Execution Time: " << (duration_cast<milliseconds>(end-start)).count()
                << " milliseconds" << std::endl;

    return result;
}

int main()
{

    bool (*initialGrid)[DIM] = new bool[DIM][DIM];
    for (int row = 0; row < DIM; row++) initialGrid[row][0] = false; // West edge
    for (int row = 0; row < DIM; row++) initialGrid[row][DIM-1] = false; // East edge
    for (int col = 0; col < DIM; col++) initialGrid[0][col] = false; // North edge
    for (int col = 0; col < DIM; col++) initialGrid[DIM-1][col] = false; // South edge

    std::cout << "Running " << GEN_COUNT << " generations on grid of " << DIM << "x" << DIM << std:: endl;

    std::cout << "Serial version: " << std:: endl;
    bool*** serialResult = measureTime(serialLife, initialGrid);


    std::cout << "CUDA version: " << std:: endl;
    bool*** cudaResult = measureTime(firstIterLife, initialGrid);

    bool matched = true;
    for (int row = 0; row < DIM && matched; row++) {
        for (int col = 0; col < DIM && matched; col++) {
            if (serialResult[GEN_COUNT-1][row][col] != cudaResult[GEN_COUNT-1][row][col]) {
                matched = false;
            }
        }
    }

    if (!matched) std::cout << "Not matched!" << std:: endl;
    else std::cout << "Last generations of serial vs CUDA matched!" << std:: endl;

    delete[] initialGrid;
    delete[] serialResult;
    delete[] cudaResult;
}
