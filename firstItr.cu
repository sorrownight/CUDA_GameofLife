#include <chrono>
#include <ctime>
#include <functional>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>

const unsigned int THREAD_1D = 21;
const unsigned int BLOCK_AREA = THREAD_1D * THREAD_1D;
const unsigned int BLOCK_1D = 2;
const unsigned int DIM = (BLOCK_AREA * BLOCK_1D);
// Reserving 4 rows/cols of 0s on the border to eliminate edge cases

const unsigned int GEN_COUNT = 5000;
const std::string DEFAULT_PATTERN_FILE = "data.txt";

__device__ static unsigned int getRow()
{
    return threadIdx.y + blockDim.y * DIM;
}

__device__ static unsigned int getCol()
{
    return threadIdx.x + blockDim.x * DIM;
}

// Each generation takes up a space of DIM * DIM
// Each row takes up a space of DIM -> offset by col
__device__ static unsigned int getActualIdx(unsigned int gen, unsigned int row, unsigned int col)
{
    unsigned int genStart = gen * DIM * DIM; // Each subarray has DIM * DIM length
    return genStart + row * DIM + col;
}

__global__ void nextGen(bool* gens, int curGen)
{
    const unsigned int row = getRow();
    const unsigned int col = getCol();

    // We want some threads working on the reserved rows/cols of zero's too! (For performance)
    if (row == 0 || col == 0 || row == DIM - 1 || col == DIM - 1)
        gens[getActualIdx(curGen,row,col)] = false;
    else {
        const unsigned int neighborCount = gens[getActualIdx(curGen-1, row - 1, col)] // South
                                         + gens[getActualIdx(curGen-1, row + 1, col)] // North
                                         + gens[getActualIdx(curGen-1, row, col - 1)] // West
                                         + gens[getActualIdx(curGen-1, row, col + 1)] // East
                                         + gens[getActualIdx(curGen-1, row - 1, col - 1)] // South West
                                         + gens[getActualIdx(curGen-1, row + 1, col - 1)] // North West
                                         + gens[getActualIdx(curGen-1, row - 1, col + 1)] // South East
                                         + gens[getActualIdx(curGen-1, row + 1, col + 1)] // North East
                                         ;

        gens[getActualIdx(curGen,row,col)] = neighborCount == 3
                                            || (neighborCount == 2 && gens[getActualIdx(curGen-1,row,col)]);
    }
}

bool* firstIterLife(const bool* initialGrid)
{
    dim3 dimGrid(BLOCK_1D, BLOCK_1D, 1);
    dim3 dimBlock(THREAD_1D, THREAD_1D, 1);

    bool* d_gens;
    cudaMalloc((void**) &d_gens, GEN_COUNT * DIM * DIM);

    // First generation needs to be copied over to start the algorithm
    cudaMemcpy(d_gens, initialGrid, DIM * DIM, cudaMemcpyHostToDevice);

    for (int gen = 1; gen < GEN_COUNT; gen++) {
        nextGen<<<dimGrid, dimBlock>>>(d_gens, gen);
    }
    bool *result = new bool[GEN_COUNT*DIM*DIM];
    cudaMemcpy(result, d_gens, GEN_COUNT * DIM * DIM, cudaMemcpyDeviceToHost);

    return result;
}

unsigned int getGridIdx(unsigned int row, unsigned int col)
{
    return row * DIM + col;
}

unsigned int getActualIdxHost(unsigned int gen, unsigned int row, unsigned int col)
{
    unsigned int genStart = gen * DIM * DIM; // Each subarray has DIM * DIM length
    return genStart + getGridIdx(row, col);
}

bool* serialLife(const bool* initialGrid)
{
    bool* gens = new bool[GEN_COUNT*DIM*DIM];

    // First generation needs to be copied over to start the algorithm
    memcpy(gens, initialGrid, DIM * DIM);

    for (int curGen = 1; curGen < GEN_COUNT; curGen++) {
        for (int row = 0; row < DIM; row++) {
            for (int col = 0; col < DIM; col ++) {
                if (row == 0 || col == 0 || row == DIM - 1 || col == DIM - 1)
                    gens[getActualIdxHost(curGen,row,col)] = false;
                else {
                    const unsigned int neighborCount =
                            gens[getActualIdxHost(curGen-1, row - 1, col)] // South
                          + gens[getActualIdxHost(curGen-1, row + 1, col)] // North
                          + gens[getActualIdxHost(curGen-1, row, col - 1)] // West
                          + gens[getActualIdxHost(curGen-1, row, col + 1)] // East
                          + gens[getActualIdxHost(curGen-1, row - 1, col - 1)] // South West
                          + gens[getActualIdxHost(curGen-1, row + 1, col - 1)] // North West
                          + gens[getActualIdxHost(curGen-1, row - 1, col + 1)] // South East
                          + gens[getActualIdxHost(curGen-1, row + 1, col + 1)] // North East
                    ;

                    gens[getActualIdxHost(curGen,row,col)] = neighborCount == 3
                            || (neighborCount == 2 && gens[getActualIdxHost(curGen-1,row,col)]);
                }
            }
        }
    }

    return gens;
}

bool* measureTime(const std::function<bool*(const bool*)>& func,
                 const bool* initialGrid)
{
    using namespace std::chrono;

    auto start = system_clock::now();
    bool* result = func(initialGrid);
    auto end = system_clock::now();

    std::cout << "Execution Time: " << (duration_cast<milliseconds>(end-start)).count()
                << " milliseconds" << std::endl;

    return result;
}

bool* createGridFromFile()
{
    std::vector<std::vector<bool>> dataGrid;

    std::ifstream file(DEFAULT_PATTERN_FILE);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            std::vector<bool> curRow;
            dataGrid.push_back(curRow);

            for (char c : line) {
                if (c == '.') curRow.push_back(false);
                else curRow.push_back(true);
            }
        }

    } else {
        std::cout << "File cannot be opened!" << std::endl;
        exit(-1);
    }

    bool* initialGrid = new bool[DIM*DIM];
    for (unsigned int row = 1; row < DIM; row++) {
        for (unsigned int col = 1; col < DIM; col++) {
            initialGrid[getGridIdx(row, col)] = dataGrid[(row-1) % dataGrid.size()][(col-1) % dataGrid[0].size()];
        }
    }

    return initialGrid;
}

int main()
{

    bool* initialGrid = createGridFromFile();
    for (int row = 0; row < DIM; row++) initialGrid[getGridIdx(row, 0)] = false; // West edge
    for (int row = 0; row < DIM; row++) initialGrid[getGridIdx(row, DIM-1)] = false; // East edge
    for (int col = 0; col < DIM; col++) initialGrid[getGridIdx(0, col)] = false; // North edge
    for (int col = 0; col < DIM; col++) initialGrid[getGridIdx(DIM-1, col)] = false; // South edge

    std::cout << "Running " << GEN_COUNT << " generations on grid of " << DIM << "x" << DIM << std:: endl;

    std::cout << "Serial version: " << std:: endl;
    bool* serialResult = measureTime(serialLife, initialGrid);


    std::cout << "CUDA version: " << std:: endl;
    bool* cudaResult = measureTime(firstIterLife, initialGrid);

    bool matched = true;
    for (int row = 0; row < DIM && matched; row++) {
        for (int col = 0; col < DIM && matched; col++) {
            if (  serialResult[getActualIdxHost(GEN_COUNT-1,row,col)]
               != cudaResult[getActualIdxHost(GEN_COUNT-1,row,col)]) {
                matched = false;
            }
        }
    }

    if (!matched) std::cout << "Not matched!" << std:: endl;
    else std::cout << "Last generation of serial vs CUDA matched!" << std:: endl;

    delete[] initialGrid;
    delete[] serialResult;
    delete[] cudaResult;
}
