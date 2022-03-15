#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>
#include <functional>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <GL/glut.h>
#include <thread>

/**
 * @author Luan (Remi) Ta
 * @version 1.0 (03/13/22)
 *
 * Conway's Game of Life simulated with OpenGL (FreeGlut/NVIDIA's CG)
 * The simulation attempts to compare the performance of a naive CUDA solution vs that of a serial one
 */

const unsigned int THREAD_1D = 16;
const unsigned int BLOCK_1D = 80;
const unsigned int DIM = (THREAD_1D * BLOCK_1D);
// Reserving 4 rows/cols of 0s on the border to eliminate edge cases

const unsigned int GEN_COUNT = 100;
const std::string DEFAULT_PATTERN_FILE = "pattern.rle";


/**
 * @param gen The generation offset (by DIMxDIM per generation)
 * @param row The row offset (By DIM per row)
 * @param col The column offset (By 1 per col)
 * @return the actual index of the linearized 3-D grid of (generation, row, col)
 */
__forceinline __device__ static unsigned int getActualIdx(unsigned int gen, unsigned int row, unsigned int col)
{
    // Each generation takes up a space of DIM * DIM
    // Each row takes up a space of DIM -> offset by col
    unsigned int genStart = gen * DIM * DIM; // Each subarray has DIM * DIM length
    return genStart + row * DIM + col;
}

/**
 * [Compute kernel]
 * Determines whether the cell owned by this thread should be alive/dead for this generation
 * @param gens linearized 3-D Data grid of (generation, row, col)
 * @param curGen The current generation
 */
__global__ void nextGen(bool* gens, int curGen)
{
    const unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
    const unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;

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

        // Referenced from:
        // http://www.marekfiser.com/Projects/Conways-Game-of-Life-on-GPU-using-CUDA/2-Basic-implementation
        gens[getActualIdx(curGen,row,col)] = neighborCount == 3
                                            || (neighborCount == 2 && gens[getActualIdx(curGen-1,row,col)]);
    }
}

/**
 * Launches the CUDA compute kernels to generate the grids for GEN_COUNT generations
 * @param initialGrid initial configuration
 * @return linearized 3-D Data grid of (generation, row, col) - length: GEN_COUNT
 */
bool* cudaLife(const bool* initialGrid)
{
    dim3 dimGrid(BLOCK_1D, BLOCK_1D);
    dim3 dimBlock(THREAD_1D, THREAD_1D);

    bool* d_gens;
    if (cudaMalloc((void**) &d_gens, (GEN_COUNT * DIM * DIM) * sizeof(bool)) != cudaSuccess)
        std::cout << "cudaMalloc failed" << std::endl;

    // First generation needs to be copied over to start the algorithm
    if (cudaMemcpy(d_gens, initialGrid, DIM * DIM * sizeof(bool), cudaMemcpyHostToDevice) != cudaSuccess)
        std::cout << "cudaMemcpy failed: Host->Device" << std::endl;

    for (int gen = 1; gen < GEN_COUNT; gen++) {
        nextGen<<<dimGrid, dimBlock>>>(d_gens, gen);
    }
    bool *result = new bool[GEN_COUNT*DIM*DIM];
    cudaError_t code;
    if ((code = cudaMemcpy(result, d_gens, (GEN_COUNT * DIM * DIM) * sizeof(bool), cudaMemcpyDeviceToHost)) != cudaSuccess)
        std::cout << "cudaMemcpy failed: Device->Host: " << cudaGetErrorString(code) << std::endl;
    
    cudaFree(d_gens);

    return result;
}

/**
 * @param row The row offset (By DIM per row)
 * @param col The column offset (By 1 per col)
 * @return the grid index not offset by generation of this row & column
 */
unsigned int getGridIdx(unsigned int row, unsigned int col)
{
    return row * DIM + col;
}

/**
 * @param gen The generation offset (by DIMxDIM per generation)
 * @param row The row offset (By DIM per row)
 * @param col The column offset (By 1 per col)
 * @return the actual index of the linearized 3-D grid of (generation, row, col)
 */
unsigned int getActualIdxHost(unsigned int gen, unsigned int row, unsigned int col)
{
    unsigned int genStart = gen * DIM * DIM; // Each subarray has DIM * DIM length
    return genStart + getGridIdx(row, col);
}

/**
 * Compute the serial version of this algorithm
 * @param initialGrid initial configuration
 * @return linearized 3-D Data grid of (generation, row, col) - length: GEN_COUNT
 */
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

/**
 * @return linearized 2-D Data grid of (row, col) of the configuration from DEFAULT_PATTERN_FILE
 */
bool* createGridFromFile()
{
    std::vector<std::vector<bool>> dataGrid;

    std::ifstream file(DEFAULT_PATTERN_FILE);
    if (file.is_open()) {
        std::string line;
        int count = 1;
        bool foundDig = false;
        std::vector<bool> curRow;
        while (std::getline(file, line)) {
            for (char c : line) {
                if (c == '#' || c =='x' || c == 'y') break;
                if (isdigit(c)) {
                    if (foundDig) count = count * 10 + (c - '0'); // increase decimal place
                    else count = c - '0';
                    foundDig = true;
                } else if (c == '$' || c == '!') {
                    std::vector<bool> tmp (curRow);
                    dataGrid.push_back(tmp);
                    curRow.clear();
                } else {
                    while (count-- > 0)
                        curRow.push_back(c == 'o');
                    count = 1;
                    foundDig = false;
                }
            }

        }

    } else {
        std::cout << "File cannot be opened!" << std::endl;
        exit(-1);
    }

    bool* initialGrid = new bool[DIM*DIM];
    for (unsigned int row = 1; row < DIM; row++) {
        for (unsigned int col = 1; col < DIM; col++) {
            unsigned int vecIdx = (row-1) % dataGrid.size();
            if (!dataGrid[vecIdx].empty())
                initialGrid[getGridIdx(row, col)] = dataGrid[vecIdx][(col-1) % dataGrid[vecIdx].size()];
        }
    }

    return initialGrid;
}

/**
 * @param gridIdx 2-D (non-linearized) index of row/col -> y/x
 * @return the GUI coordinate in the range [-1, 1]
 */
float getCoord(int gridIdx)
{
    return ((float)gridIdx * 2) / DIM - 1;
}

bool* g_lifeGrid;

/**
 * Display the Game of Life animation on the GUI
 */
void display() 
{ 
    
    glColor3f(0.0f, 0.0f, 0.0f);
    for (int gen = 0; gen < GEN_COUNT; gen++) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        for (int row = 0; row < DIM; row++) {
            for (int col = 0; col < DIM; col++) {
                if (g_lifeGrid[getActualIdxHost(gen, row, col)])
                    glRectf(getCoord(col), getCoord(row), getCoord(col + 1), getCoord(row + 1));             
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        glutSwapBuffers();
    } 
    
}

/**
 * Create a GUI panel to play the animation
 * @param lifeGrid linearized 3-D Data grid of (generation, row, col) - length: GEN_COUNT
 */
void graphics(bool* lifeGrid)
{
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(DIM + 20, DIM + 20);
    glutInitWindowPosition(10, 10);
    glutCreateWindow("CUDA Game of Life");
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    g_lifeGrid = lifeGrid;

    glutDisplayFunc(display);

    glutMainLoop();
}

/**
 * Print to the console the execution time of the argument's function
 * @param func Function to measure the execution time
 * @param initialGrid initial configuration
 * @return linearized 3-D Data grid of (generation, row, col) - length: GEN_COUNT
 */
bool* measureTime(const std::function<bool* (const bool*)>& func,
    const bool* initialGrid)
{
    using namespace std::chrono;

    auto start = system_clock::now();
    bool* result = func(initialGrid);
    auto end = system_clock::now();

    std::cout << "Execution Time: " << (duration_cast<milliseconds>(end - start)).count()
        << " milliseconds" << std::endl;

    return result;
}

int main(int argc, char* argv[])
{

    bool* initialGrid = createGridFromFile();
    for (int row = 0; row < DIM; row++) initialGrid[getGridIdx(row, 0)] = false; // West edge
    for (int row = 0; row < DIM; row++) initialGrid[getGridIdx(row, DIM-1)] = false; // East edge
    for (int col = 0; col < DIM; col++) initialGrid[getGridIdx(0, col)] = false; // North edge
    for (int col = 0; col < DIM; col++) initialGrid[getGridIdx(DIM-1, col)] = false; // South edge

    std::cout << "Running " << GEN_COUNT << " generations on grid of " << DIM << "x" << DIM << std:: endl;

    std::cout << "CUDA version: " << std::endl;
    bool* cudaResult = measureTime(cudaLife, initialGrid);

    std::cout << "Serial version: " << std:: endl;
    bool* serialResult = measureTime(serialLife, initialGrid);  

    bool matched = true;
    for (int row = 0; row < DIM && matched; row++) {
        for (int col = 0; col < DIM && matched; col++) {
            if (  serialResult[getActualIdxHost(GEN_COUNT-1,row,col)]
               != cudaResult[getActualIdxHost(GEN_COUNT-1,row,col)]) {
                matched = false;
                std::cout << "Failed at row " << row << " col " << col << std:: endl;
            }
        }
    }

    if (!matched) std::cout << "Not matched!" << std:: endl;
    else std::cout << "Last generation of serial vs CUDA matched!" << std:: endl;

    graphics(cudaResult);

    delete[] initialGrid;
    delete[] serialResult;
    delete[] cudaResult;
}
