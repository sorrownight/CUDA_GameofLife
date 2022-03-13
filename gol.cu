#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <chrono>
#include <ctime>
#include <functional>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <GL/glut.h>
#include <conio.h>
#include <thread>

const unsigned int THREAD_1D = 16;
//const unsigned int BLOCK_AREA = THREAD_1D * THREAD_1D;
const unsigned int BLOCK_1D = 80;
const unsigned int DIM = (THREAD_1D * BLOCK_1D);
// Reserving 4 rows/cols of 0s on the border to eliminate edge cases

const unsigned int GEN_COUNT = 2000;
const std::string DEFAULT_PATTERN_FILE = "data.txt";


// Each generation takes up a space of DIM * DIM
// Each row takes up a space of DIM -> offset by col
__forceinline __device__ static unsigned int getActualIdx(unsigned int gen, unsigned int row, unsigned int col)
{
    unsigned int genStart = gen * DIM * DIM; // Each subarray has DIM * DIM length
    return genStart + row * DIM + col;
}

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

        gens[getActualIdx(curGen,row,col)] = neighborCount == 3
                                            || (neighborCount == 2 && gens[getActualIdx(curGen-1,row,col)]);
    }
}

bool* firstIterLife(const bool* initialGrid)
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
        //cudaDeviceSynchronize();
        nextGen<<<dimGrid, dimBlock>>>(d_gens, gen);
    }
    bool *result = new bool[GEN_COUNT*DIM*DIM];
    cudaError_t code;
    if ((code = cudaMemcpyAsync(result, d_gens, (GEN_COUNT * DIM * DIM) * sizeof(bool), cudaMemcpyDeviceToHost)) != cudaSuccess)
        std::cout << "cudaMemcpy failed: Device->Host: " << cudaGetErrorString(code) << std::endl;
    
    cudaFree(d_gens);

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

bool* createGridFromFile()
{
    std::vector<std::vector<bool>> dataGrid;

    std::ifstream file(DEFAULT_PATTERN_FILE);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            std::vector<bool> curRow;

            for (char c : line) {
                if (c == '.') curRow.push_back(false);
                else curRow.push_back(true);
            }
            dataGrid.push_back(curRow);
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

float getCoord(int gridIdx)
{
    return ((float)gridIdx * 2) / DIM - 1;
}

bool* g_lifeGrid; // because OpenGL is dumb

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

    getch();//pause here to see results or lack there of
}

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
    bool* cudaResult = measureTime(firstIterLife, initialGrid);

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
