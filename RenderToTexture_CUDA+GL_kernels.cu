/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <iostream>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>  // Helper functions for CUDA Error handling
#include <cooperative_groups.h>
#include <helper_string.h>

#include "RenderToTexture_CUDA+GL.h"

// Texture object for reading image
cudaTextureObject_t texObject;
static cudaArray *array = NULL;
const bool DEBUG = false;
const float tau = 0.6f;
float* d_Ce; // 0: Center
float* d_No; // 1: North
float* d_NE;
float* d_Ea;
float* d_SE;
float* d_So;
float* d_SW;
float* d_We;
float* d_NW;
bool* d_cylinder;
float* d_rho;
float* d_ux;
float* d_uy;

__global__ void DriftNorth(Pixel* pixels, int width, int height, float* data) {
    //unsigned char* pData = (unsigned char*)(((char*)pixels) + blockIdx.x * width);
    int elem = blockIdx.x * width + threadIdx.x;
    if (blockIdx.x == blockDim.x - 1) {
        data[elem] = data[threadIdx.x];
    }
    else {
        float temp = data[elem + width];
        data[elem] = temp;
    }
    //pData[threadIdx.x] = (unsigned char)min(max(data[elem] * 128, 0.f), 255.f);
    if (DEBUG) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            printf("No: Blk(%d) Tred(%d): %2.5f\n", blockIdx.x, threadIdx.x, data[elem]);
        }
    }
}
__global__ void DriftNorthEast(Pixel* pixels, int width, int height, float* data) {
    //unsigned char* pData = (unsigned char*)(((char*)pixels) + blockIdx.x * width);
    int elem = blockIdx.x * width + threadIdx.x;
    if (threadIdx.x == (width - 1)) {
        // boundary absorption
    }
    else if (blockIdx.x == blockDim.x - 1 && threadIdx.x == 0) {
        data[elem] = data[width - 1];
    }
    else if (blockIdx.x == blockDim.x - 1) {
        data[elem] = data[threadIdx.x];
    } 
    else if (threadIdx.x == 0) {
        data[elem] = data[(blockIdx.x + 1) * width - 1];
    }
    else {
        float temp = data[elem + width - 1];
        data[elem] = temp;
    }
    //pData[threadIdx.x] = (unsigned char)min(max(powf(data[elem] * 2, 8.f), 0.f), 255.f);
    if (DEBUG) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            printf("NE: Blk(%d) Tred(%d): %2.5f\n", blockIdx.x, threadIdx.x, data[elem]);
        }
    }
}
__global__ void DriftEast(Pixel* pixels, int width, int height, float* data) {
    //unsigned char* pData = (unsigned char*)(((char*)pixels) + blockIdx.x * width);
    int elem = blockIdx.x * width + threadIdx.x;
    if (threadIdx.x == (width - 1)) {
        // boundary absorption
    }
    else if (threadIdx.x == 0) {
        data[elem] = data[(blockIdx.x + 1) * width - 1];
    } // East Boundary: Velocity is absorbed
    else {
        float temp = data[elem - 1];
        data[elem] = temp;
    }
    //pData[threadIdx.x] = (unsigned char)min(max(data[elem] * 128, 0.f), 255.f);
    if (DEBUG) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            printf("Ea: Blk(%d) Tred(%d): %2.5f\n", blockIdx.x, threadIdx.x, data[elem]);
        }
    }
}
__global__ void DriftSouthEast(Pixel* pixels, int width, int height, float* data) {
    //unsigned char* pData = (unsigned char*)(((char*)pixels) + blockIdx.x * width);
    int elem = blockIdx.x * width + threadIdx.x;
    if (threadIdx.x == (width - 1)) {
        // boundary absorption
    }
    else if (blockIdx.x == 0 && threadIdx.x == 0) {
        data[elem] = data[blockDim.x * width - 1];
    }
    else if (blockIdx.x == 0) {
        data[elem] = data[(blockDim.x - 1) * width + threadIdx.x];
    }
    else if (threadIdx.x == 0) {
        data[elem] = data[(blockIdx.x + 1) * width - 1];
    }
    else {
        float temp = data[elem - width - 1];
        data[elem] = temp;
    }
    //pData[threadIdx.x] = (unsigned char)min(max(powf(data[elem] * 2, 8.f), 0.f), 255.f);
    if (DEBUG) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            printf("SE: Blk(%d) Tred(%d): %2.5f\n", blockIdx.x, threadIdx.x, data[elem]);
        }
    }
}
__global__ void DriftSouth(Pixel* pixels, int width, int height, float* data) {
    //unsigned char* pData = (unsigned char*)(((char*)pixels) + blockIdx.x * width);
    int elem = blockIdx.x * width + threadIdx.x;
    if (blockIdx.x == 0) {
        data[elem] = data[(blockDim.x - 1) * width + threadIdx.x];
        //printf("Blk(% d, % d) Tred(% d, % d): %3.2f \n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, data[elem]);
    }
    else {
        float temp = data[elem - width];
        data[elem] = temp;
    }
    //pData[threadIdx.x] = (unsigned char)min(max(powf(data[elem] * 2, 8.f), 0.f), 255.f);
    if (DEBUG) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            printf("So: Blk(%d) Tred(%d): %2.5f\n", blockIdx.x, threadIdx.x, data[elem]);
        }
    }
}
__global__ void DriftSouthWest(Pixel* pixels, int width, int height, float* data) {
    //unsigned char* pData = (unsigned char*)(((char*)pixels) + blockIdx.x * width);
    int elem = blockIdx.x * width + threadIdx.x;
    if (threadIdx.x == 0) {
        // boundary absorption
    }
    else if (blockIdx.x == 0 && threadIdx.x == (width - 1)) {
        data[elem] = data[(blockDim.x - 1) * width];
    }
    else if (blockIdx.x == 0) {
        data[elem] = data[(blockDim.x - 1) * width + threadIdx.x];
    }
    else if (threadIdx.x == (width - 1)) {
        data[elem] = data[blockIdx.x * width];
    }
    else {
        float temp = data[elem - width + 1];
        data[elem] = temp;
    }
    //pData[threadIdx.x] = (unsigned char)min(max(powf(data[elem] * 2, 8.f), 0.f), 255.f);
    if (DEBUG) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            printf("SW: Blk(%d) Tred(%d): %2.5f\n", blockIdx.x, threadIdx.x, data[elem]);
        }
    }
}
__global__ void DriftWest(Pixel* pixels, int width, int height, float* data) {
    //unsigned char* pData = (unsigned char*)(((char*)pixels) + blockIdx.x * width);
    int elem = blockIdx.x * width + threadIdx.x;
    if (threadIdx.x == 0) {
        // boundary absorption
    }
    else if (threadIdx.x == (width - 1)) {
        data[elem] = data[blockIdx.x * width];
    }
    else {
        float temp = data[elem + 1];
        data[elem] = temp;
    }
    //pData[threadIdx.x] = (unsigned char)min(max(data[elem] * 128, 0.f), 255.f);
    if (DEBUG) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            printf("We: Blk(%d) Tred(%d): %2.5f\n", blockIdx.x, threadIdx.x, data[elem]);
        }
    }
}
__global__ void DriftNorthWest(Pixel* pixels, int width, int height, float* data) {
    //unsigned char* pData = (unsigned char*)(((char*)pixels) + blockIdx.x * width);
    int elem = blockIdx.x * width + threadIdx.x;
    if (threadIdx.x == 0) {
        // boundary absorption
    }
    else if (blockIdx.x == (blockDim.x - 1) && threadIdx.x == (width - 1)) {
        data[elem] = data[0];
    } else if (blockIdx.x == (blockDim.x - 1)) {
        data[elem] = data[threadIdx.x];
    } else if (threadIdx.x == (width - 1)) {
        data[elem] = data[blockIdx.x * width];
    }
    else {
        float temp = data[elem + width + 1];
        //printf("Blk(% d, % d) Tred(% d, % d) Src: %2.2f, Dst: %2.2f \n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, d_No[elem], temp);
        data[elem] = temp;
    }
    //pData[threadIdx.x] = (unsigned char)min(max(powf(data[elem] * 2, 8.f), 0.f), 255.f);
    if (DEBUG) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            printf("NW: Blk(%d) Tred(%d): %2.5f\n", blockIdx.x, threadIdx.x, data[elem]);
        }
    }
}
__global__ void Momentum(Pixel* pixels, int width, int height, float* d_rho, float* d_ux, float* d_uy, bool* d_cylinder, float* d_Ce, float* d_No, float* d_NE, float* d_Ea, float* d_SE, float* d_So, float* d_SW, float* d_We, float* d_NW) {
    //unsigned char* pData = (unsigned char*)(((char*)pixels) + blockIdx.x * width);
    int elem = blockIdx.x * width + threadIdx.x;
    //printf("%3.3f \n", d_ux[elem]);
    d_rho[elem] = (d_Ce[elem] + d_No[elem] + d_NE[elem] + d_Ea[elem] + d_SE[elem] + d_So[elem] + d_SW[elem] + d_We[elem] + d_NW[elem]);
    d_ux[elem] = (d_NE[elem] + d_Ea[elem] + d_SE[elem] - d_SW[elem] - d_We[elem] - d_NW[elem]) / d_rho[elem];
    d_uy[elem] = (d_No[elem] + d_NE[elem] - d_SE[elem] - d_So[elem] - d_SW[elem] + d_NW[elem]) / d_rho[elem];
    //pData[threadIdx.x] = (unsigned char)min(max((rho / 9.0f * 256.0f), 0.f), 255.f);
    if (DEBUG) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            printf("Rho: Blk(% d, % d) Tred(% d, % d) rho:%3.5f \n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, d_rho[elem]);
        }
    }
}
/*
* 8 -- 1 -- 2
* -    -    -
* 7 -- 0 -- 3
* -    -    -
* 6 -- 5 -- 4
*/
__global__ void ApplyBoundary(Pixel* pixels, int width, int height, float* d_ux, float* d_uy, bool* d_cylinder, float* d_No, float* d_NE, float* d_Ea, float* d_SE, float* d_So, float* d_SW, float* d_We, float* d_NW) {
    int elem = blockIdx.x * width + threadIdx.x;
    //printf("%3.3f \n", d_ux[elem]);
    if (d_cylinder[elem]) {
        // Reflect
        // Swap 1:North 5:South
        float temp = d_No[elem];
        d_No[elem] = d_So[elem];
        d_So[elem] = temp;
        // Swap 2:NorthEast 6:SouthWest
        temp = d_NE[elem];
        d_NE[elem] = d_SW[elem];
        d_SW[elem] = temp;
        // Swap 3:East 7:West
        temp = d_Ea[elem];
        d_Ea[elem] = d_We[elem];
        d_We[elem] = temp;
        // Swap 4:SouthEast 8:NorthWest
        temp = d_SE[elem];
        d_SE[elem] = d_NW[elem];
        d_NW[elem] = temp;
        // there is no fluid movement within the boundary
        d_ux[elem] = 0;
        d_uy[elem] = 0;
    }
    if (DEBUG) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            printf("BoundUX: Blk(% d, % d) Tred(% d, % d) ux:%3.5f \n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, d_ux[elem]);
            printf("BoundUY: Blk(% d, % d) Tred(% d, % d) uy:%3.5f \n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, d_uy[elem]);
        }
    }
}
__device__ float F_equilibrium_timeStep(int dir_x, int dir_y, float rho, float weight, float ux, float uy, float lattice, float tau) {
    float feq = rho * weight * (1.f + 3.f * (dir_x * ux + dir_y * uy) + 9.f * powf(dir_x * ux + dir_y * uy, 2.f) / 2.0f - 3.f * (powf(ux, 2.f) + powf(uy, 2.f)) / 2.0f);
    if (DEBUG) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            printf("feq: Blk(% d, % d) Tred(% d, % d) feq:%3.5f \n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, feq);
        }
    }
    return -(1.0f / tau) * (lattice - feq);
}
__global__ void SolveTimeStep(Pixel* pixels, int width, int height, float tau, float* d_rho, float* d_ux, float* d_uy, float* d_Ce, float* d_No, float* d_NE, float* d_Ea, float* d_SE, float* d_So, float* d_SW, float* d_We, float* d_NW) {
    int elem = blockIdx.x * width + threadIdx.x;
    d_Ce[elem] += F_equilibrium_timeStep( 0, 0, d_rho[elem], 4 / 9.0f, d_ux[elem], d_uy[elem], d_Ce[elem], tau);
    d_No[elem] += F_equilibrium_timeStep( 0, 1, d_rho[elem], 1 / 9.0f, d_ux[elem], d_uy[elem], d_No[elem], tau);
    d_NE[elem] += F_equilibrium_timeStep( 1, 1, d_rho[elem], 1 / 36.f, d_ux[elem], d_uy[elem], d_NE[elem], tau);
    d_Ea[elem] += F_equilibrium_timeStep( 1, 0, d_rho[elem], 1 / 9.0f, d_ux[elem], d_uy[elem], d_Ea[elem], tau);
    d_SE[elem] += F_equilibrium_timeStep( 1,-1, d_rho[elem], 1 / 36.f, d_ux[elem], d_uy[elem], d_SE[elem], tau);
    d_So[elem] += F_equilibrium_timeStep( 0,-1, d_rho[elem], 1 / 9.0f, d_ux[elem], d_uy[elem], d_So[elem], tau);
    d_SW[elem] += F_equilibrium_timeStep(-1,-1, d_rho[elem], 1 / 36.f, d_ux[elem], d_uy[elem], d_SW[elem], tau);
    d_We[elem] += F_equilibrium_timeStep(-1, 0, d_rho[elem], 1 / 9.0f, d_ux[elem], d_uy[elem], d_We[elem], tau);
    d_NW[elem] += F_equilibrium_timeStep(-1, 1, d_rho[elem], 1 / 36.f, d_ux[elem], d_uy[elem], d_NW[elem], tau);
    /*if (blockIdx.x == 1 && threadIdx.x == 1) {
        printf("Blk(% d, % d) Tred(% d, % d) d_NO:%3.2f \n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, d_No[elem]);
    }*/
    if (DEBUG) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            printf("d_Ce: Blk(%d) Tred(%d): %2.5f\n", blockIdx.x, threadIdx.x, d_Ce[elem]);
            printf("d_No: Blk(%d) Tred(%d): %2.5f\n", blockIdx.x, threadIdx.x, d_No[elem]);
            printf("d_NE: Blk(%d) Tred(%d): %2.5f\n", blockIdx.x, threadIdx.x, d_NE[elem]);
            printf("d_Ea: Blk(%d) Tred(%d): %2.5f\n", blockIdx.x, threadIdx.x, d_Ea[elem]);
            printf("d_SE: Blk(%d) Tred(%d): %2.5f\n", blockIdx.x, threadIdx.x, d_SE[elem]);
            printf("d_So: Blk(%d) Tred(%d): %2.5f\n", blockIdx.x, threadIdx.x, d_So[elem]);
            printf("d_SW: Blk(%d) Tred(%d): %2.5f\n", blockIdx.x, threadIdx.x, d_SW[elem]);
            printf("d_We: Blk(%d) Tred(%d): %2.5f\n", blockIdx.x, threadIdx.x, d_We[elem]);
            printf("d_NW: Blk(%d) Tred(%d): %2.5f\n", blockIdx.x, threadIdx.x, d_NW[elem]);
        }
    }
}
__global__ void DisplayVelocity(Pixel* pixels, int width, float* d_ux, float* d_uy, float brightness) {
    unsigned char* pData = (unsigned char*)(((char*)pixels) + blockIdx.x * width);
    int elem = blockIdx.x * width + threadIdx.x;
    float velocity = (powf(d_ux[elem], 2.f) + powf(d_uy[elem], 2.f));
    if (DEBUG) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            printf("Display Vel: Blk(% d, % d) Tred(% d, % d) vel:%3.5f \n\n\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, velocity);
        }
    }
    velocity = (-1 * expm1f(-velocity * brightness * 0.1f)) * 255.f; // exposure tonemapping
    if (DEBUG) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            printf("Tonemap Vel: Blk(% d, % d) Tred(% d, % d) vel:%3.5f \n\n\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, velocity);
        }
    }
    //printf("Blk(% d, % d) Tred(% d, % d): %3.2f \n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, velocity);
    //printf("Blk(% d, % d) Tred(% d, % d): %3.2f \n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, d_uy[elem]);
    pData[threadIdx.x] = (unsigned char) min( max(velocity, 0.f), 255.f );
}
/*
__global__ void DisplayVorticity(Pixel* pixels, int width, float* d_ux, float* d_uy, float brightness) {
    unsigned char* pData = (unsigned char*)(((char*)pixels) + blockIdx.x * width);
    int elem = blockIdx.x * width + threadIdx.x;
    float ux1 = 0.f; //ignore the top two rows, ignore the first and the last columns
    if (blockIdx.x >= 2 && threadIdx.x == 0 && threadIdx.x == (width - 1)) {
        ux1 = d_ux[elem];
    }
    float ux2 = 0.f; // ignore the bottom two rows, ignore the first and the last columns
    if (blockIdx.x < (blockDim.x - 2) && threadIdx.x == 0 && threadIdx.x == (width - 1)) {
        break;
    }
    float curl = 0.f;
    pData[threadIdx.x] = (unsigned char)min(max(curl, 0.f), 255.f);
}*/
__global__ void Kernel_01(Pixel *pixels, int width, int height, float fScale, cudaTextureObject_t texObj) {
    unsigned char* pData = (unsigned char*)(((char*)pixels) + blockIdx.x * width);
    pData[threadIdx.x] = min(max((tex2D<unsigned char>(texObj, (float)threadIdx.x, (float)blockIdx.x) * fScale), 0.f), 255.f);
}
// Wrapper for the __global__ call that sets up the texture and threads
extern "C" void Kernal(Pixel *odata, int width, int height, enum DisplayMode mode, float fScale) {
    dim3 THREADS(width, 1, 1); // Z-dimension is 1 by default.
    dim3 BLOCKS(height, 1, 1);
    //Kernel_01 << <BLOCKS, THREADS >> > (odata, width, height, fScale, texObject);
    /*const int ARRAY_SIZE = width * height;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);
    float* tempNo = (float*)malloc(ARRAY_BYTES);*/
    DriftNorth      << < BLOCKS, THREADS >> > (odata, width, height, d_No);
    DriftNorthEast << < BLOCKS, THREADS >> > (odata, width, height, d_NE);
    DriftEast       << < BLOCKS, THREADS >> > (odata, width, height, d_Ea);
    DriftSouthEast  << < BLOCKS, THREADS >> > (odata, width, height, d_SE);
    DriftSouth      << < BLOCKS, THREADS >> > (odata, width, height, d_So);
    DriftSouthWest << < BLOCKS, THREADS >> > (odata, width, height, d_SW);
    DriftWest       << < BLOCKS, THREADS >> > (odata, width, height, d_We);
    DriftNorthWest << < BLOCKS, THREADS >> > (odata, width, height, d_NW);
    Momentum << < BLOCKS, THREADS >> > (odata, width, height, d_rho, d_ux, d_uy, d_cylinder, d_Ce, d_No, d_NE, d_Ea, d_SE, d_So, d_SW, d_We, d_NW);
    ApplyBoundary << < BLOCKS, THREADS >> > (odata, width, height, d_ux, d_uy, d_cylinder, d_No, d_NE, d_Ea, d_SE, d_So, d_SW, d_We, d_NW);
    SolveTimeStep << < BLOCKS, THREADS >> > (odata, width, height, tau, d_rho, d_ux, d_uy, d_Ce, d_No, d_NE, d_Ea, d_SE, d_So, d_SW, d_We, d_NW);
    DisplayVelocity << < BLOCKS, THREADS >> > (odata, width, d_ux, d_uy, fScale);
    //DisplayVorticity << < BLOCKS, THREADS >> > (odata, width, d_ux, d_uy, fScale);
    
    //std::cout << "Pixel: " << mode << std::endl;
    switch (mode) {
        case VELOCITY:
            break;

        case VORTICITY:
            break;
    }
}

float distanceSq(float offset_x, float offset_y, float x, float y) {
    return std::pow(x - offset_x, 2) + std::pow(y - offset_y, 2);
}

extern "C" void initMemory(int wWidth, int wHeight) {
    // initialize lattice velocities on the host and transfer it to device
    const int ARRAY_SIZE = wWidth * wHeight;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);
    float* Ce = (float*)malloc(ARRAY_BYTES);
    float* No = (float*)malloc(ARRAY_BYTES);
    float* NE = (float*)malloc(ARRAY_BYTES);
    float* Ea = (float*)malloc(ARRAY_BYTES);
    float* SE = (float*)malloc(ARRAY_BYTES);
    float* So = (float*)malloc(ARRAY_BYTES);
    float* SW = (float*)malloc(ARRAY_BYTES);
    float* We = (float*)malloc(ARRAY_BYTES);
    float* NW = (float*)malloc(ARRAY_BYTES);
    bool* cylinder = (bool*)malloc(ARRAY_SIZE);
    float* rho = (float*)malloc(ARRAY_BYTES); // density 
    float* ux = (float*)malloc(ARRAY_BYTES);
    float* uy = (float*)malloc(ARRAY_BYTES);
    //float rho = 11.3f; // sum of all lattice velocities
    for (int i = 0; i < ARRAY_SIZE; i++) {
        Ce[i] = 0.95f + rand() % 100 / 1000.f;
        No[i] = 0.95f + rand() % 100 / 1000.f;
        NE[i] = 0.95f + rand() % 100 / 1000.f;
        Ea[i] = 0.95f + rand() % 100 / 1000.f + 2.3f; // velocity of the right cell.
        SE[i] = 0.95f + rand() % 100 / 1000.f;
        So[i] = 0.95f + rand() % 100 / 1000.f;
        SW[i] = 0.95f + rand() % 100 / 1000.f;
        We[i] = 0.95f + rand() % 100 / 1000.f;
        NW[i] = 0.95f + rand() % 100 / 1000.f;
        if (distanceSq(wWidth / 4, wHeight / 2, i % wWidth, i / wWidth ) < 256) {
            cylinder[i] = true;
            if (DEBUG) {
                std::cout << "Cylinder: " << i / wWidth << ", " << i % wWidth << std::endl;
            }
        }
        else {
            cylinder[i] = false;
        }
        rho[i] = 11.3f;
        ux[i] = 0.f;
        uy[i] = 0.f;
    }
    //
    cudaMalloc((void**)&d_Ce, ARRAY_BYTES);
    cudaMalloc((void**)&d_No, ARRAY_BYTES);
    cudaMalloc((void**)&d_NE, ARRAY_BYTES);
    cudaMalloc((void**)&d_Ea, ARRAY_BYTES);
    cudaMalloc((void**)&d_SE, ARRAY_BYTES);
    cudaMalloc((void**)&d_So, ARRAY_BYTES);
    cudaMalloc((void**)&d_SW, ARRAY_BYTES);
    cudaMalloc((void**)&d_We, ARRAY_BYTES);
    cudaMalloc((void**)&d_NW, ARRAY_BYTES);
    cudaMalloc((void**)&d_cylinder, ARRAY_SIZE);
    cudaMalloc((void**)&d_rho, ARRAY_BYTES);
    cudaMalloc((void**)&d_ux, ARRAY_BYTES);
    cudaMalloc((void**)&d_uy, ARRAY_BYTES);
    cudaMemcpy(d_Ce, Ce, ARRAY_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_No, No, ARRAY_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_NE, NE, ARRAY_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ea, Ea, ARRAY_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_SE, SE, ARRAY_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_So, So, ARRAY_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_SW, SW, ARRAY_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_We, We, ARRAY_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_NW, NW, ARRAY_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cylinder, cylinder, ARRAY_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rho, rho, ARRAY_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ux, ux, ARRAY_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_uy, uy, ARRAY_BYTES, cudaMemcpyHostToDevice);
    free(Ce);
    free(No);
    free(NE);
    free(Ea);
    free(SE);
    free(So);
    free(SW);
    free(We);
    free(NW);
    free(cylinder);
    free(ux);
    free(uy);
}

extern "C" void setupTexture(int iw, int ih, Pixel * data, int Bpp) {
    cudaChannelFormatDesc desc;

    if (Bpp == 1) {
        desc = cudaCreateChannelDesc<unsigned char>();
    }
    else {
        desc = cudaCreateChannelDesc<uchar4>();
    }

    checkCudaErrors(cudaMallocArray(&array, &desc, iw, ih));
    checkCudaErrors(cudaMemcpy2DToArray(array, 0, 0, data, iw * Bpp * sizeof(Pixel), iw * Bpp * sizeof(Pixel), ih, cudaMemcpyHostToDevice));
    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = array;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = false;
    texDescr.filterMode = cudaFilterModePoint;
    texDescr.addressMode[0] = cudaAddressModeWrap;
    texDescr.readMode = cudaReadModeElementType;

    checkCudaErrors(cudaCreateTextureObject(&texObject, &texRes, &texDescr, NULL));
}

extern "C" void deleteTexture(void) {
    checkCudaErrors(cudaFreeArray(array));
    checkCudaErrors(cudaDestroyTextureObject(texObject));
}
