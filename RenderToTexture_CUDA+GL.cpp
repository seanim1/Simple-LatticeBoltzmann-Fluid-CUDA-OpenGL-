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
// https://www.nvidia.com/content/gtc/documents/1055_gtc09.pdf
// https://gist.github.com/prabindh/c1659ad73fc99df635e2153c1474ac4a
// OpenGL Graphics includes
#include <helper_gl.h>
#include <GL/freeglut.h>

// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include "RenderToTexture_CUDA+GL.h"

// includes, project
#include <helper_functions.h>  // includes for SDK helper functions
#include <helper_cuda.h>  // includes for cuda initialization and error checking

const char *filterMode[] = {"Velocity", "Vorticity", NULL};

// Use the '-' and '=' keys to change the scale factor.
void cleanup(void);
void initializeDataNoFile();

#define REFRESH_DELAY 1  // ms

const char *sSDKsample = "Simple LBM Fluid";

static int wWidth = 512;   // Window width
static int wHeight = 512;  // Window height

// Code to handle Auto verification
const int frameCheckNumber = 4;
int fpsCount = 0;  // FPS count for averaging
int fpsLimit = 8;  // FPS limit for sampling
unsigned int g_TotalErrors = 0;
StopWatchInterface *timer = NULL;
unsigned int g_Bpp;
unsigned int g_Index = 0;

// Display Data
static GLuint pbo_buffer = 0;  // Front and back CA buffers
struct cudaGraphicsResource *cuda_pbo_resource;  // CUDA Graphics Resource (to transfer PBO)

static GLuint texid = 0;       // Texture for display
unsigned char *pixels = NULL;  // Image pixel data on the host

float imageScale = 64.f;        // Image exposure
enum DisplayMode g_DisplayMode;

#define OFFSET(i) ((char *)NULL + (i))
#define MAX(a, b) ((a > b) ? a : b)

void computeFPS() {
  fpsCount++;

  if (fpsCount == fpsLimit) {
    char fps[256];
    float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
    sprintf(fps, "(%s) (%s): %3.1f fps", sSDKsample, filterMode[g_DisplayMode], ifps);

    glutSetWindowTitle(fps);
    fpsCount = 0;

    sdkResetTimer(&timer);
  }
}

// This is the normal display path
void display(void) {
    sdkStartTimer(&timer);

    Pixel *data = NULL;

    // map PBO to get CUDA device pointer
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&data, &num_bytes, cuda_pbo_resource));
    // printf("CUDA mapped PBO: May access %ld bytes\n", num_bytes);

    // Execute Kernel
    Kernal(data, wWidth, wHeight, g_DisplayMode, imageScale);
    //Kernal(data, wWidth, wHeight, g_DisplayMode, imageScale);

    //glutDestroyWindow(glutGetWindow()); return;
    // Unmap buffer object
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
    // Render from buffer object
    glClear(GL_COLOR_BUFFER_BIT);

    glBindTexture(GL_TEXTURE_2D, texid);// select the appropriate texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_buffer); // select the appropriate buffer
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, wWidth, wHeight, GL_LUMINANCE, GL_UNSIGNED_BYTE, OFFSET(0)); // make a texture from the buffer: The last parameter: Specifies a pointer to the image data in memory is set to NULL. Data is coming from a PBO, not host memroy.

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    // Draw the image. A single Quad with texture coordinates fro each vertex
    glBegin(GL_QUADS);
    glVertex2f(0, 0);
    glTexCoord2f(0, 0);
    glVertex2f(0, 1);
    glTexCoord2f(1, 0);
    glVertex2f(1, 1);
    glTexCoord2f(1, 1);
    glVertex2f(1, 0);
    glTexCoord2f(0, 1);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);
    glutSwapBuffers(); // double buffering

    sdkStopTimer(&timer);

    computeFPS();
}

void timerEvent(int value) {
    if (glutGetWindow()) {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    }
}

void keyboard(unsigned char key, int /*x*/, int /*y*/) {
  char temp[256];

  switch (key) {
    case 27:
    case 'q':
    case 'Q':
      printf("Shutting down...\n");
      glutDestroyWindow(glutGetWindow());
      return;
      break;

    case '-':
      imageScale -= 4.0f;
      printf("brightness = %4.2f\n", imageScale);
      break;

    case '=':
      imageScale += 4.0f;
      printf("brightness = %4.2f\n", imageScale);
      break;

    case 's':
    case 'S':
      g_DisplayMode = VELOCITY;
      glutSetWindowTitle(temp);
      break;

    case 't':
    case 'T':
      g_DisplayMode = VORTICITY;
      glutSetWindowTitle(temp);
      break;

    default:
      break;
  }
}

void reshape(int x, int y) {
    glViewport(0, 0, x, y);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, 0, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void cleanup(void) {
    cudaGraphicsUnregisterResource(cuda_pbo_resource);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glDeleteBuffers(1, &pbo_buffer);
    glDeleteTextures(1, &texid);
    deleteTexture();

    sdkDeleteTimer(&timer);
}

void initGL(int *argc, char **argv) {
  glutInit(argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
  glutInitWindowSize(wWidth, wHeight);
  glutCreateWindow(sSDKsample);

  if (!isGLVersionSupported(1, 5) ||
      !areGLExtensionsSupported(
          "GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object")) {
    fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
    fprintf(stderr, "This sample requires:\n");
    fprintf(stderr, "  OpenGL version 1.5\n");
    fprintf(stderr, "  GL_ARB_vertex_buffer_object\n");
    fprintf(stderr, "  GL_ARB_pixel_buffer_object\n");
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char **argv) {
    std::cout << "Have " << argc << " arguments:" << std::endl;
    for (int i = 0; i < argc; ++i) {
        std::cout << "Argument: " << argv[i] << std::endl;
    }

    initGL(&argc, argv);
    findCudaDevice(argc, (const char **)argv);

    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutReshapeFunc(reshape);

    initMemory(wWidth, wHeight);
    initializeDataNoFile();

    glutCloseFunc(cleanup);

    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    // launch the rendering loop
    glutMainLoop();
}

void initializeDataNoFile() {
    GLint bsize;
    g_Bpp = 1;

    pixels = new unsigned char[wWidth * wHeight]();
    
    for (int y = 0; y < wHeight; y++) {
        for (int x = 0; x < wWidth; x++) {
            pixels[y * wWidth + x] = (unsigned char) (rand() % 128);
        }
    }
    setupTexture(wWidth, wHeight, pixels, g_Bpp);

    memset(pixels, 0x0, g_Bpp * sizeof(Pixel) * wWidth * wHeight);

    // use OpenGL Path  // 1. Allocate the GL Buffer. Do only once at the startup
    glGenBuffers(1, &pbo_buffer); // Generate a buffer ID
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_buffer);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, g_Bpp * sizeof(Pixel) * wWidth * wHeight, pixels, GL_STREAM_DRAW);

    glGetBufferParameteriv(GL_PIXEL_UNPACK_BUFFER, GL_BUFFER_SIZE, &bsize);

    if ((GLuint)bsize != (g_Bpp * sizeof(Pixel) * wWidth * wHeight)) {
        printf("Buffer object (%d) has incorrect size (%d).\n",
            (unsigned)pbo_buffer, (unsigned)bsize);

        exit(EXIT_FAILURE);
    }

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo_buffer, cudaGraphicsMapFlagsWriteDiscard));

    glGenTextures(1, &texid); // Generate a texture ID
    glBindTexture(GL_TEXTURE_2D, texid); // Make this the current texture
    glTexImage2D(GL_TEXTURE_2D, 0, ((g_Bpp == 1) ? GL_LUMINANCE : GL_BGRA),
        wWidth, wHeight, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, NULL); // allocate the texture memory. BGRA8 format for little-endian format
    glBindTexture(GL_TEXTURE_2D, 0);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
}