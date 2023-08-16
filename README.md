# Simple LatticeBoltzmann Fluid (CUDA + OpenGL Interop)

My implementation is based on the Python fluid sim code shown in this video ( https://www.youtube.com/watch?v=JFWqCQHg-Hs&ab_channel=MatiasOrtiz ) which is based on Philip Mocz's work, as well as the Cuda-Samples by Nvidia.

Installation:
  Unzip the Library.zip and place "Common" folder inside the project folder.

  It appears that "freeglut.dll" and "glew64.dll" has to be present in the "x64 -> Release" or "x64 -> Debug" to build it.

Demo:
  Even the unoptimized code seems to take about 150 microseconds in all the 12 kernels.
  See the video here: ( https://youtu.be/LI3d_Zj7OMo )
<div class="row">
  Velocity:
  <img src="Examples/FluidVelocity.png?raw=true" width="1000">
</div>
