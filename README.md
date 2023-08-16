# Simple LatticeBoltzmann Fluid (CUDA + OpenGL Interop)
Unzip the Library.zip and place "Common" folder inside the project folder.

It appears that "freeglut.dll" and "glew64.dll" has to be present in the "x64 -> Release" or "x64 -> Debug" to build it.

Even the unoptimized code seems to take about 150 microseconds in all the 12 kernels.

<div class="row">
  Velocity:
  <img src="Examples/FluidVelocity.png?raw=true" width="1000">
</div>
