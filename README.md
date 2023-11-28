# Simple LatticeBoltzmann Fluid (CUDA + OpenGL Interop)
<br>
<div class="row">
  Velocity:
  <img src="Examples/FluidVelocity.png?raw=true" width="1000">
</div>
<br>
My implementation is based on the Python fluid sim code shown in this video ( https://www.youtube.com/watch?v=JFWqCQHg-Hs&ab_channel=MatiasOrtiz ) which is based on Philip Mocz's work, as well as the Cuda-Samples by Nvidia.

Youtube video demo: ( https://youtu.be/LI3d_Zj7OMo )

Installation:

&emsp;Unzip the Library.zip and place "Common" folder inside the project folder.

&emsp;Go to CUDA_LatticeBoltzmann_Fluid_2D\x64\Release and click "CUDA_LatticeBoltzmann_Fluid_2D.exe" 

&emsp;It appears that "freeglut.dll" and "glew64.dll" has to be present in the "x64 -> Release" or "x64 -> Debug" to build it.

Steps:<br>
Drift:<br>
&emsp;DriftNorth()<br>
&emsp;DriftNorthEast()<br>
&emsp;DriftEast()<br>
&emsp;DriftSouthEast()<br>
&emsp;DriftSouth()<br>
&emsp;DriftSouthWest()<br>
&emsp;DriftWest()<br>
&emsp;DriftNorthWest()<br>
Momentum()<br>
ApplyBoundary()<br>
SolveTimeStep()<br>
DisplayVelocity()<br>
<br>
A kernel grid contains as many blocks as window's height<br>
A block contains as many threads as a window's width<br>
This means each thread block is responsible for each row of the image. It appears that blocks can begin in any order<br>
Each thread is responsible for one pixel. It appears that threads within a block print sequentially, although it does not mean threads are sequential<br>
Performance:
  <br>&emsp;For (512 x 512), Even the unoptimized code seems to take about 150 microseconds in all the 12 kernels.
  <br>Below is a screen capture from Nsight Compute:
  <br>&emsp;The duration is in micro-seconds.
  <img src="Examples/NsightCompute.png?raw=true" width="1000">
I am getting "tail effect" warning from all kernels except "SolveTimeStep()" and "Momentum()"<br>
<div class="row">
  <img src="Examples/ComputeSummary.png?raw=true" width="1000">
</div>
