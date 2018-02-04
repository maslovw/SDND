# CarND-Controls-MPC
Self-Driving Car Engineer Nanodegree Program

---
## Model Predictive Control (MPC)

The [MPC](https://en.wikipedia.org/wiki/Model_predictive_control) is an advanced control technique
for complex control problems.
MPC is an optmization problem to find the best set of control inputs that minimizes the cost
functions based on the prediction (dynamical) model.

### Prediction Horizon

Prediction Horizon is relatively short time duration over which future predictions are made.
It comprehend the number of timesteps `N` in the horizon and the time elapse of each timestep `dt`.

The number `N` also determines the number of variables optmized by the controller.
So, higher `N` will result in extra computational cost.

For this project, we followed an empirical approach of trial and error to choose the horizon values.
For ``N`` I tried values between 10 and 20 and for dt 0.05 and 0.1.
``dt`` must be in sync with the system latency, which is 100ms in this case.

### Model

The followind equations updates the prediction model at every timestep:

![equation](http://latex.codecogs.com/gif.latex?x_%28t&plus;1%29%20%3D%20x_t%20&plus;%20v_t%20*%20cos%28%5Cpsi_t%29*dt)

![equation](http://latex.codecogs.com/gif.latex?y_%28t&plus;1%29%20%3D%20y_t%20&plus;%20v_t%20*%20sin%28%5Cpsi_t%29*dt)

![equation](http://latex.codecogs.com/gif.latex?%5Cpsi%20_%28t&plus;1%29%20%3D%20%5Cpsi%20_t%20&plus;%20%5Cfrac%7Bv_t%7D%7BL_f%7D*%20%5Cdelta_t%20*%20dt)

![equation](http://latex.codecogs.com/gif.latex?v_%28t&plus;1%29%20%3D%20v%20_t%20&plus;%20a_t%20*%20dt)

![equation](http://latex.codecogs.com/gif.latex?cte_%28t&plus;1%29%20%3D%20f%28x_t%29%20-%20y_t%20&plus;%20v%20_t%20*%20sin%28e%5Cpsi%20_t%29%20*%20dt)

![equation](http://latex.codecogs.com/gif.latex?e%5Cpsi%20_%28t&plus;1%29%20%3D%20%5Cpsi%20_t%20-%20%5Cpsi%20dest%20&plus;%20%5Cfrac%7Bv_f%7D%7BL_f%7D%20*%20%5Cdelta_t%20*%20dt)


``Lf`` measures the distance between the front of the vehicle and its center of gravity. ``f(x)`` is the evaluation of the polynomial ``f`` at point x and ``psidest`` is the tangencial angle of the polynomial ``f`` evaluated at x.

### Polynomial Fitting and MPC Preprocessing

Before fitting the path returned from the simulator, we have to preprocess in order to move the points to the origin (x=0, y=0) and also rotate the path to follow the car orientation.

```c
for(size_t i = 0UL; i < ptsx.size(); i++){
	//shift points to the initial point
	double shift_x = ptsx[i] -px;
	double shift_y = ptsy[i] -py;

	//rotate 90 degrees
	//http://planning.cs.uiuc.edu/node99.html
	ptsx[i] = (shift_x * cos(0-psi) - shift_y*sin(0-psi));
	ptsy[i] = (shift_x * sin(0-psi) + shift_y*cos(0-psi));
}
```

After preprocessing, the polynomial is fitted using the helper function ``polyfit`` (file main.cpp at line 134).

### Constraints

The actuators constraints limits the upper and lower bounds of the steering angle and throttle acceleration/brake.

![equation](http://latex.codecogs.com/gif.latex?%5Cdelta%20%5Cepsilon%20%5B-25%5E%7B%5Ccirc%7D%2C%2025%5E%7B%5Ccirc%7D%5D)

![equation](http://latex.codecogs.com/gif.latex?a%20%5Cepsilon%20%5B-1%2C%201%5D)

The MPC cost captures the error to be minimized. The const function requires the model to predict where the vehicle will go into the future in order to compute the difference where the vehicle should be and where the model predicted.

![equation](http://latex.codecogs.com/gif.latex?J%20%3D%20%5Csum%5E%7BN%7D_%7Bt%3D1%7D%5B%28cte_t%20-%20cte_%7Bref%7D%29%5E2%20&plus;%20%28e%5Cpsi_t%20-%20e%5Cpsi_%7Bref%7D%29%5E2%20&plus;%20...%5D)

### Latency

In order to deal with the latency, we have to predict the next state before calling the MPC solver. It can be acoomplished using the Model equations. Below, the pseudocode to predict the next state.

```c
dt = 0.1;
x1    = v * cos(0) * dt;
y1    = v * sin(0) * dt;
psi1  = - v/Lf * steer_value * dt;
v1    = throttle_value * dt;
cte1  =   v * sin(epsi1) * dt;
epsi1 = - v * steer_value / Lf * dt;
```

## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1(mac, linux), 3.81(Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `install-mac.sh` or `install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets
    cd uWebSockets
    git checkout e94b6e1
    ```
    Some function signatures have changed in v0.14.x. See [this PR](https://github.com/udacity/CarND-MPC-Project/pull/3) for more details.

* **Ipopt and CppAD:** Please refer to [this document](https://github.com/udacity/CarND-MPC-Project/blob/master/install_Ipopt_CppAD.md) for installation instructions.
* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page). This is already part of the repo so you shouldn't have to worry about it.
* Simulator. You can download these from the [releases tab](https://github.com/udacity/self-driving-car-sim/releases).
* Not a dependency but read the [DATA.md](./DATA.md) for a description of the data sent back from the simulator.


## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./mpc`.

## Tips

1. It's recommended to test the MPC on basic examples to see if your implementation behaves as desired. One possible example
is the vehicle starting offset of a straight line (reference). If the MPC implementation is correct, after some number of timesteps
(not too many) it should find and track the reference line.
2. The `lake_track_waypoints.csv` file has the waypoints of the lake track. You could use this to fit polynomials and points and see of how well your model tracks curve. NOTE: This file might be not completely in sync with the simulator so your solution should NOT depend on it.
3. For visualization this C++ [matplotlib wrapper](https://github.com/lava/matplotlib-cpp) could be helpful.)
4.  Tips for setting up your environment are available [here](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/23d376c7-0195-4276-bdf0-e02f1f3c665d)
5. **VM Latency:** Some students have reported differences in behavior using VM's ostensibly a result of latency.  Please let us know if issues arise as a result of a VM environment.

## Editor Settings

We've purposefully kept editor configuration files out of this repo in order to
keep it as simple and environment agnostic as possible. However, we recommend
using the following settings:

* indent using spaces
* set tab width to 2 spaces (keeps the matrices in source code aligned)

## Code Style

Please (do your best to) stick to [Google's C++ style guide](https://google.github.io/styleguide/cppguide.html).

## Project Instructions and Rubric

Note: regardless of the changes you make, your project must be buildable using
cmake and make!

More information is only accessible by people who are already enrolled in Term 2
of CarND. If you are enrolled, see [the project page](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/f1820894-8322-4bb3-81aa-b26b3c6dcbaf/lessons/b1ff3be0-c904-438e-aad3-2b5379f0e0c3/concepts/1a2255a0-e23c-44cf-8d41-39b8a3c8264a)
for instructions and the project rubric.

## Hints!

* You don't have to follow this directory structure, but if you do, your work
  will span all of the .cpp files here. Keep an eye out for TODOs.

## Call for IDE Profiles Pull Requests

Help your fellow students!

We decided to create Makefiles with cmake to keep this project as platform
agnostic as possible. Similarly, we omitted IDE profiles in order to we ensure
that students don't feel pressured to use one IDE or another.

However! I'd love to help people get up and running with their IDEs of choice.
If you've created a profile for an IDE that you think other students would
appreciate, we'd love to have you add the requisite profile files and
instructions to ide_profiles/. For example if you wanted to add a VS Code
profile, you'd add:

* /ide_profiles/vscode/.vscode
* /ide_profiles/vscode/README.md

The README should explain what the profile does, how to take advantage of it,
and how to install it.

Frankly, I've never been involved in a project with multiple IDE profiles
before. I believe the best way to handle this would be to keep them out of the
repo root to avoid clutter. My expectation is that most profiles will include
instructions to copy files to a new location to get picked up by the IDE, but
that's just a guess.

One last note here: regardless of the IDE used, every submitted project must
still be compilable with cmake and make./

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).
