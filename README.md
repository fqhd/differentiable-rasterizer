# Drast
A fast, differentiable 2D rasterizer written in Rust that optimizes circle parameters to match a target image. Enables gradient-based shape optimization with support for animation output and resolution independent renderiong.

Traditional rasterizers draw shapes by splitting them into triangles and rendering those using highly optimized triangle rasterization algorithms such as scanline or barycentric. Unfortunately, the discrete operations used by these algorithms break differentiability and make it impossible to backpropagate a loss given a target value. Drast instead renders shapes in a fully differentiable manner using solely functions which are continuous and well defined throughout their entire domain.

---
## Demos
<table>
	<tr>
		<td align="center"><strong>Optimization Process</strong></td>
		<td align="center"><strong>Fully Converged</strong></td>
  	<td align="center"><strong>Target Image</strong></td>
	</tr>
	<tr>
		<td align="center"><img src="images/cat/animation.gif"/></td>
		<td align="center"><img src="images/cat/converged.png"/></td>
		<td align="center"><img src="images/cat/target.png"/></td>
	</tr>
	<tr>
		<td align="center"><img src="images/puppy/animation.gif"/></td>
		<td align="center"><img src="images/puppy/converged.png"/></td>
		<td align="center"><img src="images/puppy/target.png"/></td>
	</tr>
	<tr>
		<td align="center"><img src="images/paris/animation.gif"/></td>
		<td align="center"><img src="images/paris/converged.png"/></td>
		<td align="center"><img src="images/paris/target.png"/></td>
	</tr>
</table>

## How it works

### Initialization
The optimization process begins by initializing the parameters, position, size, and color, of a set of circles using some random probability distribution. A uniform distribution was used for the position and size to ensure an even spread throughout the canvas, though more testing is required to determine whether this is the best choice.

### Rasterization
To rasterize a circle differentiably, we loop through each pixel on the canvas and compute the L2 distance from that pixel to the center of the circle. Depending on whether the compute distance is greater than the radius, we can use that to determine whether the pixel is inside the circle. However, doing this process using branching would break differentiability. For this reason, we instead pass the distance through a parametrized sigmoid function which stretches values towards either 0 or 1 along the edge of the circle. This ensures a smoother rasterization with tunable antialiasing while preserving differentiability.

### Backpropagation
Finally, after summing the outputs of each circle on the raster image, we compute the L2 loss which is given to us using the equation (output - target)². To minimize this loss, we must calculate its partial derivative function with respect to each parameter. We use this function to compute the gradients(or slope) for each parameter and average them out before subtracting them from their respective parameters. This essentially "moves" each circle in a way that minimizes the global loss.

## AABB Optimization
The derivative of the sigmoid activation function approaches zero as x tends towards negative and positive infinity. This results in many of the gradients computed on the canvas to contribute very little to the overall movement of the circle. We can optimize the gradient computation by only considering points within the circle by only running the rasterization and backpropagation processes inside the bounding box of the circle. This speeds up the optimization process drastically.

## Build instructions

## Usage


## Todo
- SVG output
- GPU Acceleration
- Multiple shapes
- Better Parameter Initialization
