#define GLM_FORCE_CUDA

#include <cuda.h>
#include "kernel.h"
#include "utilityCore.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <utility>
#include <vector>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>

#include <glm/glm.hpp>

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

// LOOK-1.2 Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

/*! The neighborhood distance: the largest radius any of the three rules cares
    about. Nothing outside this radius can influence a boid. */
#define maxRuleDistance (imax(imax(rule1Distance, rule2Distance), rule3Distance))

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

/**
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rand = generateRandomVec3(time, index);
    arr[index].x = scale * rand.x;
    arr[index].y = scale * rand.y;
    arr[index].z = scale * rand.z;
  }
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  // LOOK-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  // Initialize velocity to 0
  cudaMemset(dev_vel1, 0, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMemset dev_vel1 failed!");

  cudaMemset(dev_vel2, 0, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMemset dev_vel2 failed!");

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  // LOOK-2.1 computing grid params
  gridCellWidth = 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  // 2.1 - One entry per boid: which boid this is, and which cell it landed in.
  // These two stay parallel; sorting them together is what bins the boids.
  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");

  // 2.1 - One entry per grid cell, marking the half-open range [start, end)
  // that this cell occupies in the sorted arrays.
  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");

  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

  // Wrap the raw device pointers so thrust::sort_by_key can drive them.
  dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
  dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);

  cudaDeviceSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

/**
* Running totals for the three boids rules.
*
* All three neighbor-search implementations (naive, scattered grid, coherent
* grid) funnel their candidate neighbors through this, so the physics stays
* bit-for-bit identical between them and any performance difference is purely
* a difference in how many candidates were examined and how they were fetched.
*/
struct RuleAccumulator {
  glm::vec3 center;    // Rule 1: sum of neighbor positions
  glm::vec3 separate;  // Rule 2: accumulated push-away vector
  glm::vec3 velocity;  // Rule 3: sum of neighbor velocities
  int centerCount;
  int velocityCount;

  __device__ RuleAccumulator()
    : center(0.0f), separate(0.0f), velocity(0.0f),
      centerCount(0), velocityCount(0) {}

  /** Fold one candidate neighbor into the totals. */
  __device__ void add(const glm::vec3 &selfPos, const glm::vec3 &otherPos,
                      const glm::vec3 &otherVel) {
    float dist = glm::distance(otherPos, selfPos);

    if (dist < rule1Distance) {
      center += otherPos;
      centerCount++;
    }
    if (dist < rule2Distance) {
      separate -= (otherPos - selfPos);
    }
    if (dist < rule3Distance) {
      velocity += otherVel;
      velocityCount++;
    }
  }

  /** Turn the totals into this timestep's velocity delta. */
  __device__ glm::vec3 delta(const glm::vec3 &selfPos) const {
    glm::vec3 change(0.0f);

    if (centerCount > 0) {
      glm::vec3 perceivedCenter = center / (float)centerCount;
      change += (perceivedCenter - selfPos) * rule1Scale;
    }

    change += separate * rule2Scale;

    if (velocityCount > 0) {
      // NOTE: per the project spec we do NOT subtract the boid's own velocity
      // here, which is where this diverges from Conrad Parker's notes. The
      // shipped rule scales are tuned for this version.
      glm::vec3 perceivedVelocity = velocity / (float)velocityCount;
      change += perceivedVelocity * rule3Scale;
    }

    return change;
  }
};

/** Cap a velocity at maxSpeed without changing its direction. */
__device__ glm::vec3 clampSpeed(glm::vec3 v) {
  float speed = glm::length(v);
  if (speed > maxSpeed) {
    v *= maxSpeed / speed;
  }
  return v;
}

/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
  // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
  // Rule 2: boids try to stay a distance d away from each other
  // Rule 3: boids try to match the speed of surrounding boids
  glm::vec3 selfPos = pos[iSelf];
  RuleAccumulator acc;

  // The naive search: every boid considers every other boid, O(N) per boid.
  for (int i = 0; i < N; i++) {
    if (i == iSelf) {
      continue;
    }
    acc.add(selfPos, pos[i], vel[i]);
  }

  return acc.delta(selfPos);
}

/**
* 1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }

  // Compute a new velocity based on pos and vel1
  glm::vec3 newVel = vel1[index] + computeVelocityChange(N, index, pos, vel1);

  // Clamp the speed
  newVel = clampSpeed(newVel);

  // Record the new velocity into vel2. Question: why NOT vel1?
  // Because every other thread in this launch is still reading vel1 as this
  // timestep's input. Writing in place would let a boid see some neighbors at
  // their old velocity and some at their new one, depending on scheduling -
  // a data race whose outcome varies run to run. vel2 keeps the read set
  // frozen for the whole step; the caller then ping-pongs the buffers.
  vel2[index] = newVel;
}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
  // Update position by velocity
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  glm::vec3 thisPos = pos[index];
  thisPos += vel[index] * dt;

  // Wrap the boids around so we don't lose them
  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  pos[index] = thisPos;
}

// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {
    // 2.1
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
      return;
    }

    // - Label each boid with the index of its grid cell.
    glm::vec3 offset = (pos[index] - gridMin) * inverseCellWidth;
    int x = imin(imax((int)floorf(offset.x), 0), gridResolution - 1);
    int y = imin(imax((int)floorf(offset.y), 0), gridResolution - 1);
    int z = imin(imax((int)floorf(offset.z), 0), gridResolution - 1);
    gridIndices[index] = gridIndex3Dto1D(x, y, z, gridResolution);

    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2. Right now this is just the identity;
    //   sorting it alongside gridIndices is what makes it a useful lookup.
    indices[index] = index;
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) {
  // 2.1
  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }

  int cell = particleGridIndices[index];

  // Each thread compares itself against its left neighbor. A mismatch is a
  // boundary: it closes the previous cell and opens this one. Ends are
  // exclusive, so a cell's boids are [start, end). Cells that no thread ever
  // touches keep the -1 that kernResetIntBuffer left behind, which is how the
  // neighbor search recognizes an empty cell.
  if (index == 0) {
    gridCellStartIndices[cell] = 0;
  } else {
    int prevCell = particleGridIndices[index - 1];
    if (prevCell != cell) {
      gridCellEndIndices[prevCell] = index;
      gridCellStartIndices[cell] = index;
    }
  }

  if (index == N - 1) {
    gridCellEndIndices[cell] = N;
  }
}

/**
* 2.3 - Gather the boid data into cell-sorted order.
* After this, boids that share a grid cell are contiguous in posSorted and
* velSorted, so the coherent kernel can walk a cell as a straight memory run.
*/

/**
* The body of the uniform-grid neighbor search.
*
* Rather than hard-coding "check 8 cells", this derives the cell range from the
* search radius. That skips cells a hard-coded neighborhood would have visited
* only to reject - the Grid-Looping Optimization from the extra credit.
*/
__device__ glm::vec3 gridNeighborVelocityChange(
  int iSelf, const glm::vec3 &selfPos, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth,
  const int *gridCellStartIndices, const int *gridCellEndIndices,
  const int *particleArrayIndices,
  const glm::vec3 *pos, const glm::vec3 *vel) {

  RuleAccumulator acc;

  // - Identify which cells may contain neighbors. This isn't always 8.
  //   Only cells overlapping the axis-aligned box of half-width
  //   maxRuleDistance around this boid can hold anything relevant.
  glm::vec3 lo = (selfPos - maxRuleDistance - gridMin) * inverseCellWidth;
  glm::vec3 hi = (selfPos + maxRuleDistance - gridMin) * inverseCellWidth;

  int xMin = imax((int)floorf(lo.x), 0);
  int yMin = imax((int)floorf(lo.y), 0);
  int zMin = imax((int)floorf(lo.z), 0);
  int xMax = imin((int)floorf(hi.x), gridResolution - 1);
  int yMax = imin((int)floorf(hi.y), gridResolution - 1);
  int zMax = imin((int)floorf(hi.z), gridResolution - 1);

  // gridIndex3Dto1D is x + y*R + z*R*R, so x varies fastest. Looping z outer
  // and x inner walks whole runs of consecutive cell indices, which in the
  // coherent case means consecutive boid data too.
  for (int z = zMin; z <= zMax; z++) {
    for (int y = yMin; y <= yMax; y++) {
      for (int x = xMin; x <= xMax; x++) {
        int cell = gridIndex3Dto1D(x, y, z, gridResolution);

        // - For each cell, read the start/end indices in the boid pointer array.
        int start = gridCellStartIndices[cell];
        if (start < 0) {
          continue;  // empty cell
        }
        int end = gridCellEndIndices[cell];

        // - Access each boid in the cell and compute velocity change from
        //   the boids rules, if this boid is within the neighborhood distance.
        for (int k = start; k < end; k++) {
          int b = particleArrayIndices[k];
          if (b == iSelf) {
            continue;
          }
          acc.add(selfPos, pos[b], vel[b]);
        }
      }
    }
  }

  return acc.delta(selfPos);
}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // 2.1 - Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }

  // Here a thread's index IS the boid index: pos and vel1 are still in their
  // original order, and only the neighbor lookup goes through the grid. That
  // is what makes this the "scattered" version - consecutive k in a cell land
  // on unrelated addresses in pos/vel.
  glm::vec3 selfPos = pos[index];

  glm::vec3 change = gridNeighborVelocityChange(
    index, selfPos, gridResolution, gridMin, inverseCellWidth,
    gridCellStartIndices, gridCellEndIndices, particleArrayIndices, pos, vel1);

  // - Clamp the speed change before putting the new speed in vel2
  vel2[index] = clampSpeed(vel1[index] + change);
}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
  // except with one less level of indirection.
  // This should expect gridCellStartIndices and gridCellEndIndices to refer
  // directly to pos and vel1.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  //   DIFFERENCE: For best results, consider what order the cells should be
  //   checked in to maximize the memory benefits of reordering the boids data.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  // 1.2 - use the kernels you wrote to step the simulation forward in time.
  kernUpdateVelocityBruteForce<<<fullBlocksPerGrid, blockSize>>>(
    numObjects, dev_pos, dev_vel1, dev_vel2);
  checkCUDAErrorWithLine("kernUpdateVelocityBruteForce failed!");

  kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel2);
  checkCUDAErrorWithLine("kernUpdatePos failed!");

  // 1.2 ping-pong the velocity buffers: this step's output becomes next
  // step's input, and the stale buffer becomes the next scratch target.
  std::swap(dev_vel1, dev_vel2);
}

void Boids::stepSimulationScatteredGrid(float dt) {
  // 2.1 - Uniform Grid Neighbor search using Thrust sort.
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  dim3 fullBlocksPerCellGrid((gridCellCount + blockSize - 1) / blockSize);

  // - label each particle with its array index as well as its grid index.
  kernComputeIndices<<<fullBlocksPerGrid, blockSize>>>(
    numObjects, gridSideCount, gridMinimum, gridInverseCellWidth,
    dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
  checkCUDAErrorWithLine("kernComputeIndices failed!");

  // - Unstable key sort using Thrust. Sorting the boid ids by cell id is what
  //   makes each cell's members contiguous in dev_particleArrayIndices.
  thrust::sort_by_key(dev_thrust_particleGridIndices,
    dev_thrust_particleGridIndices + numObjects,
    dev_thrust_particleArrayIndices);
  checkCUDAErrorWithLine("thrust::sort_by_key failed!");

  // -1 marks a cell as empty; without this reset, cells that emptied out since
  // the previous frame would still be advertising last frame's ranges.
  kernResetIntBuffer<<<fullBlocksPerCellGrid, blockSize>>>(
    gridCellCount, dev_gridCellStartIndices, -1);
  kernResetIntBuffer<<<fullBlocksPerCellGrid, blockSize>>>(
    gridCellCount, dev_gridCellEndIndices, -1);
  checkCUDAErrorWithLine("kernResetIntBuffer failed!");

  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  kernIdentifyCellStartEnd<<<fullBlocksPerGrid, blockSize>>>(
    numObjects, dev_particleGridIndices,
    dev_gridCellStartIndices, dev_gridCellEndIndices);
  checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

  // - Perform velocity updates using neighbor search
  kernUpdateVelNeighborSearchScattered<<<fullBlocksPerGrid, blockSize>>>(
    numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
    dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices,
    dev_pos, dev_vel1, dev_vel2);
  checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScattered failed!");

  // - Update positions
  kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel2);
  checkCUDAErrorWithLine("kernUpdatePos failed!");

  // - Ping-pong buffers as needed
  std::swap(dev_vel1, dev_vel2);
}

void Boids::stepSimulationCoherentGrid(float dt) {
  // TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
  // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
  // In Parallel:
  // - Label each particle with its array index as well as its grid index.
  //   Use 2x width grids
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
  //   the particle data in the simulation array.
  //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  // 2.1 - Free any additional buffers here.
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);
  checkCUDAErrorWithLine("cudaFree failed!");
}

void Boids::unitTest() {
  // LOOK-1.2 Feel free to write additional tests here.

  // test unstable sort
  int *dev_intKeys;
  int *dev_intValues;
  int N = 10;

  std::unique_ptr<int[]>intKeys{ new int[N] };
  std::unique_ptr<int[]>intValues{ new int[N] };

  intKeys[0] = 0; intValues[0] = 0;
  intKeys[1] = 1; intValues[1] = 1;
  intKeys[2] = 0; intValues[2] = 2;
  intKeys[3] = 3; intValues[3] = 3;
  intKeys[4] = 0; intValues[4] = 4;
  intKeys[5] = 2; intValues[5] = 5;
  intKeys[6] = 2; intValues[6] = 6;
  intKeys[7] = 0; intValues[7] = 7;
  intKeys[8] = 5; intValues[8] = 8;
  intKeys[9] = 6; intValues[9] = 9;

  cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void**)&dev_intValues, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  std::cout << "before unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // How to copy data to the GPU
  cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  thrust::device_ptr<int> dev_thrust_values(dev_intValues);
  // LOOK-2.1 Example for using thrust::sort_by_key
  thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  // How to copy data back to the CPU side from the GPU
  cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("memcpy back failed!");

  std::cout << "after unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // cleanup
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
  checkCUDAErrorWithLine("cudaFree failed!");
  return;
}
