# Guide: Harvest Algorithm Performance Optimization like Crops with Autonomous LLM Agents

This is a practical loop for pushing one hot function down toward the metal with autonomous LLM agents. The goal is to generate many directly applicable ideas, filter them against the actual function, benchmark candidates, verify output quality, and only keep practical wins.

## Core Loop

### 1. Generate directly applicable candidates

Prompt:

```text
Give me another unique list of 30 possible directly applicable code-level optimizations only on {function name}.
```

Paste the full LLM output into the next prompt:

```text
{LLM full output}

---

Can you check every suggestion in the list is directly applicable to {function name}?
```

### 2. Benchmark the directly applicable candidates

Prompt:

```text
Can you test the directly applicable candidates for speed and output quality from the list?
```

### 3. Patch obvious practical wins

Prompt:

```text
Patch the obvious winning practical candidates, or say no if none are worth keeping.
```

## Optional ASM Inspection

Prompt:

```text
Return the ASM of {function name} without inline to me.
```

Use this to send the assembly language code to a language model instead of the source code.

## Tips

- You can run this on multiple autonomous LLM agents and prompt them in a non-sequential manner.
- Keep hot code as state-independent functions when possible. Pure data in, data out is easier to benchmark, inspect, fuzz, and optimize.
- If there is a compute-heavy algorithm structure used as an object instead of data to pass, request an LLM to remove the method. Then, convert it into state-independent functions to prevent it from becoming an undebuggable black box.
- The fewer constraints an agent has, the faster it can solve a problem without getting stuck in local minima. To improve the chance of finding the global minimum, try to be as unspecific as possible.
- Always ask for `output quality` for every patch, not just speed.

## Project Benchmark

Put the `FastNoise2-master` folder at the repo root to include the FastNoise2 row in the benchmark. The Zig build only looks inside `FastNoise2-master/` for FastNoise2, FastSIMD, and generated FastSIMD dispatch files; it does not use a root `deps/` folder.

Run the benchmark:

```sh
zig build bench-ca-128-5
```

By default, the benchmark tries to build FastNoise2 support. If FastSIMD or generated dispatch files are missing, it runs CMake to populate FastNoise2's own build tree before compiling the benchmark.

Run without FastNoise2:

```sh
zig build bench-ca-128-5 -Denable-fastnoise2=false
```

Run just the FastNoise2 setup step:

```sh
zig build setup-fastnoise2
```

If CMake is not on `PATH`, pass its executable path:

```sh
zig build setup-fastnoise2 -Dcmake=/path/to/cmake
```

The setup step runs CMake so these paths exist under `FastNoise2-master/`:

- `FastNoise2-master/build/_deps/fastsimd-src/src/FastSIMD.cpp`
- `FastNoise2-master/build/src/fastsimd/FastSIMD_FastNoise/include/FastSIMD/FastSIMD_FastNoise_config.h`

Run tests:

```sh
zig build test
```

## Benchmark Results

Latest local run:

```sh
zig build bench-ca-128-5
```

Config:

```text
chunk_sizes=16^3,64^3,128^3 benchmark_passes=5 ca_smooth_passes=5 repeats=1000 max_scratch_bytes=5143528 max_voxel_count=2097152 cpu_has_sme=true cpu_has_sme2=true fastnoise2_enabled=true
```

These numbers are machine-specific. Rerun the benchmark before comparing optimization patches.

| Implementation | Feature | Chunk size | Function name | iteration per sample | Median seconds | Median M samples/s | Median ns/sample | Checksum |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Zig | sme2 | 16^3 | `cellular_automata.cellularAutomataChunk3dWithScratch` | 5 | 0.000038 | 534.838 | 1.870 | 15790809018075169208 |
| Zig | sme1 | 16^3 | `cellular_automata_sme1.cellularAutomataChunk3dWithScratch` | 5 | 0.000041 | 499.013 | 2.004 | 15790809018075169208 |
| Zig | no-sme | 16^3 | `cellular_automata_no_sme.cellularAutomataChunk3dWithScratch` | 5 | 0.000041 | 501.297 | 1.995 | 15790809018075169208 |
| Zig | current | 16^3 | `simplex.simplexNoiseUniformGrid3d` | 1 | 0.000058 | 350.709 | 2.851 | 5.914207 |
| FastNoise2 | AARCH64 | 16^3 | `FastNoise::Generator::GenUniformGrid3D` | 1 | 0.000078 | 262.213 | 3.814 | 11.404961 |
| Zig | sme2 | 64^3 | `cellular_automata.cellularAutomataChunk3dWithScratch` | 5 | 0.001079 | 1215.271 | 0.823 | 9044746895454651478 |
| Zig | sme1 | 64^3 | `cellular_automata_sme1.cellularAutomataChunk3dWithScratch` | 5 | 0.001078 | 1215.881 | 0.822 | 9044746895454651478 |
| Zig | no-sme | 64^3 | `cellular_automata_no_sme.cellularAutomataChunk3dWithScratch` | 5 | 0.000987 | 1328.152 | 0.753 | 9044746895454651478 |
| Zig | current | 64^3 | `simplex.simplexNoiseUniformGrid3d` | 1 | 0.004048 | 323.808 | 3.088 | 23.651272 |
| FastNoise2 | AARCH64 | 64^3 | `FastNoise::Generator::GenUniformGrid3D` | 1 | 0.005287 | 247.893 | 4.034 | 14.364635 |
| Zig | sme2 | 128^3 | `cellular_automata.cellularAutomataChunk3dWithScratch` | 5 | 0.007696 | 1362.495 | 0.734 | 14277450722080343520 |
| Zig | sme1 | 128^3 | `cellular_automata_sme1.cellularAutomataChunk3dWithScratch` | 5 | 0.007761 | 1351.029 | 0.740 | 14277450722080343520 |
| Zig | no-sme | 128^3 | `cellular_automata_no_sme.cellularAutomataChunk3dWithScratch` | 5 | 0.008454 | 1240.282 | 0.806 | 14277450722080343520 |
| Zig | current | 128^3 | `simplex.simplexNoiseUniformGrid3d` | 1 | 0.031494 | 332.945 | 3.004 | 31.880825 |
| FastNoise2 | AARCH64 | 128^3 | `FastNoise::Generator::GenUniformGrid3D` | 1 | 0.043499 | 241.057 | 4.148 | 15.233884 |
