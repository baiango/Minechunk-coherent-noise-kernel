const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const ca = @import("cellular_automata");
const ca_sme1 = @import("cellular_automata_sme1");
const ca_no_sme = @import("cellular_automata_no_sme");
const simplex = @import("simplex");

const Seed: u32 = 1337;
const FillPercent: f32 = 0.49;
const SmoothPasses: usize = 5;
const BenchmarkPasses: usize = 5;
const SolidThreshold: u32 = 14;
const Repeats: usize = 1000;
const SimplexFrequency: f32 = 0.05;
const SimplexGridStep: f32 = 0.05;

const MaxChunkSize: usize = 128;
const MaxVoxelCount: usize = voxelCount(MaxChunkSize);
const MaxScratchBytes: usize = maxScratchBytes(MaxChunkSize);

var bench_io: std.Io = undefined;
var output: [MaxVoxelCount]bool align(64) = undefined;
var simplex_output: [MaxVoxelCount]f32 align(64) = undefined;
var scratch: [MaxScratchBytes]u8 align(64) = undefined;

const CellularAutomataRun = struct {
    elapsed_ns: u64,
    checksum: u64,
};

const CellularAutomataResult = struct {
    median_elapsed_ns: f64,
    units: usize,
    checksum: u64,
};

const SimplexRun = struct {
    elapsed_ns: u64,
    checksum: f64,
};

const SimplexResult = struct {
    median_elapsed_ns: f64,
    units: usize,
    checksum: f64,
};

const FastNoise2 = if (build_options.fastnoise2_enabled) struct {
    const Result = extern struct {
        median_elapsed_ns: f64,
        samples: usize,
        checksum: f64,
        active_feature: [*:0]const u8,
    };

    extern fn minechunk_bench_fastnoise2_simplex_uniform_grid3d(
        output: [*]f32,
        output_len: usize,
        grid_size: usize,
        passes: usize,
        repeats: usize,
        result: *Result,
    ) u8;
} else struct {};

inline fn nowNs() i96 {
    return std.Io.Clock.Timestamp.now(bench_io, .awake).raw.nanoseconds;
}

fn voxelCount(comptime chunk_size: usize) usize {
    return chunk_size * chunk_size * chunk_size;
}

fn baseOptions(comptime Impl: anytype, comptime chunk_size: usize) Impl.GenerateOptions {
    return .{
        .chunk_size = chunk_size,
        .chunk_x = 0,
        .chunk_y = 0,
        .chunk_z = 0,
        .seed = Seed,
        .fill_percent = FillPercent,
        .iterations = SmoothPasses,
        .neighborhood_radius = 1,
        .solid_threshold = SolidThreshold,
        .boundary_is_solid = true,
    };
}

fn scratchBytes(comptime Impl: anytype, comptime chunk_size: usize) usize {
    return Impl.cellularAutomataChunk3dScratchByteCount(baseOptions(Impl, chunk_size)) catch unreachable;
}

fn maxVariantScratchBytes(comptime chunk_size: usize) usize {
    return @max(
        scratchBytes(ca, chunk_size),
        @max(scratchBytes(ca_sme1, chunk_size), scratchBytes(ca_no_sme, chunk_size)),
    );
}

fn maxScratchBytes(comptime chunk_size: usize) usize {
    return @max(
        maxVariantScratchBytes(16),
        @max(maxVariantScratchBytes(64), maxVariantScratchBytes(chunk_size)),
    );
}

fn optionsForChunk(comptime Impl: anytype, comptime chunk_size: usize, index: usize) Impl.GenerateOptions {
    var options = baseOptions(Impl, chunk_size);
    options.chunk_x = @intCast(index);
    options.chunk_y = @intCast(index / 3);
    options.chunk_z = -@as(i64, @intCast(index / 5));
    options.seed = Seed +% @as(u32, @intCast(index * 17));
    return options;
}

fn checksumOutput(values: []const bool, chunk_index: usize) u64 {
    var sum: u64 = 0;
    const stride = @max(@as(usize, 1), values.len / 256);
    var index: usize = chunk_index % stride;
    while (index < values.len) : (index += stride) {
        sum +%= if (values[index]) 0x9e37_79b9_7f4a_7c15 else 0x85eb_ca6b;
        sum +%= @as(u64, @intCast(index));
    }
    return sum;
}

noinline fn generateCellularAutomataChunk(comptime Impl: anytype, comptime chunk_size: usize, chunk_index: usize) void {
    const cell_count = voxelCount(chunk_size);
    const scratch_bytes = scratchBytes(Impl, chunk_size);
    Impl.cellularAutomataChunk3dWithScratch(
        output[0..cell_count],
        scratch[0..scratch_bytes],
        optionsForChunk(Impl, chunk_size, chunk_index),
    ) catch unreachable;
}

fn runCellularAutomataBenchmark(comptime Impl: anytype, comptime chunk_size: usize) CellularAutomataRun {
    const cell_count = voxelCount(chunk_size);
    var checksum: u64 = 0;
    const start = nowNs();

    var chunk_index: usize = 0;
    while (chunk_index < BenchmarkPasses) : (chunk_index += 1) {
        generateCellularAutomataChunk(Impl, chunk_size, chunk_index);
        checksum +%= checksumOutput(output[0..cell_count], chunk_index);
    }

    const elapsed: u64 = @intCast(nowNs() - start);
    std.mem.doNotOptimizeAway(checksum);
    return .{
        .elapsed_ns = elapsed,
        .checksum = checksum,
    };
}

noinline fn fillSimplexGrid(comptime chunk_size: usize, pass: usize, seed: u32) void {
    const cell_count = voxelCount(chunk_size);
    const offset: f32 = @as(f32, @floatFromInt(pass)) * 0.125;
    simplex.simplexNoiseUniformGrid3d(
        simplex_output[0..cell_count],
        offset,
        -offset * 0.5,
        offset * 0.25,
        chunk_size,
        chunk_size,
        chunk_size,
        SimplexGridStep,
        SimplexGridStep,
        SimplexGridStep,
        seed,
        SimplexFrequency,
    );
}

fn checksumFloat(values: []const f32) f64 {
    var sum: f64 = 0.0;
    const stride = @max(@as(usize, 1), values.len / 64);
    var index: usize = 0;
    while (index < values.len) : (index += stride) {
        sum += values[index];
    }
    return sum;
}

fn runSimplexBenchmark(comptime chunk_size: usize) SimplexRun {
    const cell_count = voxelCount(chunk_size);
    var seed: u32 = Seed;
    const start = nowNs();

    var pass: usize = 0;
    while (pass < BenchmarkPasses) : (pass += 1) {
        fillSimplexGrid(chunk_size, pass, seed);
        seed +%= 1;
    }

    const elapsed: u64 = @intCast(nowNs() - start);
    const sum = checksumFloat(simplex_output[0..cell_count]);
    std.mem.doNotOptimizeAway(sum);
    return .{
        .elapsed_ns = elapsed,
        .checksum = sum,
    };
}

fn medianOfCellularAutomata(comptime Impl: anytype, comptime chunk_size: usize) CellularAutomataResult {
    var timings: [Repeats]u64 = undefined;
    var checksum: u64 = 0;
    var repeat: usize = 0;
    while (repeat < Repeats) : (repeat += 1) {
        const result = runCellularAutomataBenchmark(Impl, chunk_size);
        timings[repeat] = result.elapsed_ns;
        checksum = result.checksum;
    }
    return .{
        .median_elapsed_ns = medianElapsedNs(timings[0..]),
        .units = voxelCount(chunk_size) * BenchmarkPasses,
        .checksum = checksum,
    };
}

fn medianOfSimplex(comptime chunk_size: usize) SimplexResult {
    var timings: [Repeats]u64 = undefined;
    var checksum: f64 = 0.0;
    var repeat: usize = 0;
    while (repeat < Repeats) : (repeat += 1) {
        const result = runSimplexBenchmark(chunk_size);
        timings[repeat] = result.elapsed_ns;
        checksum = result.checksum;
    }
    return .{
        .median_elapsed_ns = medianElapsedNs(timings[0..]),
        .units = voxelCount(chunk_size) * BenchmarkPasses,
        .checksum = checksum,
    };
}

fn medianElapsedNs(timings: []u64) f64 {
    std.debug.assert(timings.len > 0);
    std.mem.sort(u64, timings, {}, std.sort.asc(u64));
    const midpoint = timings.len / 2;
    if (timings.len % 2 == 1) {
        return @floatFromInt(timings[midpoint]);
    }
    return (@as(f64, @floatFromInt(timings[midpoint - 1])) + @as(f64, @floatFromInt(timings[midpoint]))) / 2.0;
}

fn printMarkdownHeader() void {
    std.debug.print("| Implementation | Feature | Chunk size | Function name | Benchmark passes | Median seconds | Median M samples/s | Median ns/sample | Checksum |\n", .{});
    std.debug.print("|---|---|---:|---|---:|---:|---:|---:|---:|\n", .{});
}

fn printU64ChecksumTableRow(
    implementation: []const u8,
    feature: []const u8,
    comptime chunk_size: usize,
    name: []const u8,
    passes: usize,
    units: usize,
    median_elapsed_ns: f64,
    checksum: u64,
) void {
    const median_seconds = median_elapsed_ns / @as(f64, std.time.ns_per_s);
    const median_million_units_per_sec = @as(f64, @floatFromInt(units)) / median_seconds / 1_000_000.0;
    const median_ns_per_unit = median_elapsed_ns / @as(f64, @floatFromInt(units));

    std.debug.print(
        "| {s} | {s} | {d}^3 | `{s}` | {d} | {d:.6} | {d:.3} | {d:.3} | {d} |\n",
        .{
            implementation,
            feature,
            chunk_size,
            name,
            passes,
            median_seconds,
            median_million_units_per_sec,
            median_ns_per_unit,
            checksum,
        },
    );
}

fn printF64ChecksumTableRow(
    implementation: []const u8,
    feature: []const u8,
    comptime chunk_size: usize,
    name: []const u8,
    passes: usize,
    units: usize,
    median_elapsed_ns: f64,
    checksum: f64,
) void {
    if (median_elapsed_ns == 0.0 or units == 0) {
        std.debug.print(
            "| {s} | {s} | {d}^3 | `{s}` | {d} | 0.000000 | 0.000 | 0.000 | {d:.6} |\n",
            .{
                implementation,
                feature,
                chunk_size,
                name,
                passes,
                checksum,
            },
        );
        return;
    }

    const median_seconds = median_elapsed_ns / @as(f64, std.time.ns_per_s);
    const median_million_units_per_sec = @as(f64, @floatFromInt(units)) / median_seconds / 1_000_000.0;
    const median_ns_per_unit = median_elapsed_ns / @as(f64, @floatFromInt(units));

    std.debug.print(
        "| {s} | {s} | {d}^3 | `{s}` | {d} | {d:.6} | {d:.3} | {d:.3} | {d:.6} |\n",
        .{
            implementation,
            feature,
            chunk_size,
            name,
            passes,
            median_seconds,
            median_million_units_per_sec,
            median_ns_per_unit,
            checksum,
        },
    );
}

fn benchmarkChunkSize(comptime chunk_size: usize, cpu_has_sme: bool, cpu_has_sme2: bool) void {
    const ca_feature = if (cpu_has_sme2) "sme2" else if (cpu_has_sme) "sme" else "sme-unavailable";
    const ca_result = medianOfCellularAutomata(ca, chunk_size);
    printU64ChecksumTableRow(
        "Zig",
        ca_feature,
        chunk_size,
        "cellular_automata.cellularAutomataChunk3dWithScratch",
        BenchmarkPasses,
        ca_result.units,
        ca_result.median_elapsed_ns,
        ca_result.checksum,
    );

    const ca_sme1_feature = if (cpu_has_sme) "sme1" else "sme1-unavailable";
    const ca_sme1_result = medianOfCellularAutomata(ca_sme1, chunk_size);
    printU64ChecksumTableRow(
        "Zig",
        ca_sme1_feature,
        chunk_size,
        "cellular_automata_sme1.cellularAutomataChunk3dWithScratch",
        BenchmarkPasses,
        ca_sme1_result.units,
        ca_sme1_result.median_elapsed_ns,
        ca_sme1_result.checksum,
    );

    const ca_no_sme_result = medianOfCellularAutomata(ca_no_sme, chunk_size);
    printU64ChecksumTableRow(
        "Zig",
        "no-sme",
        chunk_size,
        "cellular_automata_no_sme.cellularAutomataChunk3dWithScratch",
        BenchmarkPasses,
        ca_no_sme_result.units,
        ca_no_sme_result.median_elapsed_ns,
        ca_no_sme_result.checksum,
    );

    const simplex_result = medianOfSimplex(chunk_size);
    printF64ChecksumTableRow(
        "Zig",
        "current",
        chunk_size,
        "simplex.simplexNoiseUniformGrid3d",
        BenchmarkPasses,
        simplex_result.units,
        simplex_result.median_elapsed_ns,
        simplex_result.checksum,
    );

    if (comptime build_options.fastnoise2_enabled) {
        var fastnoise2_result: FastNoise2.Result = undefined;
        const sample_count = voxelCount(chunk_size);
        const ok = FastNoise2.minechunk_bench_fastnoise2_simplex_uniform_grid3d(
            simplex_output[0..sample_count].ptr,
            sample_count,
            chunk_size,
            BenchmarkPasses,
            Repeats,
            &fastnoise2_result,
        ) != 0;
        if (ok) {
            printF64ChecksumTableRow(
                "FastNoise2",
                std.mem.span(fastnoise2_result.active_feature),
                chunk_size,
                "FastNoise::Generator::GenUniformGrid3D",
                BenchmarkPasses,
                fastnoise2_result.samples,
                fastnoise2_result.median_elapsed_ns,
                fastnoise2_result.checksum,
            );
        } else {
            printF64ChecksumTableRow(
                "FastNoise2",
                "unavailable",
                chunk_size,
                "FastNoise::Generator::GenUniformGrid3D",
                BenchmarkPasses,
                0,
                0.0,
                0.0,
            );
        }
    }
}

pub fn main(init: std.process.Init) void {
    bench_io = init.io;

    const cpu_has_sme = builtin.cpu.arch == .aarch64 and
        std.Target.aarch64.featureSetHas(builtin.cpu.features, .sme);
    const cpu_has_sme2 = builtin.cpu.arch == .aarch64 and
        std.Target.aarch64.featureSetHas(builtin.cpu.features, .sme2);

    std.debug.print(
        "config: chunk_sizes=16^3,64^3,128^3 benchmark_passes={d} ca_smooth_passes={d} repeats={d} max_scratch_bytes={d} max_voxel_count={d} cpu_has_sme={any} cpu_has_sme2={any} fastnoise2_enabled={any}\n",
        .{ BenchmarkPasses, SmoothPasses, Repeats, MaxScratchBytes, MaxVoxelCount, cpu_has_sme, cpu_has_sme2, build_options.fastnoise2_enabled },
    );
    printMarkdownHeader();

    inline for (.{ 16, 64, 128 }) |chunk_size| {
        benchmarkChunkSize(chunk_size, cpu_has_sme, cpu_has_sme2);
    }
}
