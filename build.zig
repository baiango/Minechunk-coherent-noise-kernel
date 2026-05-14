const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const xcode_developer_dir = b.option(
        []const u8,
        "xcode-developer-dir",
        "Developer directory used by xctrace",
    ) orelse "/Applications/Xcode.app/Contents/Developer";
    const fastnoise2_dir = b.option(
        []const u8,
        "fastnoise2-dir",
        "Path to the FastNoise2 checkout",
    ) orelse "FastNoise2-master";
    const cmake_exe = b.option(
        []const u8,
        "cmake",
        "CMake executable used to configure FastNoise2",
    ) orelse "cmake";
    const fastnoise2_inputs_exist = fastNoise2InputsExist(b, fastnoise2_dir);
    const enable_fastnoise2 = b.option(
        bool,
        "enable-fastnoise2",
        "Compile the FastNoise2 simplex benchmark row",
    ) orelse true;
    const setup_fastnoise2 = b.option(
        bool,
        "setup-fastnoise2",
        "Run CMake configure for FastNoise2 before building the benchmark",
    ) orelse (enable_fastnoise2 and !fastnoise2_inputs_exist);

    const setup_fastnoise2_cmd = addFastNoise2SetupCommand(b, cmake_exe, fastnoise2_dir);
    const setup_fastnoise2_step = b.step("setup-fastnoise2", "Fetch FastSIMD and generate FastNoise2 dispatch files with CMake");
    setup_fastnoise2_step.dependOn(&setup_fastnoise2_cmd.step);

    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .stack_protector = false,
    });
    addCellularAutomataOptions(b, exe_mod, true, true);
    addCellularAutomataAssembly(b, exe_mod, target);

    const exe = b.addExecutable(.{
        .name = "Minechunk-coherent-noise-kernel",
        .root_module = exe_mod,
    });

    b.installArtifact(exe);

    const run_step = b.step("run", "Run the app");
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    run_step.dependOn(&run_cmd.step);

    const exe_tests = b.addTest(.{
        .root_module = exe_mod,
    });
    const run_exe_tests = b.addRunArtifact(exe_tests);

    const cellular_automata_tests_mod = b.createModule(.{
        .root_source_file = b.path("src/cellular_automata.zig"),
        .target = target,
        .optimize = optimize,
        .stack_protector = false,
    });
    addCellularAutomataOptions(b, cellular_automata_tests_mod, true, true);
    addCellularAutomataAssembly(b, cellular_automata_tests_mod, target);

    const cellular_automata_tests = b.addTest(.{
        .root_module = cellular_automata_tests_mod,
    });
    const run_cellular_automata_tests = b.addRunArtifact(cellular_automata_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_exe_tests.step);
    test_step.dependOn(&run_cellular_automata_tests.step);

    const ca_128_5_bench_mod = b.createModule(.{
        .root_source_file = b.path("bench/cellular_automata_chunk_128_5.zig"),
        .target = target,
        .optimize = .ReleaseFast,
        .stack_protector = false,
    });
    const ca_mod = b.createModule(.{
        .root_source_file = b.path("src/cellular_automata.zig"),
        .target = target,
        .optimize = .ReleaseFast,
        .stack_protector = false,
    });
    addCellularAutomataOptions(b, ca_mod, true, true);
    const ca_sme1_files = b.addWriteFiles();
    const ca_sme1_root = ca_sme1_files.addCopyFile(
        b.path("src/cellular_automata.zig"),
        "cellular_automata_sme1.zig",
    );
    const ca_sme1_mod = b.createModule(.{
        .root_source_file = ca_sme1_root,
        .target = target,
        .optimize = .ReleaseFast,
        .stack_protector = false,
    });
    addCellularAutomataOptions(b, ca_sme1_mod, true, false);
    const ca_no_sme_files = b.addWriteFiles();
    const ca_no_sme_root = ca_no_sme_files.addCopyFile(
        b.path("src/cellular_automata.zig"),
        "cellular_automata_no_sme.zig",
    );
    const ca_no_sme_mod = b.createModule(.{
        .root_source_file = ca_no_sme_root,
        .target = target,
        .optimize = .ReleaseFast,
        .stack_protector = false,
    });
    addCellularAutomataOptions(b, ca_no_sme_mod, false, false);
    const simplex_mod = b.createModule(.{
        .root_source_file = b.path("src/simplex.zig"),
        .target = target,
        .optimize = .ReleaseFast,
        .stack_protector = false,
    });
    const bench_options = b.addOptions();
    bench_options.addOption(bool, "fastnoise2_enabled", enable_fastnoise2);
    ca_128_5_bench_mod.addOptions("build_options", bench_options);
    ca_128_5_bench_mod.addImport("cellular_automata", ca_mod);
    ca_128_5_bench_mod.addImport("cellular_automata_sme1", ca_sme1_mod);
    ca_128_5_bench_mod.addImport("cellular_automata_no_sme", ca_no_sme_mod);
    ca_128_5_bench_mod.addImport("simplex", simplex_mod);
    addCellularAutomataAssembly(b, ca_128_5_bench_mod, target);
    if (enable_fastnoise2) {
        if (!fastNoise2SourceInputsExist(b, fastnoise2_dir)) {
            @panic("FastNoise2 support is enabled, but FastNoise2 source files were not found under the FastNoise2 checkout");
        }
        const fastnoise2_paths = findFastNoise2DependencyPaths(b, fastnoise2_dir, setup_fastnoise2) orelse
            @panic("FastNoise2 support is enabled, but FastSIMD/generated files were not found. Run `zig build setup-fastnoise2`, or disable it with `zig build bench-ca-128-5 -Denable-fastnoise2=false`.");
        addFastNoise2SimplexBenchmark(
            b,
            ca_128_5_bench_mod,
            fastnoise2_dir,
            fastnoise2_paths,
        );
    }

    const ca_128_5_bench = b.addExecutable(.{
        .name = "bench_cellular_automata_chunk_128_5",
        .root_module = ca_128_5_bench_mod,
    });
    if (enable_fastnoise2 and setup_fastnoise2) {
        ca_128_5_bench.step.dependOn(&setup_fastnoise2_cmd.step);
    }

    const run_ca_128_5_bench = b.addRunArtifact(ca_128_5_bench);
    const ca_128_5_bench_step = b.step("bench-ca-128-5", "Benchmark 16^3, 64^3, and 128^3 chunks with 5 smoothing passes");
    ca_128_5_bench_step.dependOn(&run_ca_128_5_bench.step);

    const f16_corner_bench_mod = b.createModule(.{
        .root_source_file = b.path("bench_f16x8_simplex_corner.zig"),
        .target = target,
        .optimize = .ReleaseFast,
        .stack_protector = false,
    });

    const f16_corner_bench = b.addExecutable(.{
        .name = "bench_f16x8_simplex_corner",
        .root_module = f16_corner_bench_mod,
    });

    const run_f16_corner_bench = b.addRunArtifact(f16_corner_bench);
    const f16_corner_bench_step = b.step("bench-f16-corner", "Benchmark simplexCornerSample3d8F16Preconverted");
    f16_corner_bench_step.dependOn(&run_f16_corner_bench.step);

    const trace_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = .ReleaseFast,
        .stack_protector = false,
    });
    addCellularAutomataOptions(b, trace_mod, true, true);
    addCellularAutomataAssembly(b, trace_mod, target);

    const trace_exe = b.addExecutable(.{
        .name = "three_d_value_noise_trace",
        .root_module = trace_mod,
    });

    const trace_cmd = b.addSystemCommand(&.{
        "/usr/bin/xctrace",
        "record",
        "--template",
        "Time Profiler",
        "--append-run",
        "--output",
        "zig-out/noise.trace",
    });
    trace_cmd.setEnvironmentVariable("DEVELOPER_DIR", xcode_developer_dir);
    trace_cmd.stdio = .inherit;

    trace_cmd.addArgs(&.{ "--launch", "--" });
    trace_cmd.addArtifactArg(trace_exe);
    trace_cmd.addFileInput(b.path("src/simplex.zig"));

    const trace_step = b.step("trace-noise", "Record src/simplex.zig workload with xctrace Time Profiler");
    trace_step.dependOn(&trace_cmd.step);
}

fn addCellularAutomataAssembly(b: *std.Build, module: *std.Build.Module, target: std.Build.ResolvedTarget) void {
    if (target.result.cpu.arch == .aarch64) {
        module.addAssemblyFile(b.path("src/cellular_automata_aarch64.S"));
    }
}

fn addCellularAutomataOptions(
    b: *std.Build,
    module: *std.Build.Module,
    enable_trusted_sme: bool,
    enable_trusted_sme2: bool,
) void {
    const options = b.addOptions();
    options.addOption(bool, "enable_trusted_sme", enable_trusted_sme);
    options.addOption(bool, "enable_trusted_sme2", enable_trusted_sme2);
    module.addOptions("cellular_automata_options", options);
}

fn addFastNoise2SimplexBenchmark(
    b: *std.Build,
    module: *std.Build.Module,
    fastnoise2_dir: []const u8,
    fastnoise2_paths: FastNoise2DependencyPaths,
) void {
    const cpp_flags = &.{
        "-std=c++17",
        "-DFASTNOISE_STATIC_LIB",
        "-DFASTSIMD_STATIC_LIB",
        "-ffast-math",
        "-fno-stack-protector",
        "-Wno-nan-infinity-disabled",
    };

    module.linkSystemLibrary("c++", .{});
    module.addIncludePath(cwdPath(b.pathJoin(&.{ fastnoise2_dir, "include" })));
    module.addIncludePath(cwdPath(b.pathJoin(&.{ fastnoise2_dir, "src" })));
    module.addIncludePath(cwdPath(b.pathJoin(&.{ fastnoise2_paths.fastsimd_dir, "include" })));
    module.addIncludePath(cwdPath(b.pathJoin(&.{ fastnoise2_paths.fastsimd_dir, "dispatch/impl" })));
    module.addIncludePath(cwdPath(b.pathJoin(&.{ fastnoise2_paths.generated_dir, "include" })));

    module.addCSourceFiles(.{
        .root = b.path("bench"),
        .files = &.{
            "fastnoise2_simplex_bridge.cpp",
            "fastnoise2_dispatch_scalar.cpp",
            "fastnoise2_dispatch_neon.cpp",
            "fastnoise2_dispatch_aarch64.cpp",
        },
        .flags = cpp_flags,
        .language = .cpp,
    });
    module.addCSourceFiles(.{
        .root = cwdPath(fastnoise2_dir),
        .files = &.{
            "src/FastNoise/SmartNode.cpp",
            "src/FastNoise/Metadata.cpp",
        },
        .flags = cpp_flags,
        .language = .cpp,
    });
    module.addCSourceFiles(.{
        .root = cwdPath(fastnoise2_paths.fastsimd_dir),
        .files = &.{"src/FastSIMD.cpp"},
        .flags = cpp_flags,
        .language = .cpp,
    });
}

fn fastNoise2InputsExist(
    b: *std.Build,
    fastnoise2_dir: []const u8,
) bool {
    return fastNoise2SourceInputsExist(b, fastnoise2_dir) and
        findFastNoise2DependencyPaths(b, fastnoise2_dir, false) != null;
}

fn fastNoise2SourceInputsExist(b: *std.Build, fastnoise2_dir: []const u8) bool {
    return fileExists(b, fastnoise2_dir, "CMakeLists.txt") and
        fileExists(b, fastnoise2_dir, "include/FastNoise/FastNoise.h") and
        fileExists(b, fastnoise2_dir, "src/FastNoise/SmartNode.cpp");
}

fn addFastNoise2SetupCommand(
    b: *std.Build,
    cmake_exe: []const u8,
    fastnoise2_dir: []const u8,
) *std.Build.Step.Run {
    const setup = b.addSystemCommand(&.{
        cmake_exe,
        "-S",
        fastnoise2_dir,
        "-B",
        b.pathJoin(&.{ fastnoise2_dir, "build" }),
        "-DFASTNOISE2_TOOLS=OFF",
        "-DFASTNOISE2_TESTS=OFF",
        "-DFASTNOISE2_UTILITY=OFF",
    });
    setup.stdio = .inherit;
    return setup;
}

const FastNoise2DependencyPaths = struct {
    fastsimd_dir: []const u8,
    generated_dir: []const u8,
};

fn findFastNoise2DependencyPaths(
    b: *std.Build,
    fastnoise2_dir: []const u8,
    allow_expected_cmake_paths: bool,
) ?FastNoise2DependencyPaths {
    const fastsimd_dir = findFastSimdDir(b, fastnoise2_dir) orelse blk: {
        if (!allow_expected_cmake_paths) return null;
        break :blk b.pathJoin(&.{ fastnoise2_dir, "build/_deps/fastsimd-src" });
    };
    const generated_dir = findFastNoise2GeneratedDir(b, fastnoise2_dir) orelse blk: {
        if (!allow_expected_cmake_paths) return null;
        break :blk b.pathJoin(&.{ fastnoise2_dir, "build/src/fastsimd/FastSIMD_FastNoise" });
    };
    return .{
        .fastsimd_dir = fastsimd_dir,
        .generated_dir = generated_dir,
    };
}

fn findFastSimdDir(b: *std.Build, fastnoise2_dir: []const u8) ?[]const u8 {
    return findFastNoise2SubdirWithFile(
        b,
        fastnoise2_dir,
        &.{
            "build/_deps/fastsimd-src",
            "_deps/fastsimd-src",
            "FastSIMD",
        },
        "src/FastSIMD.cpp",
    );
}

fn findFastNoise2GeneratedDir(b: *std.Build, fastnoise2_dir: []const u8) ?[]const u8 {
    return findFastNoise2SubdirWithFile(
        b,
        fastnoise2_dir,
        &.{
            "build/src/fastsimd/FastSIMD_FastNoise",
            "src/fastsimd/FastSIMD_FastNoise",
            "fastsimd/FastSIMD_FastNoise",
        },
        "include/FastSIMD/FastSIMD_FastNoise_config.h",
    );
}

fn findFastNoise2SubdirWithFile(
    b: *std.Build,
    fastnoise2_dir: []const u8,
    subdirs: []const []const u8,
    required_file: []const u8,
) ?[]const u8 {
    for (subdirs) |subdir| {
        if (fileExists(b, fastnoise2_dir, b.pathJoin(&.{ subdir, required_file }))) {
            return b.pathJoin(&.{ fastnoise2_dir, subdir });
        }
    }
    return null;
}

fn fileExists(b: *std.Build, root: []const u8, sub_path: []const u8) bool {
    const path = b.pathJoin(&.{ root, sub_path });
    std.Io.Dir.cwd().access(b.graph.io, path, .{}) catch return false;
    return true;
}

fn cwdPath(path: []const u8) std.Build.LazyPath {
    return .{ .cwd_relative = path };
}
