const std = @import("std");
const Io = std.Io;
const noise = @import("simplex.zig");

comptime {
    @setFloatMode(.optimized);
}

const simplexNoise2d = noise.simplexNoise2d;
const simplexNoise3d = noise.simplexNoise3d;

fn writeIntLittle(writer: *Io.Writer, comptime T: type, value: T) Io.Writer.Error!void {
    var buf: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(T, &buf, value, .little);
    try writer.writeAll(&buf);
}

fn writeNoiseBmp(io: Io, path: []const u8, width: u32, height: u32, seed: u32, frequency: f32, z: f32) !void {
    const row_stride = std.math.divCeil(u32, width * 3, 4) catch unreachable;
    const aligned_row_stride = row_stride * 4;
    const pixel_data_size = aligned_row_stride * height;
    const file_size = 14 + 40 + pixel_data_size;

    const file = try Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);

    var file_buffer: [4096]u8 = undefined;
    var file_writer = file.writer(io, &file_buffer);
    const writer = &file_writer.interface;

    try writer.writeAll("BM");
    try writeIntLittle(writer, u32, file_size);
    try writer.writeAll(&[_]u8{0} ** 4);
    try writeIntLittle(writer, u32, 14 + 40);

    try writeIntLittle(writer, u32, 40);
    try writeIntLittle(writer, i32, @intCast(width));
    try writeIntLittle(writer, i32, @intCast(height));
    try writeIntLittle(writer, u16, 1);
    try writeIntLittle(writer, u16, 24);
    try writeIntLittle(writer, u32, 0);
    try writeIntLittle(writer, u32, pixel_data_size);
    try writeIntLittle(writer, i32, 2835);
    try writeIntLittle(writer, i32, 2835);
    try writeIntLittle(writer, u32, 0);
    try writeIntLittle(writer, u32, 0);

    const padding_len: usize = @intCast(aligned_row_stride - width * 3);
    const padding = [_]u8{0} ** 3;

    var y = height;
    while (y > 0) {
        y -= 1;

        var x: u32 = 0;
        while (x < width) : (x += 1) {
            const value = simplexNoise3d(
                seed,
                @floatFromInt(x),
                @floatFromInt(y),
                z,
                frequency,
            );
            const normalized = std.math.clamp(value * 0.5 + 0.5, 0.0, 1.0);
            const gray: u8 = @intFromFloat(@round(normalized * 255.0));

            try writer.writeAll(&.{ gray, gray, gray });
        }

        try writer.writeAll(padding[0..padding_len]);
    }

    try writer.flush();
}

fn benchmarkSimplexNoise(io: Io, seed: u32, frequency: f32, iterations: u64) struct { elapsed_ns: i96, acc: f32 } {
    var acc = @as(f32, 0.0);
    var x = @as(f32, 0.125);
    var y = @as(f32, 17.375);
    var z = @as(f32, -42.625);

    const start = Io.Clock.Timestamp.now(io, .awake);

    var index: u64 = 0;
    while (index < iterations) : (index += 1) {
        std.mem.doNotOptimizeAway(seed);
        std.mem.doNotOptimizeAway(x);
        std.mem.doNotOptimizeAway(y);
        std.mem.doNotOptimizeAway(z);
        std.mem.doNotOptimizeAway(frequency);

        const value = simplexNoise3d(seed, x, y, z, frequency);
        std.mem.doNotOptimizeAway(value);
        acc += value;

        x += 0.731_123;
        y += 0.517_931;
        z += 0.263_721;

        if (x > 4096.0) {
            x -= 4096.0;
        }
        if (y > 4096.0) {
            y -= 4096.0;
        }
        if (z > 4096.0) {
            z -= 4096.0;
        }
    }

    const elapsed = start.durationTo(Io.Clock.Timestamp.now(io, .awake));
    std.mem.doNotOptimizeAway(acc);

    return .{ .elapsed_ns = elapsed.raw.nanoseconds, .acc = acc };
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const seed: u32 = 1337;
    const frequency: f32 = 0.05;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const stdout = &stdout_file_writer.interface;

    const samples = [_]struct { x: f32, y: f32, z: f32 }{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 1.0, .y = 2.0, .z = 3.0 },
        .{ .x = 10.5, .y = -4.25, .z = 8.75 },
        .{ .x = 42.0, .y = 12.0, .z = -7.0 },
    };

    for (samples) |sample| {
        const value = simplexNoise3d(seed, sample.x, sample.y, sample.z, frequency);
        try stdout.print(
            "simplex_noise_3d({d: >6.2}, {d: >6.2}, {d: >6.2}) = {d:.6}\n",
            .{ sample.x, sample.y, sample.z, value },
        );
    }

    const bmp_path = "simplex_noise.bmp";
    try writeNoiseBmp(io, bmp_path, 512, 512, seed, frequency, 0.0);
    try stdout.print("wrote {s}\n", .{bmp_path});

    const iterations: u64 = 20_000_000;
    const result = benchmarkSimplexNoise(io, seed, frequency, iterations);
    const seconds = @as(f64, @floatFromInt(result.elapsed_ns)) / @as(f64, std.time.ns_per_s);
    const samples_per_second = @as(f64, @floatFromInt(iterations)) / seconds;
    const ns_per_sample = @as(f64, @floatFromInt(result.elapsed_ns)) / @as(f64, @floatFromInt(iterations));

    try stdout.print("profiled {d} simplex_noise_3d calls in {d:.3}s\n", .{ iterations, seconds });
    try stdout.print(
        "speed: {d:.2} million samples/sec, {d:.2} ns/sample\n",
        .{ samples_per_second / 1_000_000.0, ns_per_sample },
    );
    try stdout.print("accumulator: {d:.6}\n", .{result.acc});

    try stdout.flush();
}

test "simplex noise is finite and deterministic" {
    const cases = [_]struct { seed: u32, x: f32, y: f32, z: f32, frequency: f32 }{
        .{ .seed = 0, .x = 0.0, .y = 0.0, .z = 0.0, .frequency = 0.05 },
        .{ .seed = 1337, .x = 1.0, .y = 2.0, .z = 3.0, .frequency = 0.05 },
        .{ .seed = 42, .x = 10.5, .y = -4.25, .z = 8.75, .frequency = 0.125 },
        .{ .seed = 99, .x = -512.0, .y = 256.5, .z = -128.25, .frequency = 0.01 },
    };

    for (cases) |case| {
        const a = simplexNoise3d(case.seed, case.x, case.y, case.z, case.frequency);
        const b = simplexNoise3d(case.seed, case.x, case.y, case.z, case.frequency);

        try std.testing.expect(std.math.isFinite(a));
        try std.testing.expectEqual(@as(u32, @bitCast(a)), @as(u32, @bitCast(b)));
    }
}

test "simplex 2d noise is finite and deterministic" {
    const cases = [_]struct { seed: u32, x: f32, y: f32, frequency: f32 }{
        .{ .seed = 0, .x = 0.0, .y = 0.0, .frequency = 0.05 },
        .{ .seed = 1337, .x = 1.0, .y = 2.0, .frequency = 0.05 },
        .{ .seed = 42, .x = 10.5, .y = -4.25, .frequency = 0.125 },
        .{ .seed = 99, .x = -512.0, .y = 256.5, .frequency = 0.01 },
    };

    for (cases) |case| {
        const x = case.x * case.frequency;
        const y = case.y * case.frequency;
        const a = simplexNoise2d(case.seed, x, y);
        const b = simplexNoise2d(case.seed, x, y);

        try std.testing.expect(std.math.isFinite(a));
        try std.testing.expectEqual(@as(u32, @bitCast(a)), @as(u32, @bitCast(b)));
    }
}

test "simplex entrypoint matches rounded simplex sample baseline" {
    const seed: u32 = 1337;
    const frequency: f32 = 0.05;
    const cases = [_]struct { x: f32, y: f32, z: f32, expected: f32 }{
        .{ .x = 0.0, .y = 0.0, .z = 0.0, .expected = 0.000000 },
        .{ .x = 1.0, .y = 2.0, .z = 3.0, .expected = 0.167804 },
        .{ .x = 10.5, .y = -4.25, .z = 8.75, .expected = 0.764147 },
        .{ .x = 42.0, .y = 12.0, .z = -7.0, .expected = 0.210602 },
    };

    for (cases) |case| {
        const value = simplexNoise3d(seed, case.x, case.y, case.z, frequency);
        const rounded = @round(value * 1_000_000.0) / 1_000_000.0;
        try std.testing.expectApproxEqAbs(case.expected, rounded, 0.0000005);
    }
}

test "simplex 2d entrypoint matches rounded simplex sample baseline" {
    const seed: u32 = 1337;
    const frequency: f32 = 0.05;
    const cases = [_]struct { x: f32, y: f32, expected: f32 }{
        .{ .x = 0.0, .y = 0.0, .expected = 0.000000 },
        .{ .x = 1.0, .y = 2.0, .expected = -0.395364 },
        .{ .x = 10.5, .y = -4.25, .expected = 0.646757 },
        .{ .x = 42.0, .y = 12.0, .expected = 0.612702 },
    };

    for (cases) |case| {
        const value = simplexNoise2d(seed, case.x * frequency, case.y * frequency);
        const rounded = @round(value * 1_000_000.0) / 1_000_000.0;
        try std.testing.expectApproxEqAbs(case.expected, rounded, 0.0000005);
    }
}

test "shifted-square simplex entrypoints match rounded sample baseline" {
    const seed: u32 = 1337;
    const frequency: f32 = 0.05;

    const cases3d = [_]struct { x: f32, y: f32, z: f32, expected: f32 }{
        .{ .x = 0.0, .y = 0.0, .z = 0.0, .expected = 0.000000 },
        .{ .x = 1.0, .y = 2.0, .z = 3.0, .expected = 0.172660 },
        .{ .x = 10.5, .y = -4.25, .z = 8.75, .expected = 0.872534 },
        .{ .x = 42.0, .y = 12.0, .z = -7.0, .expected = 0.246863 },
    };

    for (cases3d) |case| {
        const value = noise.simplexNoise3dShiftedSquare(seed, case.x, case.y, case.z, frequency);
        const rounded = @round(value * 1_000_000.0) / 1_000_000.0;
        try std.testing.expectApproxEqAbs(case.expected, rounded, 0.0000005);
    }

    const cases2d = [_]struct { x: f32, y: f32, expected: f32 }{
        .{ .x = 0.0, .y = 0.0, .expected = 0.000000 },
        .{ .x = 1.0, .y = 2.0, .expected = -0.405303 },
        .{ .x = 10.5, .y = -4.25, .expected = 0.720641 },
        .{ .x = 42.0, .y = 12.0, .expected = 0.760877 },
    };

    for (cases2d) |case| {
        const value = noise.simplexNoise2dShiftedSquare(seed, case.x * frequency, case.y * frequency);
        const rounded = @round(value * 1_000_000.0) / 1_000_000.0;
        try std.testing.expectApproxEqAbs(case.expected, rounded, 0.0000005);
    }
}

test {
    _ = @import("cellular_automata.zig");
}
