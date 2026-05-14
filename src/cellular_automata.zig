const std = @import("std");
const builtin = @import("builtin");
const cellular_automata_options = @import("cellular_automata_options");

const ByteLaneCount: usize = 16;
const U8x16 = @Vector(ByteLaneCount, u8);
const U32x16 = @Vector(ByteLaneCount, u32);
const ByteLaneOffsets = U32x16{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
const MaxStackChunkScratchBytes: usize = 256 * 1024;
const SmoothTrustedSmeOutputChunkBytes: usize = 64;
const SmoothTrustedSmeMinOutputSide: usize = 64;
const SmoothTrustedSmeMinInputSide: usize = SmoothTrustedSmeMinOutputSide + 2;
const use_smooth_trusted_sme = cellular_automata_options.enable_trusted_sme and
    builtin.cpu.arch == .aarch64 and
    std.Target.aarch64.featureSetHas(builtin.cpu.features, .sme);
const use_smooth_trusted_sme2 = cellular_automata_options.enable_trusted_sme and
    cellular_automata_options.enable_trusted_sme2 and
    builtin.cpu.arch == .aarch64 and
    std.Target.aarch64.featureSetHas(builtin.cpu.features, .sme2);

extern fn cellularAutomataSmooth3dBytesShrinkToBytesTrustedAarch64(
    output: [*]u8,
    input: [*]const u8,
    input_side: usize,
    output_side: usize,
    threshold: u8,
) callconv(.c) void;

extern fn cellularAutomataSmooth3dBytesShrinkToBytesTrustedAarch64Sme2(
    output: [*]u8,
    input: [*]const u8,
    input_side: usize,
    output_side: usize,
    threshold: u8,
) callconv(.c) void;

inline fn shouldUseSmoothTrustedSme(output_side: usize) bool {
    return output_side >= SmoothTrustedSmeMinOutputSide and
        output_side % SmoothTrustedSmeOutputChunkBytes == 0;
}

pub const Error = error{
    InvalidChunkSize,
    InvalidGridSide,
    InvalidNeighborhoodRadius,
    InvalidFillPercent,
    InvalidThreshold,
    BufferTooSmall,
    CellCountOverflow,
    CoordinateOverflow,
    CoordinateOutOfBounds,
    InPlaceSmoothNotSupported,
    OutOfMemory,
};

pub const GridOrigin = struct {
    x: i64,
    y: i64,
    z: i64,
};

pub const SmoothOptions = struct {
    neighborhood_radius: usize = 1,
    solid_threshold: u32 = 14,
    boundary_is_solid: bool = true,
};

pub const GenerateOptions = struct {
    chunk_size: usize = 32,
    chunk_x: i64 = 0,
    chunk_y: i64 = 0,
    chunk_z: i64 = 0,
    seed: u32 = 0,
    fill_percent: f32 = 0.48,
    iterations: usize = 5,
    neighborhood_radius: usize = 1,
    solid_threshold: u32 = 14,
    boundary_is_solid: bool = true,
};

const MaxStackSweepCandidates: usize = 64;

pub const PreparedSmoothPlan = struct {
    side: usize,
    count: usize,
    plane: usize,
    solid_threshold: u32,
    boundary_is_solid: bool,
    tiny_indices: [8][26]usize = [_][26]usize{[_]usize{0} ** 26} ** 8,
    tiny_counts: [8]u8 = [_]u8{0} ** 8,
    tiny_outside_counts: [8]u8 = [_]u8{0} ** 8,

    pub fn init(side: usize, options: SmoothOptions) Error!PreparedSmoothPlan {
        try validateSmoothOptions(side, options);
        if (options.neighborhood_radius != 1) return error.InvalidNeighborhoodRadius;
        const count = try cellularAutomataCellCount(side);
        var plan = PreparedSmoothPlan{
            .side = side,
            .count = count,
            .plane = side * side,
            .solid_threshold = options.solid_threshold,
            .boundary_is_solid = options.boundary_is_solid,
        };
        if (side < 3) plan.buildTinyNeighborLists();
        return plan;
    }

    fn buildTinyNeighborLists(self: *PreparedSmoothPlan) void {
        const side_i: isize = @intCast(self.side);
        var z: usize = 0;
        while (z < self.side) : (z += 1) {
            var y: usize = 0;
            while (y < self.side) : (y += 1) {
                var x: usize = 0;
                while (x < self.side) : (x += 1) {
                    const cell = cellularAutomataIndex3d(x, y, z, self.side);
                    var valid_count: u8 = 0;
                    var outside_count: u8 = 0;
                    var dz: isize = -1;
                    while (dz <= 1) : (dz += 1) {
                        var dy: isize = -1;
                        while (dy <= 1) : (dy += 1) {
                            var dx: isize = -1;
                            while (dx <= 1) : (dx += 1) {
                                if (dx == 0 and dy == 0 and dz == 0) continue;
                                const nx = @as(isize, @intCast(x)) + dx;
                                const ny = @as(isize, @intCast(y)) + dy;
                                const nz = @as(isize, @intCast(z)) + dz;
                                if (nx < 0 or ny < 0 or nz < 0 or nx >= side_i or ny >= side_i or nz >= side_i) {
                                    outside_count += 1;
                                } else {
                                    self.tiny_indices[cell][valid_count] = cellularAutomataIndex3d(@intCast(nx), @intCast(ny), @intCast(nz), self.side);
                                    valid_count += 1;
                                }
                            }
                        }
                    }
                    self.tiny_counts[cell] = valid_count;
                    self.tiny_outside_counts[cell] = outside_count;
                }
            }
        }
    }

    pub fn smooth(self: *const PreparedSmoothPlan, output: []bool, input: []const bool) Error!void {
        if (input.len < self.count or output.len < self.count) return error.BufferTooSmall;
        if (@intFromPtr(output.ptr) == @intFromPtr(input.ptr)) return error.InPlaceSmoothNotSupported;
        if (self.side < 3) {
            smoothTinyPrepared(self, output[0..self.count], input[0..self.count]);
        } else {
            smoothRadius1Prepared(self, output[0..self.count], input[0..self.count]);
        }
    }
};

pub const OffsetSmoothPlan = struct {
    allocator: std.mem.Allocator,
    side: usize,
    count: usize,
    plane: usize,
    solid_threshold: u8,
    threshold_minus_one: u8,
    boundary_is_solid: bool,
    interior_bases: []usize,
    boundary_indices: []usize,
    boundary_starts: []usize,
    boundary_counts: []u8,
    boundary_outside_counts: []u8,
    boundary_offsets: []i32,

    pub fn init(allocator: std.mem.Allocator, side: usize, options: SmoothOptions) Error!OffsetSmoothPlan {
        try validateSmoothOptions(side, options);
        if (options.neighborhood_radius != 1) return error.InvalidNeighborhoodRadius;
        if (side < 3) return error.InvalidGridSide;

        const count = try cellularAutomataCellCount(side);
        const interior_side = side - 2;
        const interior_count = interior_side * interior_side * interior_side;
        const boundary_count = count - interior_count;

        const interior_bases = try allocator.alloc(usize, interior_count);
        errdefer allocator.free(interior_bases);
        const boundary_indices = try allocator.alloc(usize, boundary_count);
        errdefer allocator.free(boundary_indices);
        const boundary_starts = try allocator.alloc(usize, boundary_count);
        errdefer allocator.free(boundary_starts);
        const boundary_counts = try allocator.alloc(u8, boundary_count);
        errdefer allocator.free(boundary_counts);
        const boundary_outside_counts = try allocator.alloc(u8, boundary_count);
        errdefer allocator.free(boundary_outside_counts);
        const boundary_offsets = try allocator.alloc(i32, boundary_count * 26);
        errdefer allocator.free(boundary_offsets);

        var plan = OffsetSmoothPlan{
            .allocator = allocator,
            .side = side,
            .count = count,
            .plane = side * side,
            .solid_threshold = @intCast(options.solid_threshold),
            .threshold_minus_one = @as(u8, @intCast(options.solid_threshold)) -| 1,
            .boundary_is_solid = options.boundary_is_solid,
            .interior_bases = interior_bases,
            .boundary_indices = boundary_indices,
            .boundary_starts = boundary_starts,
            .boundary_counts = boundary_counts,
            .boundary_outside_counts = boundary_outside_counts,
            .boundary_offsets = boundary_offsets,
        };
        plan.build();
        return plan;
    }

    pub fn deinit(self: *OffsetSmoothPlan) void {
        self.allocator.free(self.interior_bases);
        self.allocator.free(self.boundary_indices);
        self.allocator.free(self.boundary_starts);
        self.allocator.free(self.boundary_counts);
        self.allocator.free(self.boundary_outside_counts);
        self.allocator.free(self.boundary_offsets);
        self.* = undefined;
    }

    fn build(self: *OffsetSmoothPlan) void {
        var interior_index: usize = 0;
        var boundary_index: usize = 0;
        var offset_index: usize = 0;
        const side_i: isize = @intCast(self.side);

        var z: usize = 0;
        while (z < self.side) : (z += 1) {
            var y: usize = 0;
            while (y < self.side) : (y += 1) {
                var x: usize = 0;
                while (x < self.side) : (x += 1) {
                    const base = cellularAutomataIndex3d(x, y, z, self.side);
                    if (x != 0 and y != 0 and z != 0 and x + 1 != self.side and y + 1 != self.side and z + 1 != self.side) {
                        self.interior_bases[interior_index] = base;
                        interior_index += 1;
                        continue;
                    }

                    self.boundary_indices[boundary_index] = base;
                    self.boundary_starts[boundary_index] = offset_index;
                    const start = offset_index;
                    var outside_count: u8 = 0;
                    var dz: isize = -1;
                    while (dz <= 1) : (dz += 1) {
                        var dy: isize = -1;
                        while (dy <= 1) : (dy += 1) {
                            var dx: isize = -1;
                            while (dx <= 1) : (dx += 1) {
                                if (dx == 0 and dy == 0 and dz == 0) continue;
                                const nx = @as(isize, @intCast(x)) + dx;
                                const ny = @as(isize, @intCast(y)) + dy;
                                const nz = @as(isize, @intCast(z)) + dz;
                                if (nx < 0 or ny < 0 or nz < 0 or nx >= side_i or ny >= side_i or nz >= side_i) {
                                    outside_count += 1;
                                } else {
                                    const neighbor = cellularAutomataIndex3d(@intCast(nx), @intCast(ny), @intCast(nz), self.side);
                                    self.boundary_offsets[offset_index] = @intCast(@as(isize, @intCast(neighbor)) - @as(isize, @intCast(base)));
                                    offset_index += 1;
                                }
                            }
                        }
                    }
                    self.boundary_counts[boundary_index] = @intCast(offset_index - start);
                    self.boundary_outside_counts[boundary_index] = outside_count;
                    boundary_index += 1;
                }
            }
        }
    }

    pub fn smooth(self: *const OffsetSmoothPlan, output: []bool, input: []const bool) Error!void {
        if (input.len < self.count or output.len < self.count) return error.BufferTooSmall;
        if (@intFromPtr(output.ptr) == @intFromPtr(input.ptr)) return error.InPlaceSmoothNotSupported;

        for (self.interior_bases) |base| {
            output[base] = meetsRadius1Threshold(
                countRadius1InteriorBool(input, base, self.side, self.plane),
                self.solid_threshold,
                self.threshold_minus_one,
            );
        }

        for (self.boundary_indices, 0..) |base, index| {
            var count: u32 = if (self.boundary_is_solid) self.boundary_outside_counts[index] else 0;
            const start = self.boundary_starts[index];
            var offset_index: usize = 0;
            while (offset_index < self.boundary_counts[index]) : (offset_index += 1) {
                const neighbor_index: usize = @intCast(@as(isize, @intCast(base)) + @as(isize, self.boundary_offsets[start + offset_index]));
                count += @intFromBool(input[neighbor_index]);
            }
            output[base] = meetsRadius1Threshold(count, self.solid_threshold, self.threshold_minus_one);
        }
    }
};

pub const CellularAutomataChunkScratch = struct {
    allocator: std.mem.Allocator,
    buffer: []u8,

    pub fn init(allocator: std.mem.Allocator, options: GenerateOptions) Error!CellularAutomataChunkScratch {
        const byte_count = try cellularAutomataChunk3dScratchByteCount(options);
        return .{
            .allocator = allocator,
            .buffer = if (byte_count == 0) &.{} else try allocator.alloc(u8, byte_count),
        };
    }

    pub fn initCapacity(allocator: std.mem.Allocator, byte_count: usize) Error!CellularAutomataChunkScratch {
        return .{
            .allocator = allocator,
            .buffer = if (byte_count == 0) &.{} else try allocator.alloc(u8, byte_count),
        };
    }

    pub fn deinit(self: *CellularAutomataChunkScratch) void {
        if (self.buffer.len != 0) self.allocator.free(self.buffer);
        self.buffer = &.{};
    }

    pub fn generateChunk3d(self: *CellularAutomataChunkScratch, out: []bool, options: GenerateOptions) Error!void {
        return cellularAutomataChunk3dWithScratch(out, self.buffer, options);
    }
};

pub inline fn cellularAutomataIndex3d(x: usize, y: usize, z: usize, side: usize) usize {
    return x + side * (y + side * z);
}

pub fn cellularAutomataRequiredPadding(iterations: usize, neighborhood_radius: usize) Error!usize {
    return std.math.mul(usize, iterations, neighborhood_radius) catch error.CellCountOverflow;
}

pub fn cellularAutomataTemporarySide(chunk_size: usize, padding: usize) Error!usize {
    const doubled_padding = std.math.mul(usize, padding, 2) catch return error.CellCountOverflow;
    return std.math.add(usize, chunk_size, doubled_padding) catch error.CellCountOverflow;
}

pub fn cellularAutomataCellCount(side: usize) Error!usize {
    if (side == 0) return error.InvalidGridSide;
    const plane_count = std.math.mul(usize, side, side) catch return error.CellCountOverflow;
    return std.math.mul(usize, plane_count, side) catch error.CellCountOverflow;
}

pub fn cellularAutomataRandomSolid(world_x: i64, world_y: i64, world_z: i64, seed: u32, fill_percent: f32) bool {
    if (fill_percent <= 0.0) return false;
    if (fill_percent >= 1.0) return true;
    return cellularAutomataHash3(world_x, world_y, world_z, seed) < cellularAutomataFillThreshold(fill_percent);
}

pub fn cellularAutomataHashToUnitFloat(hash: u32) f32 {
    return @as(f32, @floatFromInt(hash)) / @as(f32, @floatFromInt(std.math.maxInt(u32)));
}

pub fn cellularAutomataHash3(x: i64, y: i64, z: i64, seed: u32) u32 {
    return cellularAutomataHash3U32(coordBits32(x), coordBits32(y), coordBits32(z), seed);
}

fn cellularAutomataFillThreshold(fill_percent: f32) u32 {
    if (fill_percent <= 0.0) return 0;
    if (fill_percent >= 1.0) return std.math.maxInt(u32);

    const scaled = @as(f64, fill_percent) * 4_294_967_296.0;
    if (scaled >= @as(f64, @floatFromInt(std.math.maxInt(u32)))) return std.math.maxInt(u32);
    return @intFromFloat(@ceil(scaled));
}

inline fn cellularAutomataHash3U32(x: u32, y: u32, z: u32, seed: u32) u32 {
    var h = seed;
    h ^= x *% 374_761_393;
    h ^= y *% 668_265_263;
    h ^= z *% 2_246_822_519;
    h = (h ^ (h >> 13)) *% 1_274_126_177;
    return h ^ (h >> 16);
}

inline fn cellularAutomataHash3x16(x: U32x16, y: U32x16, z: U32x16, seed: u32) U32x16 {
    var h = @as(U32x16, @splat(seed));
    h ^= x *% @as(U32x16, @splat(@as(u32, 374_761_393)));
    h ^= y *% @as(U32x16, @splat(@as(u32, 668_265_263)));
    h ^= z *% @as(U32x16, @splat(@as(u32, 2_246_822_519)));
    h = (h ^ (h >> @as(U32x16, @splat(@as(u32, 13))))) *% @as(U32x16, @splat(@as(u32, 1_274_126_177)));
    return h ^ (h >> @as(U32x16, @splat(@as(u32, 16))));
}

fn coordBits32(value: i64) u32 {
    return @truncate(@as(u64, @bitCast(value)));
}

inline fn storeU8x16(pointer: [*]u8, value: U8x16) void {
    const vector_pointer: *align(1) U8x16 = @ptrCast(pointer);
    vector_pointer.* = value;
}

inline fn loadU8x16(pointer: [*]const u8) U8x16 {
    const vector_pointer: *align(1) const U8x16 = @ptrCast(pointer);
    return vector_pointer.*;
}

fn boolSliceBytes(slice: []bool) []u8 {
    comptime {
        if (@sizeOf(bool) != 1 or @alignOf(bool) != 1) {
            @compileError("cellular automata byte copy expects one-byte bool storage");
        }
    }

    return @as([*]u8, @ptrCast(slice.ptr))[0..slice.len];
}

fn boolSliceConstBytes(slice: []const bool) []const u8 {
    comptime {
        if (@sizeOf(bool) != 1 or @alignOf(bool) != 1) {
            @compileError("cellular automata byte count expects one-byte bool storage");
        }
    }

    return @as([*]const u8, @ptrCast(slice.ptr))[0..slice.len];
}

fn boolSliceFromBytes(slice: []u8) []bool {
    comptime {
        if (@sizeOf(bool) != 1 or @alignOf(bool) != 1) {
            @compileError("cellular automata bool scratch expects one-byte bool storage");
        }
    }

    return @as([*]bool, @ptrCast(slice.ptr))[0..slice.len];
}

inline fn solidRowSum3(input: [*]const u8, center: usize) U8x16 {
    const left = loadU8x16(input + center - 1);
    const center_v = loadU8x16(input + center);
    const right = loadU8x16(input + center + 1);
    return (left + center_v) + right;
}

const RowAllCore = struct {
    all: U8x16,
    core: U8x16,
};

const BandAllCore = struct {
    all: U8x16,
    core: U8x16,
};

inline fn solidRowSum3Ptr(center: [*]const u8) U8x16 {
    const left = loadU8x16(center - 1);
    const center_v = loadU8x16(center);
    const right = loadU8x16(center + 1);
    return (left + center_v) + right;
}

inline fn solidBandSum3Ptr(
    lower: [*]const u8,
    middle: [*]const u8,
    upper: [*]const u8,
) U8x16 {
    return (solidRowSum3Ptr(lower) + solidRowSum3Ptr(middle)) + solidRowSum3Ptr(upper);
}

inline fn solidRowAllCorePtr(center: [*]const u8) RowAllCore {
    const left = loadU8x16(center - 1);
    const center_v = loadU8x16(center);
    const right = loadU8x16(center + 1);
    const core = left + right;
    return .{
        .all = core + center_v,
        .core = core,
    };
}

inline fn solidBandAllCorePtr(
    lower: [*]const u8,
    middle: [*]const u8,
    upper: [*]const u8,
) BandAllCore {
    const side = solidRowSum3Ptr(lower) + solidRowSum3Ptr(upper);
    const middle_row = solidRowAllCorePtr(middle);
    return .{
        .all = side + middle_row.all,
        .core = side + middle_row.core,
    };
}

inline fn solidRowSum2WithoutCenter(input: [*]const u8, center: usize) U8x16 {
    return loadU8x16(input + center - 1) + loadU8x16(input + center + 1);
}

inline fn solidPlaneSum3(input: [*]const u8, center: usize, side: usize) U8x16 {
    return (solidRowSum3(input, center - side) + solidRowSum3(input, center)) + solidRowSum3(input, center + side);
}

inline fn smoothInteriorByteBlock(
    output: [*]u8,
    input: [*]const u8,
    base: usize,
    side: usize,
    plane: usize,
    threshold_v: U8x16,
    one_v: U8x16,
    zero_v: U8x16,
) void {
    const lower_plane = solidPlaneSum3(input, base - plane, side);
    const upper_plane = solidPlaneSum3(input, base + plane, side);
    const middle_plane =
        (solidRowSum3(input, base - side) + solidRowSum2WithoutCenter(input, base)) +
        solidRowSum3(input, base + side);
    const solid_neighbors = (lower_plane + middle_plane) + upper_plane;

    storeU8x16(output + base, @select(u8, solid_neighbors >= threshold_v, one_v, zero_v));
}

inline fn smoothInteriorByteColumnToBytes(
    output: [*]u8,
    lower_first: [*]const u8,
    middle_first: [*]const u8,
    upper_first: [*]const u8,
    src_stride: usize,
    dst_stride: usize,
    count: usize,
    threshold: u8,
) void {
    if (count == 0) return;

    const threshold_v = @as(U8x16, @splat(threshold));
    const zero_v = @as(U8x16, @splat(@as(u8, 0)));
    const prev_full = solidBandSum3Ptr(lower_first - src_stride, middle_first - src_stride, upper_first - src_stride);
    const band_curr_pair = solidBandAllCorePtr(lower_first, middle_first, upper_first);
    var cur_full = band_curr_pair.all;
    var seed = prev_full + band_curr_pair.core;
    const band_next_pair = solidBandAllCorePtr(lower_first + src_stride, middle_first + src_stride, upper_first + src_stride);
    const true_v = @as(U8x16, @splat(@as(u8, 0xff)));

    var output_base = seed + band_next_pair.all;
    var mask = @select(u8, output_base >= threshold_v, true_v, zero_v);
    storeU8x16(output, mask >> @as(U8x16, @splat(@as(u3, 7))));
    if (count == 1) return;

    seed = cur_full + band_next_pair.core;
    cur_full = band_next_pair.all;
    var output_ptr = output + dst_stride;
    var remaining = count - 1;
    var incoming_lower = lower_first + src_stride * 2;
    var incoming_middle = middle_first + src_stride * 2;
    var incoming_upper = upper_first + src_stride * 2;

    while (remaining != 0) {
        const band = solidBandAllCorePtr(incoming_lower, incoming_middle, incoming_upper);
        output_base = seed + band.all;
        mask = @select(u8, output_base >= threshold_v, true_v, zero_v);
        storeU8x16(output_ptr, mask >> @as(U8x16, @splat(@as(u3, 7))));

        seed = cur_full + band.core;
        cur_full = band.all;
        output_ptr += dst_stride;
        incoming_lower += src_stride;
        incoming_middle += src_stride;
        incoming_upper += src_stride;
        remaining -= 1;
    }
}

pub fn cellularAutomataRandomSolidGrid3d(
    out: []bool,
    side: usize,
    origin: GridOrigin,
    seed: u32,
    fill_percent: f32,
) Error!void {
    try validateFillPercent(fill_percent);
    if (side == 0) return error.InvalidGridSide;
    const count = try cellularAutomataCellCount(side);
    if (out.len < count) return error.BufferTooSmall;

    if (fill_percent <= 0.0) {
        @memset(out[0..count], false);
        return;
    }
    if (fill_percent >= 1.0) {
        @memset(out[0..count], true);
        return;
    }

    const threshold = cellularAutomataFillThreshold(fill_percent);
    const max_offset = side - 1;
    _ = try addOffset(origin.x, max_offset);
    _ = try addOffset(origin.y, max_offset);
    _ = try addOffset(origin.z, max_offset);
    const origin_x_bits = coordBits32(origin.x);
    const origin_y_bits = coordBits32(origin.y);
    const origin_z_bits = coordBits32(origin.z);

    var z: usize = 0;
    while (z < side) : (z += 1) {
        const z_bits = origin_z_bits +% @as(u32, @truncate(z));
        var y: usize = 0;
        while (y < side) : (y += 1) {
            const y_bits = origin_y_bits +% @as(u32, @truncate(y));
            const row_start = side * (y + side * z);
            var x: usize = 0;
            while (x < side) : (x += 1) {
                const x_bits = origin_x_bits +% @as(u32, @truncate(x));
                out[row_start + x] = cellularAutomataHash3U32(x_bits, y_bits, z_bits, seed) < threshold;
            }
        }
    }
}

pub fn cellularAutomataRandomSolidGrid3dFillSweep(
    out: []bool,
    side: usize,
    origin: GridOrigin,
    seed: u32,
    fill_percents: []const f32,
) Error!void {
    if (side == 0) return error.InvalidGridSide;
    const count = try cellularAutomataCellCount(side);
    const output_count = std.math.mul(usize, count, fill_percents.len) catch return error.CellCountOverflow;
    if (out.len < output_count) return error.BufferTooSmall;

    for (fill_percents) |fill_percent| {
        try validateFillPercent(fill_percent);
    }
    if (fill_percents.len == 0) return;

    if (fill_percents.len > MaxStackSweepCandidates) {
        for (fill_percents, 0..) |fill_percent, candidate| {
            try cellularAutomataRandomSolidGrid3d(
                out[candidate * count ..][0..count],
                side,
                origin,
                seed,
                fill_percent,
            );
        }
        return;
    }

    var modes: [MaxStackSweepCandidates]u8 = undefined;
    var thresholds: [MaxStackSweepCandidates]u32 = undefined;
    for (fill_percents, 0..) |fill_percent, candidate| {
        if (fill_percent <= 0.0) {
            modes[candidate] = 0;
            thresholds[candidate] = 0;
        } else if (fill_percent >= 1.0) {
            modes[candidate] = 1;
            thresholds[candidate] = std.math.maxInt(u32);
        } else {
            modes[candidate] = 2;
            thresholds[candidate] = cellularAutomataFillThreshold(fill_percent);
        }
    }

    const max_offset = side - 1;
    _ = try addOffset(origin.x, max_offset);
    _ = try addOffset(origin.y, max_offset);
    _ = try addOffset(origin.z, max_offset);
    const origin_x_bits = coordBits32(origin.x);
    const origin_y_bits = coordBits32(origin.y);
    const origin_z_bits = coordBits32(origin.z);

    var z: usize = 0;
    while (z < side) : (z += 1) {
        const z_bits = origin_z_bits +% @as(u32, @truncate(z));
        var y: usize = 0;
        while (y < side) : (y += 1) {
            const y_bits = origin_y_bits +% @as(u32, @truncate(y));
            const row_start = side * (y + side * z);
            var x: usize = 0;
            while (x < side) : (x += 1) {
                const x_bits = origin_x_bits +% @as(u32, @truncate(x));
                const hash = cellularAutomataHash3U32(x_bits, y_bits, z_bits, seed);
                const index = row_start + x;
                for (0..fill_percents.len) |candidate| {
                    out[candidate * count + index] = switch (modes[candidate]) {
                        0 => false,
                        1 => true,
                        else => hash < thresholds[candidate],
                    };
                }
            }
        }
    }
}

fn cellularAutomataRandomSolidGrid3dBytes(
    out: []u8,
    side: usize,
    origin: GridOrigin,
    seed: u32,
    fill_percent: f32,
) Error!void {
    try validateFillPercent(fill_percent);
    if (side == 0) return error.InvalidGridSide;
    const count = try cellularAutomataCellCount(side);
    if (out.len < count) return error.BufferTooSmall;

    if (fill_percent <= 0.0) {
        @memset(out[0..count], 0);
        return;
    }
    if (fill_percent >= 1.0) {
        @memset(out[0..count], 1);
        return;
    }

    const threshold = cellularAutomataFillThreshold(fill_percent);
    const threshold_v = @as(U32x16, @splat(threshold));
    const one_v = @as(U8x16, @splat(@as(u8, 1)));
    const zero_v = @as(U8x16, @splat(@as(u8, 0)));
    const max_offset = side - 1;
    _ = try addOffset(origin.x, max_offset);
    _ = try addOffset(origin.y, max_offset);
    _ = try addOffset(origin.z, max_offset);
    const origin_x_bits = coordBits32(origin.x);
    const origin_y_bits = coordBits32(origin.y);
    const origin_z_bits = coordBits32(origin.z);

    var z: usize = 0;
    while (z < side) : (z += 1) {
        const z_bits = origin_z_bits +% @as(u32, @truncate(z));
        const z_v = @as(U32x16, @splat(z_bits));
        var y: usize = 0;
        while (y < side) : (y += 1) {
            const y_bits = origin_y_bits +% @as(u32, @truncate(y));
            const y_v = @as(U32x16, @splat(y_bits));
            const row_start = side * (y + side * z);

            var x: usize = 0;
            var x_v = @as(U32x16, @splat(origin_x_bits)) +% ByteLaneOffsets;
            const x_step_v = @as(U32x16, @splat(@as(u32, ByteLaneCount)));
            while (x + ByteLaneCount <= side) : ({
                x += ByteLaneCount;
                x_v +%= x_step_v;
            }) {
                const hash = cellularAutomataHash3x16(x_v, y_v, z_v, seed);
                const solid = @select(u8, hash < threshold_v, one_v, zero_v);
                storeU8x16(out.ptr + row_start + x, solid);
            }
            while (x < side) : (x += 1) {
                const x_bits = origin_x_bits +% @as(u32, @truncate(x));
                out[row_start + x] = if (cellularAutomataHash3U32(x_bits, y_bits, z_bits, seed) < threshold) 1 else 0;
            }
        }
    }
}

fn cellularAutomataRandomSolidGrid3dBytesThresholdTrusted(
    out: []u8,
    side: usize,
    origin_x_bits: u32,
    origin_y_bits: u32,
    origin_z_bits: u32,
    seed: u32,
    fill_threshold: u32,
) void {
    const count = side * side * side;
    std.debug.assert(out.len >= count);

    if (fill_threshold == 0) {
        @memset(out[0..count], 0);
        return;
    }
    if (fill_threshold == std.math.maxInt(u32)) {
        @memset(out[0..count], 1);
        return;
    }

    const threshold_v = @as(U32x16, @splat(fill_threshold));
    const one_v = @as(U8x16, @splat(@as(u8, 1)));
    const zero_v = @as(U8x16, @splat(@as(u8, 0)));

    var z: usize = 0;
    while (z < side) : (z += 1) {
        const z_bits = origin_z_bits +% @as(u32, @truncate(z));
        const z_v = @as(U32x16, @splat(z_bits));
        var y: usize = 0;
        while (y < side) : (y += 1) {
            const y_bits = origin_y_bits +% @as(u32, @truncate(y));
            const y_v = @as(U32x16, @splat(y_bits));
            const row_start = side * (y + side * z);

            var x: usize = 0;
            var x_v = @as(U32x16, @splat(origin_x_bits)) +% ByteLaneOffsets;
            const x_step_v = @as(U32x16, @splat(@as(u32, ByteLaneCount)));
            while (x + ByteLaneCount <= side) : ({
                x += ByteLaneCount;
                x_v +%= x_step_v;
            }) {
                const hash = cellularAutomataHash3x16(x_v, y_v, z_v, seed);
                storeU8x16(out.ptr + row_start + x, @select(u8, hash < threshold_v, one_v, zero_v));
            }
            while (x < side) : (x += 1) {
                const x_bits = origin_x_bits +% @as(u32, @truncate(x));
                out[row_start + x] = if (cellularAutomataHash3U32(x_bits, y_bits, z_bits, seed) < fill_threshold) 1 else 0;
            }
        }
    }
}

fn cellularAutomataRandomSolidGrid3dBytesThresholdTrustedNoWrapXYZ(
    out: []u8,
    side: usize,
    origin_x_bits: u32,
    origin_y_bits: u32,
    origin_z_bits: u32,
    seed: u32,
    fill_threshold: u32,
) void {
    const count = side * side * side;
    std.debug.assert(out.len >= count);

    if (fill_threshold == 0) {
        @memset(out[0..count], 0);
        return;
    }
    if (fill_threshold == std.math.maxInt(u32)) {
        @memset(out[0..count], 1);
        return;
    }

    const threshold_v = @as(U32x16, @splat(fill_threshold));
    const one_v = @as(U8x16, @splat(@as(u8, 1)));
    const zero_v = @as(U8x16, @splat(@as(u8, 0)));
    const vector_end = side & ~(ByteLaneCount - 1);

    var z: usize = 0;
    while (z < side) : (z += 1) {
        const z_bits = origin_z_bits + @as(u32, @intCast(z));
        const z_v = @as(U32x16, @splat(z_bits));
        var y: usize = 0;
        while (y < side) : (y += 1) {
            const y_bits = origin_y_bits + @as(u32, @intCast(y));
            const y_v = @as(U32x16, @splat(y_bits));
            const row_start = side * (y + side * z);

            var x: usize = 0;
            while (x < vector_end) : (x += ByteLaneCount) {
                const x_base = origin_x_bits + @as(u32, @intCast(x));
                const x_v = @as(U32x16, @splat(x_base)) + ByteLaneOffsets;
                const hash = cellularAutomataHash3x16(x_v, y_v, z_v, seed);
                storeU8x16(out.ptr + row_start + x, @select(u8, hash < threshold_v, one_v, zero_v));
            }
            while (x < side) : (x += 1) {
                const x_bits = origin_x_bits + @as(u32, @intCast(x));
                out[row_start + x] = @intFromBool(cellularAutomataHash3U32(x_bits, y_bits, z_bits, seed) < fill_threshold);
            }
        }
    }
}

fn cellularAutomataRandomSolidGrid3dBytesThresholdTrustedChunk(
    out: []u8,
    side: usize,
    origin_x_bits: u32,
    origin_y_bits: u32,
    origin_z_bits: u32,
    origin_x_nowrap: bool,
    origin_y_nowrap: bool,
    origin_z_nowrap: bool,
    seed: u32,
    fill_threshold: u32,
) void {
    if (origin_x_nowrap and origin_y_nowrap and origin_z_nowrap) {
        cellularAutomataRandomSolidGrid3dBytesThresholdTrustedNoWrapXYZ(
            out,
            side,
            origin_x_bits,
            origin_y_bits,
            origin_z_bits,
            seed,
            fill_threshold,
        );
        return;
    }

    cellularAutomataRandomSolidGrid3dBytesThresholdTrusted(
        out,
        side,
        origin_x_bits,
        origin_y_bits,
        origin_z_bits,
        seed,
        fill_threshold,
    );
}

fn cellularAutomataRandomSolidGrid3dBoolThresholdTrusted(
    out: []bool,
    side: usize,
    origin_x_bits: u32,
    origin_y_bits: u32,
    origin_z_bits: u32,
    seed: u32,
    fill_threshold: u32,
) void {
    const count = side * side * side;
    std.debug.assert(out.len >= count);
    cellularAutomataRandomSolidGrid3dBytesThresholdTrusted(
        boolSliceBytes(out[0..count]),
        side,
        origin_x_bits,
        origin_y_bits,
        origin_z_bits,
        seed,
        fill_threshold,
    );
}

fn cellularAutomataRandomSolidGrid3dBoolThresholdTrustedChunk(
    out: []bool,
    side: usize,
    origin_x_bits: u32,
    origin_y_bits: u32,
    origin_z_bits: u32,
    origin_x_nowrap: bool,
    origin_y_nowrap: bool,
    origin_z_nowrap: bool,
    seed: u32,
    fill_threshold: u32,
) void {
    const count = side * side * side;
    std.debug.assert(out.len >= count);
    cellularAutomataRandomSolidGrid3dBytesThresholdTrustedChunk(
        boolSliceBytes(out[0..count]),
        side,
        origin_x_bits,
        origin_y_bits,
        origin_z_bits,
        origin_x_nowrap,
        origin_y_nowrap,
        origin_z_nowrap,
        seed,
        fill_threshold,
    );
}

pub fn cellularAutomataSolidNeighborCount(
    grid: []const bool,
    side: usize,
    x: usize,
    y: usize,
    z: usize,
    options: SmoothOptions,
) Error!u32 {
    try validateSmoothOptions(side, options);
    const count = try cellularAutomataCellCount(side);
    if (grid.len < count) return error.BufferTooSmall;
    if (x >= side or y >= side or z >= side) return error.CoordinateOutOfBounds;

    return countSolidNeighborsUnchecked(
        grid,
        side,
        x,
        y,
        z,
        try usizeToIsize(options.neighborhood_radius),
        options.boundary_is_solid,
    );
}

pub fn cellularAutomataSmooth3d(output: []bool, input: []const bool, side: usize, options: SmoothOptions) Error!void {
    try validateSmoothOptions(side, options);
    const count = try cellularAutomataCellCount(side);
    if (input.len < count or output.len < count) return error.BufferTooSmall;
    if (@intFromPtr(output.ptr) == @intFromPtr(input.ptr)) return error.InPlaceSmoothNotSupported;

    if (options.neighborhood_radius == 1) {
        smoothRadius1RollingX(output[0..count], input[0..count], side, options.solid_threshold, options.boundary_is_solid);
        return;
    }

    const radius = try usizeToIsize(options.neighborhood_radius);
    var z: usize = 0;
    while (z < side) : (z += 1) {
        var y: usize = 0;
        while (y < side) : (y += 1) {
            var x: usize = 0;
            while (x < side) : (x += 1) {
                const solid_neighbors = countSolidNeighborsUnchecked(
                    input,
                    side,
                    x,
                    y,
                    z,
                    radius,
                    options.boundary_is_solid,
                );
                output[cellularAutomataIndex3d(x, y, z, side)] = solid_neighbors >= options.solid_threshold;
            }
        }
    }
}

pub fn cellularAutomataSmooth3dRadius1ThresholdSweep(
    output: []bool,
    input: []const bool,
    side: usize,
    solid_thresholds: []const u32,
    boundary_is_solid: bool,
) Error!void {
    if (side == 0) return error.InvalidGridSide;
    const count = try cellularAutomataCellCount(side);
    const output_count = std.math.mul(usize, count, solid_thresholds.len) catch return error.CellCountOverflow;
    if (input.len < count or output.len < output_count) return error.BufferTooSmall;
    if (@intFromPtr(output.ptr) == @intFromPtr(input.ptr)) return error.InPlaceSmoothNotSupported;

    for (solid_thresholds) |solid_threshold| {
        if (solid_threshold > 26) return error.InvalidThreshold;
    }
    if (solid_thresholds.len == 0) return;

    const plane = side * side;
    var z: usize = 0;
    while (z < side) : (z += 1) {
        var y: usize = 0;
        while (y < side) : (y += 1) {
            var left = radius1RollingColumnSum(input[0..count], side, plane, -1, y, z, boundary_is_solid);
            var middle = radius1RollingColumnSum(input[0..count], side, plane, 0, y, z, boundary_is_solid);
            var right = radius1RollingColumnSum(input[0..count], side, plane, 1, y, z, boundary_is_solid);
            var x: usize = 0;
            while (x < side) : (x += 1) {
                const index = cellularAutomataIndex3d(x, y, z, side);
                const solid_neighbors = left + middle + right - @intFromBool(input[index]);
                for (solid_thresholds, 0..) |solid_threshold, candidate| {
                    output[candidate * count + index] = solid_neighbors >= solid_threshold;
                }

                left = middle;
                middle = right;
                right = radius1RollingColumnSum(input[0..count], side, plane, @as(isize, @intCast(x)) + 2, y, z, boundary_is_solid);
            }
        }
    }
}

fn radius1RollingColumnSum(
    grid: []const bool,
    side: usize,
    plane: usize,
    x_i: isize,
    y: usize,
    z: usize,
    boundary_is_solid: bool,
) u32 {
    const side_i: isize = @intCast(side);
    if (x_i < 0 or x_i >= side_i) return if (boundary_is_solid) 9 else 0;

    const ys = axisStart(y);
    const ye = axisEnd(y, side);
    const zs = axisStart(z);
    const ze = axisEnd(z, side);
    const valid_cells = (ye - ys + 1) * (ze - zs + 1);
    var count: u32 = if (boundary_is_solid) @intCast(9 - valid_cells) else 0;
    const x: usize = @intCast(x_i);

    var nz = zs;
    while (nz <= ze) : (nz += 1) {
        const plane_base = nz * plane;
        var ny = ys;
        while (ny <= ye) : (ny += 1) {
            count += @intFromBool(grid[plane_base + ny * side + x]);
        }
    }
    return count;
}

fn smoothRadius1RollingX(
    output: []bool,
    input: []const bool,
    side: usize,
    solid_threshold: u32,
    boundary_is_solid: bool,
) void {
    const plane = side * side;
    var z: usize = 0;
    while (z < side) : (z += 1) {
        var y: usize = 0;
        while (y < side) : (y += 1) {
            var left = radius1RollingColumnSum(input, side, plane, -1, y, z, boundary_is_solid);
            var middle = radius1RollingColumnSum(input, side, plane, 0, y, z, boundary_is_solid);
            var right = radius1RollingColumnSum(input, side, plane, 1, y, z, boundary_is_solid);
            var x: usize = 0;
            while (x < side) : (x += 1) {
                const base = cellularAutomataIndex3d(x, y, z, side);
                const solid_neighbors = left + middle + right - @intFromBool(input[base]);
                output[base] = solid_neighbors >= solid_threshold;

                left = middle;
                middle = right;
                right = radius1RollingColumnSum(input, side, plane, @as(isize, @intCast(x)) + 2, y, z, boundary_is_solid);
            }
        }
    }
}

inline fn countRadius1InteriorBool(grid: []const bool, base: usize, side: usize, plane: usize) u32 {
    var count: u32 = 0;
    count += @intFromBool(grid[base - plane - side - 1]);
    count += @intFromBool(grid[base - plane - side]);
    count += @intFromBool(grid[base - plane - side + 1]);
    count += @intFromBool(grid[base - plane - 1]);
    count += @intFromBool(grid[base - plane]);
    count += @intFromBool(grid[base - plane + 1]);
    count += @intFromBool(grid[base - plane + side - 1]);
    count += @intFromBool(grid[base - plane + side]);
    count += @intFromBool(grid[base - plane + side + 1]);
    count += @intFromBool(grid[base - side - 1]);
    count += @intFromBool(grid[base - side]);
    count += @intFromBool(grid[base - side + 1]);
    count += @intFromBool(grid[base - 1]);
    count += @intFromBool(grid[base + 1]);
    count += @intFromBool(grid[base + side - 1]);
    count += @intFromBool(grid[base + side]);
    count += @intFromBool(grid[base + side + 1]);
    count += @intFromBool(grid[base + plane - side - 1]);
    count += @intFromBool(grid[base + plane - side]);
    count += @intFromBool(grid[base + plane - side + 1]);
    count += @intFromBool(grid[base + plane - 1]);
    count += @intFromBool(grid[base + plane]);
    count += @intFromBool(grid[base + plane + 1]);
    count += @intFromBool(grid[base + plane + side - 1]);
    count += @intFromBool(grid[base + plane + side]);
    count += @intFromBool(grid[base + plane + side + 1]);
    return count;
}

inline fn countRadius1InteriorByte(grid: []const u8, base: usize, side: usize, plane: usize) u8 {
    var count: u8 = 0;
    count += grid[base - plane - side - 1];
    count += grid[base - plane - side];
    count += grid[base - plane - side + 1];
    count += grid[base - plane - 1];
    count += grid[base - plane];
    count += grid[base - plane + 1];
    count += grid[base - plane + side - 1];
    count += grid[base - plane + side];
    count += grid[base - plane + side + 1];
    count += grid[base - side - 1];
    count += grid[base - side];
    count += grid[base - side + 1];
    count += grid[base - 1];
    count += grid[base + 1];
    count += grid[base + side - 1];
    count += grid[base + side];
    count += grid[base + side + 1];
    count += grid[base + plane - side - 1];
    count += grid[base + plane - side];
    count += grid[base + plane - side + 1];
    count += grid[base + plane - 1];
    count += grid[base + plane];
    count += grid[base + plane + 1];
    count += grid[base + plane + side - 1];
    count += grid[base + plane + side];
    count += grid[base + plane + side + 1];
    return count;
}

inline fn anyRadius1InteriorByte(grid: []const u8, base: usize, side: usize, plane: usize) bool {
    return grid[base - plane - side - 1] != 0 or
        grid[base - plane - side] != 0 or
        grid[base - plane - side + 1] != 0 or
        grid[base - plane - 1] != 0 or
        grid[base - plane] != 0 or
        grid[base - plane + 1] != 0 or
        grid[base - plane + side - 1] != 0 or
        grid[base - plane + side] != 0 or
        grid[base - plane + side + 1] != 0 or
        grid[base - side - 1] != 0 or
        grid[base - side] != 0 or
        grid[base - side + 1] != 0 or
        grid[base - 1] != 0 or
        grid[base + 1] != 0 or
        grid[base + side - 1] != 0 or
        grid[base + side] != 0 or
        grid[base + side + 1] != 0 or
        grid[base + plane - side - 1] != 0 or
        grid[base + plane - side] != 0 or
        grid[base + plane - side + 1] != 0 or
        grid[base + plane - 1] != 0 or
        grid[base + plane] != 0 or
        grid[base + plane + 1] != 0 or
        grid[base + plane + side - 1] != 0 or
        grid[base + plane + side] != 0 or
        grid[base + plane + side + 1] != 0;
}

inline fn allRadius1InteriorByte(grid: []const u8, base: usize, side: usize, plane: usize) bool {
    return grid[base - plane - side - 1] != 0 and
        grid[base - plane - side] != 0 and
        grid[base - plane - side + 1] != 0 and
        grid[base - plane - 1] != 0 and
        grid[base - plane] != 0 and
        grid[base - plane + 1] != 0 and
        grid[base - plane + side - 1] != 0 and
        grid[base - plane + side] != 0 and
        grid[base - plane + side + 1] != 0 and
        grid[base - side - 1] != 0 and
        grid[base - side] != 0 and
        grid[base - side + 1] != 0 and
        grid[base - 1] != 0 and
        grid[base + 1] != 0 and
        grid[base + side - 1] != 0 and
        grid[base + side] != 0 and
        grid[base + side + 1] != 0 and
        grid[base + plane - side - 1] != 0 and
        grid[base + plane - side] != 0 and
        grid[base + plane - side + 1] != 0 and
        grid[base + plane - 1] != 0 and
        grid[base + plane] != 0 and
        grid[base + plane + 1] != 0 and
        grid[base + plane + side - 1] != 0 and
        grid[base + plane + side] != 0 and
        grid[base + plane + side + 1] != 0;
}

inline fn axisStart(coord: usize) usize {
    return if (coord == 0) 0 else coord - 1;
}

inline fn axisEnd(coord: usize, side: usize) usize {
    return if (coord + 1 >= side) side - 1 else coord + 1;
}

fn countRadius1BoundaryPlanned(
    grid: []const bool,
    side: usize,
    plane: usize,
    x: usize,
    y: usize,
    z: usize,
    boundary_is_solid: bool,
) u32 {
    const xs = axisStart(x);
    const xe = axisEnd(x, side);
    const ys = axisStart(y);
    const ye = axisEnd(y, side);
    const zs = axisStart(z);
    const ze = axisEnd(z, side);
    const valid_cells = (xe - xs + 1) * (ye - ys + 1) * (ze - zs + 1) - 1;
    var count: u32 = if (boundary_is_solid) @intCast(26 - valid_cells) else 0;

    var nz = zs;
    while (nz <= ze) : (nz += 1) {
        const plane_base = nz * plane;
        var ny = ys;
        while (ny <= ye) : (ny += 1) {
            const row_base = plane_base + ny * side;
            var nx = xs;
            while (nx <= xe) : (nx += 1) {
                if (nx == x and ny == y and nz == z) continue;
                count += @intFromBool(grid[row_base + nx]);
            }
        }
    }
    return count;
}

inline fn meetsRadius1Threshold(count: u32, solid_threshold: u8, threshold_minus_one: u8) bool {
    return if (solid_threshold == 0) true else count > threshold_minus_one;
}

fn smoothTinyPrepared(plan: *const PreparedSmoothPlan, output: []bool, input: []const bool) void {
    var index: usize = 0;
    while (index < plan.count) : (index += 1) {
        var count: u32 = if (plan.boundary_is_solid) plan.tiny_outside_counts[index] else 0;
        var neighbor_index: usize = 0;
        while (neighbor_index < plan.tiny_counts[index]) : (neighbor_index += 1) {
            count += @intFromBool(input[plan.tiny_indices[index][neighbor_index]]);
        }
        output[index] = count >= plan.solid_threshold;
    }
}

fn smoothRadius1Prepared(plan: *const PreparedSmoothPlan, output: []bool, input: []const bool) void {
    var z: usize = 0;
    while (z < plan.side) : (z += 1) {
        const plane_base = z * plan.plane;
        var y: usize = 0;
        while (y < plan.side) : (y += 1) {
            const row_base = plane_base + y * plan.side;
            var x: usize = 0;
            while (x < plan.side) : (x += 1) {
                const base = row_base + x;
                const count = if (x == 0 or y == 0 or z == 0 or x + 1 == plan.side or y + 1 == plan.side or z + 1 == plan.side)
                    countRadius1BoundaryPlanned(input, plan.side, plan.plane, x, y, z, plan.boundary_is_solid)
                else
                    countRadius1InteriorBool(input, base, plan.side, plan.plane);
                output[base] = count >= plan.solid_threshold;
            }
        }
    }
}

fn cellularAutomataSmooth3dBytes(output: []u8, input: []const u8, side: usize, options: SmoothOptions) Error!void {
    try validateSmoothOptions(side, options);
    if (options.neighborhood_radius != 1) return error.InvalidNeighborhoodRadius;
    const count = try cellularAutomataCellCount(side);
    if (input.len < count or output.len < count) return error.BufferTooSmall;
    if (@intFromPtr(output.ptr) == @intFromPtr(input.ptr)) return error.InPlaceSmoothNotSupported;

    const threshold: u8 = @intCast(options.solid_threshold);
    if (side < 3) {
        var index: usize = 0;
        while (index < count) : (index += 1) {
            const x = index % side;
            const yz = index / side;
            const y = yz % side;
            const z = yz / side;
            output[index] = if (countSolidNeighborsByteUnchecked(input, side, x, y, z, options.boundary_is_solid) >= threshold) 1 else 0;
        }
        return;
    }

    const plane = side * side;
    const threshold_v = @as(U8x16, @splat(threshold));
    const one_v = @as(U8x16, @splat(@as(u8, 1)));
    const zero_v = @as(U8x16, @splat(@as(u8, 0)));

    var z: usize = 0;
    while (z < side) : (z += 1) {
        var y: usize = 0;
        while (y < side) : (y += 1) {
            var x: usize = 0;
            const row_start = side * (y + side * z);

            if (z == 0 or y == 0 or z + 1 == side or y + 1 == side) {
                while (x < side) : (x += 1) {
                    output[row_start + x] = if (countSolidNeighborsByteUnchecked(input, side, x, y, z, options.boundary_is_solid) >= threshold) 1 else 0;
                }
                continue;
            }

            output[row_start] = if (countSolidNeighborsByteUnchecked(input, side, 0, y, z, options.boundary_is_solid) >= threshold) 1 else 0;
            x = 1;
            while (x + ByteLaneCount <= side - 1) : (x += ByteLaneCount) {
                const base = row_start + x;
                smoothInteriorByteBlock(output.ptr, input.ptr, base, side, plane, threshold_v, one_v, zero_v);
            }

            if (side > ByteLaneCount + 1 and x < side - 1) {
                const tail_x = side - 1 - ByteLaneCount;
                if (tail_x > 0) {
                    smoothInteriorByteBlock(output.ptr, input.ptr, row_start + tail_x, side, plane, threshold_v, one_v, zero_v);
                    x = side - 1;
                }
            }

            while (x < side) : (x += 1) {
                output[row_start + x] = if (countSolidNeighborsByteUnchecked(input, side, x, y, z, options.boundary_is_solid) >= threshold) 1 else 0;
            }
        }
    }
}

pub fn cellularAutomataCopyChunkCenter(out: []bool, temp: []const bool, chunk_size: usize, padding: usize) Error!void {
    if (chunk_size == 0) return error.InvalidChunkSize;
    const temp_side = try cellularAutomataTemporarySide(chunk_size, padding);
    const chunk_count = try cellularAutomataCellCount(chunk_size);
    const temp_count = try cellularAutomataCellCount(temp_side);
    if (out.len < chunk_count or temp.len < temp_count) return error.BufferTooSmall;

    const out_bytes = boolSliceBytes(out[0..chunk_count]);
    const temp_bytes = @as([*]const u8, @ptrCast(temp.ptr))[0..temp.len];
    var z: usize = 0;
    while (z < chunk_size) : (z += 1) {
        var y: usize = 0;
        while (y < chunk_size) : (y += 1) {
            const out_row = cellularAutomataIndex3d(0, y, z, chunk_size);
            const temp_row = cellularAutomataIndex3d(padding, y + padding, z + padding, temp_side);
            @memcpy(out_bytes[out_row..][0..chunk_size], temp_bytes[temp_row..][0..chunk_size]);
        }
    }
}

fn cellularAutomataCopyChunkCenterBytes(out: []bool, temp: []const u8, chunk_size: usize, padding: usize) Error!void {
    if (chunk_size == 0) return error.InvalidChunkSize;
    const temp_side = try cellularAutomataTemporarySide(chunk_size, padding);
    const chunk_count = try cellularAutomataCellCount(chunk_size);
    const temp_count = try cellularAutomataCellCount(temp_side);
    if (out.len < chunk_count or temp.len < temp_count) return error.BufferTooSmall;

    const out_bytes = boolSliceBytes(out[0..chunk_count]);
    var z: usize = 0;
    while (z < chunk_size) : (z += 1) {
        var y: usize = 0;
        while (y < chunk_size) : (y += 1) {
            const out_row = cellularAutomataIndex3d(0, y, z, chunk_size);
            const temp_row = cellularAutomataIndex3d(padding, y + padding, z + padding, temp_side);
            @memcpy(out_bytes[out_row..][0..chunk_size], temp[temp_row..][0..chunk_size]);
        }
    }
}

fn cellularAutomataBytesEqualCenteredCrop(
    output: []const u8,
    input: []const u8,
    input_side: usize,
    output_side: usize,
    crop: usize,
) bool {
    std.debug.assert(output_side + crop * 2 == input_side);
    std.debug.assert(output.len >= output_side * output_side * output_side);
    std.debug.assert(input.len >= input_side * input_side * input_side);

    const input_plane = input_side * input_side;
    const output_plane = output_side * output_side;
    var z: usize = 0;
    while (z < output_side) : (z += 1) {
        var y: usize = 0;
        while (y < output_side) : (y += 1) {
            const input_row = (z + crop) * input_plane + (y + crop) * input_side + crop;
            const output_row = z * output_plane + y * output_side;
            if (!std.mem.eql(u8, output[output_row..][0..output_side], input[input_row..][0..output_side])) {
                return false;
            }
        }
    }
    return true;
}

fn cellularAutomataCopyCenteredCropBytesToOut(
    out: []bool,
    input: []const u8,
    input_side: usize,
    output_side: usize,
    crop: usize,
) void {
    std.debug.assert(output_side + crop * 2 == input_side);
    std.debug.assert(out.len >= output_side * output_side * output_side);
    std.debug.assert(input.len >= input_side * input_side * input_side);

    const out_bytes = boolSliceBytes(out[0 .. output_side * output_side * output_side]);
    cellularAutomataCopyCenteredCropBytes(out_bytes, input, input_side, output_side, crop);
}

fn cellularAutomataCopyCenteredCropBytes(
    out_bytes: []u8,
    input: []const u8,
    input_side: usize,
    output_side: usize,
    crop: usize,
) void {
    std.debug.assert(output_side + crop * 2 == input_side);
    std.debug.assert(out_bytes.len >= output_side * output_side * output_side);
    std.debug.assert(input.len >= input_side * input_side * input_side);

    const input_plane = input_side * input_side;
    const output_plane = output_side * output_side;
    var z: usize = 0;
    while (z < output_side) : (z += 1) {
        var y: usize = 0;
        while (y < output_side) : (y += 1) {
            const input_row = (z + crop) * input_plane + (y + crop) * input_side + crop;
            const output_row = z * output_plane + y * output_side;
            @memcpy(out_bytes[output_row..][0..output_side], input[input_row..][0..output_side]);
        }
    }
}

fn cellularAutomataSmooth3dBytesShrinkToBytes(
    output: []u8,
    input: []const u8,
    input_side: usize,
    output_side: usize,
    solid_threshold: u32,
) Error!void {
    if (input_side < 3 or output_side + 2 != input_side) return error.InvalidGridSide;
    const input_count = try cellularAutomataCellCount(input_side);
    const output_count = try cellularAutomataCellCount(output_side);
    if (input.len < input_count or output.len < output_count) return error.BufferTooSmall;

    cellularAutomataSmooth3dBytesShrinkToBytesTrusted(output, input, input_side, output_side, solid_threshold);
}

fn cellularAutomataSmooth3dBytesShrinkToBytesTrusted(
    output: []u8,
    input: []const u8,
    input_side: usize,
    output_side: usize,
    solid_threshold: u32,
) void {
    std.debug.assert(input_side >= 3);
    std.debug.assert(output_side + 2 == input_side);
    std.debug.assert(input.len >= input_side * input_side * input_side);
    std.debug.assert(output.len >= output_side * output_side * output_side);

    const start: usize = 1;
    const end = input_side - 1;
    const width = output_side;
    const plane = input_side * input_side;
    const threshold: u8 = @intCast(solid_threshold);
    const input_count = input_side * input_side * input_side;
    const output_count = output_side * output_side * output_side;
    if (threshold == 0) {
        @memset(output[0..output_count], 1);
        return;
    }
    if (threshold > 26 or byteGridSolidCountLessThan(input[0..input_count], threshold)) {
        @memset(output[0..output_count], 0);
        return;
    }
    if (byteGridEmptyCountAtMost(input[0..input_count], 26 - threshold)) {
        @memset(output[0..output_count], 1);
        return;
    }

    if (comptime use_smooth_trusted_sme) {
        if (input_side >= SmoothTrustedSmeMinInputSide and shouldUseSmoothTrustedSme(output_side)) {
            if (comptime use_smooth_trusted_sme2) {
                cellularAutomataSmooth3dBytesShrinkToBytesTrustedAarch64Sme2(
                    output.ptr,
                    input.ptr,
                    input_side,
                    output_side,
                    threshold,
                );
            } else {
                cellularAutomataSmooth3dBytesShrinkToBytesTrustedAarch64(
                    output.ptr,
                    input.ptr,
                    input_side,
                    output_side,
                    threshold,
                );
            }
            return;
        }
    }

    if (width >= ByteLaneCount) {
        const output_plane = output_side * output_side;
        const column_count = end - start;

        var z: usize = start;
        while (z < end) : (z += 1) {
            const lower_base = (z - 1) * plane + start * input_side;
            const middle_base = z * plane + start * input_side;
            const upper_base = (z + 1) * plane + start * input_side;
            const output_base = (z - 1) * output_plane + (start - 1) * output_side;
            var x: usize = start;

            while (x + ByteLaneCount <= end) : (x += ByteLaneCount) {
                smoothInteriorByteColumnToBytes(
                    output.ptr + output_base + (x - 1),
                    input.ptr + lower_base + x,
                    input.ptr + middle_base + x,
                    input.ptr + upper_base + x,
                    input_side,
                    output_side,
                    column_count,
                    threshold,
                );
            }

            if (x < end) {
                const tail_x = end - ByteLaneCount;
                smoothInteriorByteColumnToBytes(
                    output.ptr + output_base + (tail_x - 1),
                    input.ptr + lower_base + tail_x,
                    input.ptr + middle_base + tail_x,
                    input.ptr + upper_base + tail_x,
                    input_side,
                    output_side,
                    column_count,
                    threshold,
                );
                x = end;
            }

            var scalar_x = x;
            while (scalar_x < end) : (scalar_x += 1) {
                var y: usize = start;
                while (y < end) : (y += 1) {
                    const output_index = cellularAutomataIndex3d(scalar_x - 1, y - 1, z - 1, output_side);
                    const input_base = z * plane + y * input_side + scalar_x;
                    output[output_index] = if (countRadius1InteriorByte(input, input_base, input_side, plane) >= threshold) 1 else 0;
                }
            }
        }
    } else {
        if (threshold == 1) {
            cellularAutomataSmooth3dBytesShrinkScalarAnyToBytes(output, input, input_side, output_side);
            return;
        }
        if (threshold == 26) {
            cellularAutomataSmooth3dBytesShrinkScalarAllToBytes(output, input, input_side, output_side);
            return;
        }
        cellularAutomataSmooth3dBytesShrinkScalarCountToBytes(output, input, input_side, output_side, threshold);
    }
}

fn cellularAutomataSmooth3dBytesShrinkScalarCountToBytes(
    output: []u8,
    input: []const u8,
    input_side: usize,
    output_side: usize,
    threshold: u8,
) void {
    const plane = input_side * input_side;

    var out_index: usize = 0;
    var z: usize = 0;
    while (z < output_side) : (z += 1) {
        const input_z_base = (z + 1) * plane;
        var y: usize = 0;
        while (y < output_side) : (y += 1) {
            const input_row = input_z_base + (y + 1) * input_side;
            var x: usize = 0;
            while (x < output_side) : ({
                x += 1;
                out_index += 1;
            }) {
                const input_base = input_row + x + 1;
                output[out_index] = if (countRadius1InteriorByte(input, input_base, input_side, plane) >= threshold) 1 else 0;
            }
        }
    }
}

fn cellularAutomataSmooth3dBytesShrinkScalarAnyToBytes(
    output: []u8,
    input: []const u8,
    input_side: usize,
    output_side: usize,
) void {
    const end = input_side - 1;
    const plane = input_side * input_side;
    const output_plane = output_side * output_side;

    var z: usize = 1;
    while (z < end) : (z += 1) {
        const input_z_base = z * plane;
        const output_z_base = (z - 1) * output_plane;
        var y: usize = 1;
        while (y < end) : (y += 1) {
            const input_row = input_z_base + y * input_side;
            const output_row = output_z_base + (y - 1) * output_side;
            var x: usize = 1;
            while (x < end) : (x += 1) {
                const input_base = input_row + x;
                output[output_row + x - 1] = @intFromBool(anyRadius1InteriorByte(input, input_base, input_side, plane));
            }
        }
    }
}

fn cellularAutomataSmooth3dBytesShrinkScalarAllToBytes(
    output: []u8,
    input: []const u8,
    input_side: usize,
    output_side: usize,
) void {
    const end = input_side - 1;
    const plane = input_side * input_side;
    const output_plane = output_side * output_side;

    var z: usize = 1;
    while (z < end) : (z += 1) {
        const input_z_base = z * plane;
        const output_z_base = (z - 1) * output_plane;
        var y: usize = 1;
        while (y < end) : (y += 1) {
            const input_row = input_z_base + y * input_side;
            const output_row = output_z_base + (y - 1) * output_side;
            var x: usize = 1;
            while (x < end) : (x += 1) {
                const input_base = input_row + x;
                output[output_row + x - 1] = @intFromBool(allRadius1InteriorByte(input, input_base, input_side, plane));
            }
        }
    }
}

fn cellularAutomataSmooth3dBoolShrinkToBool(
    output: []bool,
    input: []const bool,
    input_side: usize,
    output_side: usize,
    radius: usize,
    solid_threshold: u32,
) Error!void {
    const doubled_radius = std.math.mul(usize, radius, 2) catch return error.CellCountOverflow;
    if (input_side < doubled_radius + 1 or output_side + doubled_radius != input_side) return error.InvalidGridSide;
    const input_count = try cellularAutomataCellCount(input_side);
    const output_count = try cellularAutomataCellCount(output_side);
    if (input.len < input_count or output.len < output_count) return error.BufferTooSmall;

    if (solid_threshold == 0) {
        @memset(output[0..output_count], true);
        return;
    }
    const max_neighbors = try maxNeighborCount(radius);
    if (solid_threshold == 1) {
        cellularAutomataSmooth3dBoolShrinkAnyToBool(output, input, input_side, output_side, radius);
        return;
    }
    if (solid_threshold == max_neighbors) {
        cellularAutomataSmooth3dBoolShrinkAllToBool(output, input, input_side, output_side, radius);
        return;
    }
    const input_bytes = boolSliceConstBytes(input[0..input_count]);
    if (byteGridSolidCountLessThan(input_bytes, solid_threshold)) {
        @memset(output[0..output_count], false);
        return;
    }
    if (byteGridEmptyCountAtMost(input_bytes, max_neighbors - solid_threshold)) {
        @memset(output[0..output_count], true);
        return;
    }

    cellularAutomataSmooth3dBoolShrinkBytesToBool(output, input, input_side, output_side, radius, solid_threshold);
}

fn cellularAutomataSmooth3dBoolShrinkAnyToBool(
    output: []bool,
    input: []const bool,
    input_side: usize,
    output_side: usize,
    radius: usize,
) void {
    const input_bytes = boolSliceConstBytes(input);
    const output_count = output_side * output_side * output_side;
    const output_bytes = boolSliceBytes(output[0..output_count]);
    const plane = input_side * input_side;
    const window_side = radius * 2 + 1;
    const input_count = input_side * input_side * input_side;
    const inverse_cutoff = chunkInverseScatterDensityCutoff(input_count);
    const nonzero_count = countNonZeroBytesUpTo(input_bytes[0..input_count], inverse_cutoff);
    if (nonzero_count == 0) {
        @memset(output_bytes, 0);
        return;
    }
    if (nonzero_count <= inverse_cutoff) {
        cellularAutomataSmooth3dBytesShrinkAnyInverseToBytes(output_bytes, input_bytes, input_side, output_side, radius);
        return;
    }

    var out_index: usize = 0;
    var z: usize = 0;
    while (z < output_side) : (z += 1) {
        const z_window_base = z * plane;
        var y: usize = 0;
        while (y < output_side) : (y += 1) {
            var window_base = z_window_base + y * input_side;
            var x: usize = 0;
            while (x < output_side) : ({
                x += 1;
                window_base += 1;
                out_index += 1;
            }) {
                output_bytes[out_index] = @intFromBool(anySolidNeighborsShrinkWindowBytes(
                    input_bytes,
                    input_side,
                    plane,
                    window_base,
                    radius,
                    window_side,
                ));
            }
        }
    }
}

fn cellularAutomataSmooth3dBoolShrinkAllToBool(
    output: []bool,
    input: []const bool,
    input_side: usize,
    output_side: usize,
    radius: usize,
) void {
    const input_bytes = boolSliceConstBytes(input);
    const output_count = output_side * output_side * output_side;
    const output_bytes = boolSliceBytes(output[0..output_count]);
    const input_count = input_side * input_side * input_side;
    const inverse_cutoff = chunkInverseScatterDensityCutoff(input_count);
    const zero_count = countZeroBytesUpTo(input_bytes[0..input_count], inverse_cutoff);
    if (zero_count == 0) {
        @memset(output_bytes, 1);
        return;
    }
    if (zero_count <= inverse_cutoff) {
        cellularAutomataSmooth3dBytesShrinkAllInverseToBytes(output_bytes, input_bytes, input_side, output_side, radius);
        return;
    }

    const plane = input_side * input_side;
    var out_index: usize = 0;
    var z: usize = 0;
    while (z < output_side) : (z += 1) {
        const z_window_base = z * plane;
        var y: usize = 0;
        while (y < output_side) : (y += 1) {
            var window_base = z_window_base + y * input_side;
            var x: usize = 0;
            while (x < output_side) : ({
                x += 1;
                window_base += 1;
            }) {
                output[out_index] = allSolidNeighborsShrinkWindowBytes(
                    input_bytes,
                    input_side,
                    plane,
                    window_base,
                    radius,
                );
                out_index += 1;
            }
        }
    }
}

inline fn allNonZeroSpan(pointer: [*]const u8, len: usize) bool {
    switch (len) {
        5 => return pointer[0] != 0 and
            pointer[1] != 0 and
            pointer[2] != 0 and
            pointer[3] != 0 and
            pointer[4] != 0,
        7 => return pointer[0] != 0 and
            pointer[1] != 0 and
            pointer[2] != 0 and
            pointer[3] != 0 and
            pointer[4] != 0 and
            pointer[5] != 0 and
            pointer[6] != 0,
        9 => return pointer[0] != 0 and
            pointer[1] != 0 and
            pointer[2] != 0 and
            pointer[3] != 0 and
            pointer[4] != 0 and
            pointer[5] != 0 and
            pointer[6] != 0 and
            pointer[7] != 0 and
            pointer[8] != 0,
        else => {},
    }

    var index: usize = 0;
    while (index < len) : (index += 1) {
        if (pointer[index] == 0) return false;
    }
    return true;
}

inline fn anyNonZeroSpan(pointer: [*]const u8, len: usize) bool {
    var index: usize = 0;
    while (index < len) : (index += 1) {
        if (pointer[index] != 0) return true;
    }
    return false;
}

fn anySolidNeighborsShrinkWindowBytes(
    input: []const u8,
    input_side: usize,
    plane: usize,
    window_base: usize,
    radius: usize,
    window_side: usize,
) bool {
    var window_z: usize = 0;
    while (window_z < window_side) : (window_z += 1) {
        const plane_base = window_base + window_z * plane;
        var window_y: usize = 0;
        while (window_y < window_side) : (window_y += 1) {
            const row_base = plane_base + window_y * input_side;
            if (window_z == radius and window_y == radius) {
                if (anyNonZeroSpan(input.ptr + row_base, radius)) return true;
                if (anyNonZeroSpan(input.ptr + row_base + radius + 1, radius)) return true;
            } else if (anyNonZeroSpan(input.ptr + row_base, window_side)) {
                return true;
            }
        }
    }

    return false;
}

fn allSolidNeighborsShrinkWindowBytes(
    input: []const u8,
    input_side: usize,
    plane: usize,
    window_base: usize,
    radius: usize,
) bool {
    const window_side = radius * 2 + 1;
    var window_z: usize = 0;
    while (window_z < window_side) : (window_z += 1) {
        const plane_base = window_base + window_z * plane;
        var window_y: usize = 0;
        while (window_y < window_side) : (window_y += 1) {
            const row_base = plane_base + window_y * input_side;
            if (window_z == radius and window_y == radius) {
                if (!allNonZeroSpan(input.ptr + row_base, radius)) return false;
                if (!allNonZeroSpan(input.ptr + row_base + radius + 1, radius)) return false;
            } else if (!allNonZeroSpan(input.ptr + row_base, window_side)) {
                return false;
            }
        }
    }

    return true;
}

inline fn chunkInverseScatterDensityCutoff(input_count: usize) usize {
    return @max(input_count / 32, 1);
}

const MinInverseScatterRunLength: usize = 3;

fn countNonZeroBytesUpTo(input: []const u8, limit: usize) usize {
    if (input.len >= 256) {
        const zero_v = @as(U8x16, @splat(@as(u8, 0)));
        const one_v = @as(U8x16, @splat(@as(u8, 1)));
        var count: usize = 0;
        var index: usize = 0;
        while (index + ByteLaneCount <= input.len) : (index += ByteLaneCount) {
            const value = loadU8x16(input.ptr + index);
            count += @reduce(.Add, @select(u8, value != zero_v, one_v, zero_v));
            if (count > limit) return count;
        }
        while (index < input.len) : (index += 1) {
            if (input[index] != 0) {
                count += 1;
                if (count > limit) return count;
            }
        }
        return count;
    }

    var count: usize = 0;
    for (input) |value| {
        if (value != 0) {
            count += 1;
            if (count > limit) return count;
        }
    }
    return count;
}

fn countZeroBytesUpTo(input: []const u8, limit: usize) usize {
    if (input.len >= 256) {
        const zero_v = @as(U8x16, @splat(@as(u8, 0)));
        const one_v = @as(U8x16, @splat(@as(u8, 1)));
        var count: usize = 0;
        var index: usize = 0;
        while (index + ByteLaneCount <= input.len) : (index += ByteLaneCount) {
            const value = loadU8x16(input.ptr + index);
            count += @reduce(.Add, @select(u8, value == zero_v, one_v, zero_v));
            if (count > limit) return count;
        }
        while (index < input.len) : (index += 1) {
            if (input[index] == 0) {
                count += 1;
                if (count > limit) return count;
            }
        }
        return count;
    }

    var count: usize = 0;
    for (input) |value| {
        if (value == 0) {
            count += 1;
            if (count > limit) return count;
        }
    }
    return count;
}

inline fn byteGridSolidCountLessThan(input: []const u8, threshold: u32) bool {
    std.debug.assert(threshold > 0);
    const limit: usize = @intCast(threshold - 1);
    return countNonZeroBytesUpTo(input, limit) <= limit;
}

inline fn byteGridEmptyCountAtMost(input: []const u8, allowed_missing: u32) bool {
    const limit: usize = @intCast(allowed_missing);
    return countZeroBytesUpTo(input, limit) <= limit;
}

inline fn affectedOutputStart(input_coord: usize, radius_step: usize) usize {
    return if (input_coord > radius_step) input_coord - radius_step else 0;
}

inline fn affectedOutputEndExclusive(input_coord: usize, output_side: usize) usize {
    return @min(input_coord + 1, output_side);
}

fn markAffectedOutputBoxExceptCenterRadiusN(
    output: []u8,
    output_side: usize,
    radius: usize,
    input_x: usize,
    input_y: usize,
    input_z: usize,
    unmarked_count: *usize,
) void {
    const radius_step = radius * 2;
    const x0 = affectedOutputStart(input_x, radius_step);
    const y0 = affectedOutputStart(input_y, radius_step);
    const z0 = affectedOutputStart(input_z, radius_step);
    const x1 = affectedOutputEndExclusive(input_x, output_side);
    const y1 = affectedOutputEndExclusive(input_y, output_side);
    const z1 = affectedOutputEndExclusive(input_z, output_side);
    if (x0 >= x1 or y0 >= y1 or z0 >= z1) return;

    const has_center_output = input_x >= radius and input_y >= radius and input_z >= radius and
        input_x - radius < output_side and input_y - radius < output_side and input_z - radius < output_side;
    const center_x = if (has_center_output) input_x - radius else 0;
    const center_y = if (has_center_output) input_y - radius else 0;
    const center_z = if (has_center_output) input_z - radius else 0;

    var z = z0;
    while (z < z1 and unmarked_count.* != 0) : (z += 1) {
        var y = y0;
        while (y < y1 and unmarked_count.* != 0) : (y += 1) {
            const row = output_side * (y + output_side * z);
            var x = x0;
            while (x < x1) : (x += 1) {
                if (has_center_output and x == center_x and y == center_y and z == center_z) continue;
                const output_index = row + x;
                if (output[output_index] == 0) {
                    output[output_index] = 1;
                    unmarked_count.* -= 1;
                    if (unmarked_count.* == 0) return;
                }
            }
        }
    }
}

fn clearAffectedOutputBoxExceptCenterRadiusN(
    output: []u8,
    output_side: usize,
    radius: usize,
    input_x: usize,
    input_y: usize,
    input_z: usize,
    remaining_true_count: *usize,
) void {
    const radius_step = radius * 2;
    const x0 = affectedOutputStart(input_x, radius_step);
    const y0 = affectedOutputStart(input_y, radius_step);
    const z0 = affectedOutputStart(input_z, radius_step);
    const x1 = affectedOutputEndExclusive(input_x, output_side);
    const y1 = affectedOutputEndExclusive(input_y, output_side);
    const z1 = affectedOutputEndExclusive(input_z, output_side);
    if (x0 >= x1 or y0 >= y1 or z0 >= z1) return;

    const has_center_output = input_x >= radius and input_y >= radius and input_z >= radius and
        input_x - radius < output_side and input_y - radius < output_side and input_z - radius < output_side;
    const center_x = if (has_center_output) input_x - radius else 0;
    const center_y = if (has_center_output) input_y - radius else 0;
    const center_z = if (has_center_output) input_z - radius else 0;

    var z = z0;
    while (z < z1 and remaining_true_count.* != 0) : (z += 1) {
        var y = y0;
        while (y < y1 and remaining_true_count.* != 0) : (y += 1) {
            const row = output_side * (y + output_side * z);
            var x = x0;
            while (x < x1) : (x += 1) {
                if (has_center_output and x == center_x and y == center_y and z == center_z) continue;
                const output_index = row + x;
                if (output[output_index] != 0) {
                    output[output_index] = 0;
                    remaining_true_count.* -= 1;
                    if (remaining_true_count.* == 0) return;
                }
            }
        }
    }
}

fn markOutputSpanTrackingZeros(row: []u8, unmarked_count: *usize) void {
    var start: usize = 0;
    while (start < row.len and unmarked_count.* != 0) {
        const offset = std.mem.indexOfScalar(u8, row[start..], 0) orelse break;
        const index = start + offset;
        row[index] = 1;
        unmarked_count.* -= 1;
        start = index + 1;
    }
}

fn clearOutputSpanTrackingOnes(row: []u8, remaining_true_count: *usize) void {
    var start: usize = 0;
    while (start < row.len and remaining_true_count.* != 0) {
        const offset = std.mem.indexOfScalar(u8, row[start..], 1) orelse break;
        const index = start + offset;
        row[index] = 0;
        remaining_true_count.* -= 1;
        start = index + 1;
    }
}

fn markAffectedOutputRunExceptCenterRadiusN(
    output: []u8,
    output_side: usize,
    radius: usize,
    run_start: usize,
    run_end: usize,
    input_y: usize,
    input_z: usize,
    unmarked_count: *usize,
) void {
    const radius_step = radius * 2;
    const x0 = affectedOutputStart(run_start, radius_step);
    const y0 = affectedOutputStart(input_y, radius_step);
    const z0 = affectedOutputStart(input_z, radius_step);
    const x1 = @min(run_end, output_side);
    const y1 = affectedOutputEndExclusive(input_y, output_side);
    const z1 = affectedOutputEndExclusive(input_z, output_side);
    if (x0 >= x1 or y0 >= y1 or z0 >= z1) return;

    const single_cell_run = run_end == run_start + 1;
    const has_center_output = single_cell_run and
        run_start >= radius and input_y >= radius and input_z >= radius and
        run_start - radius < output_side and input_y - radius < output_side and input_z - radius < output_side;
    const center_x = if (has_center_output) run_start - radius else 0;
    const center_y = if (has_center_output) input_y - radius else 0;
    const center_z = if (has_center_output) input_z - radius else 0;

    var z = z0;
    while (z < z1 and unmarked_count.* != 0) : (z += 1) {
        var y = y0;
        while (y < y1 and unmarked_count.* != 0) : (y += 1) {
            const row_start = output_side * (y + output_side * z);
            if (has_center_output and y == center_y and z == center_z) {
                if (x0 < center_x) {
                    markOutputSpanTrackingZeros(output[row_start + x0 .. row_start + center_x], unmarked_count);
                }
                if (center_x + 1 < x1) {
                    markOutputSpanTrackingZeros(output[row_start + center_x + 1 .. row_start + x1], unmarked_count);
                }
            } else {
                markOutputSpanTrackingZeros(output[row_start + x0 .. row_start + x1], unmarked_count);
            }
        }
    }
}

fn clearAffectedOutputRunExceptCenterRadiusN(
    output: []u8,
    output_side: usize,
    radius: usize,
    run_start: usize,
    run_end: usize,
    input_y: usize,
    input_z: usize,
    remaining_true_count: *usize,
) void {
    const radius_step = radius * 2;
    const x0 = affectedOutputStart(run_start, radius_step);
    const y0 = affectedOutputStart(input_y, radius_step);
    const z0 = affectedOutputStart(input_z, radius_step);
    const x1 = @min(run_end, output_side);
    const y1 = affectedOutputEndExclusive(input_y, output_side);
    const z1 = affectedOutputEndExclusive(input_z, output_side);
    if (x0 >= x1 or y0 >= y1 or z0 >= z1) return;

    const single_cell_run = run_end == run_start + 1;
    const has_center_output = single_cell_run and
        run_start >= radius and input_y >= radius and input_z >= radius and
        run_start - radius < output_side and input_y - radius < output_side and input_z - radius < output_side;
    const center_x = if (has_center_output) run_start - radius else 0;
    const center_y = if (has_center_output) input_y - radius else 0;
    const center_z = if (has_center_output) input_z - radius else 0;

    var z = z0;
    while (z < z1 and remaining_true_count.* != 0) : (z += 1) {
        var y = y0;
        while (y < y1 and remaining_true_count.* != 0) : (y += 1) {
            const row_start = output_side * (y + output_side * z);
            if (has_center_output and y == center_y and z == center_z) {
                if (x0 < center_x) {
                    clearOutputSpanTrackingOnes(output[row_start + x0 .. row_start + center_x], remaining_true_count);
                }
                if (center_x + 1 < x1) {
                    clearOutputSpanTrackingOnes(output[row_start + center_x + 1 .. row_start + x1], remaining_true_count);
                }
            } else {
                clearOutputSpanTrackingOnes(output[row_start + x0 .. row_start + x1], remaining_true_count);
            }
        }
    }
}

fn cellularAutomataSmooth3dBytesShrinkAnyInverseToBytes(
    output: []u8,
    input: []const u8,
    input_side: usize,
    output_side: usize,
    radius: usize,
) void {
    const output_count = output_side * output_side * output_side;
    @memset(output[0..output_count], 0);
    var unmarked_count = output_count;

    var z: usize = 0;
    while (z < input_side and unmarked_count != 0) : (z += 1) {
        var y: usize = 0;
        while (y < input_side and unmarked_count != 0) : (y += 1) {
            const row = input_side * (y + input_side * z);
            const row_values = input[row..][0..input_side];
            var start: usize = 0;
            while (start < input_side and unmarked_count != 0) {
                const offset = std.mem.indexOfScalar(u8, row_values[start..], 1) orelse break;
                const run_start = start + offset;
                const end_offset = std.mem.indexOfScalar(u8, row_values[run_start..], 0);
                const run_end = if (end_offset) |end| run_start + end else input_side;
                if (run_end - run_start >= MinInverseScatterRunLength) {
                    markAffectedOutputRunExceptCenterRadiusN(output, output_side, radius, run_start, run_end, y, z, &unmarked_count);
                } else {
                    var x = run_start;
                    while (x < run_end and unmarked_count != 0) : (x += 1) {
                        markAffectedOutputBoxExceptCenterRadiusN(output, output_side, radius, x, y, z, &unmarked_count);
                    }
                }
                start = run_end;
            }
        }
    }
}

fn cellularAutomataSmooth3dBytesShrinkAllInverseToBytes(
    output: []u8,
    input: []const u8,
    input_side: usize,
    output_side: usize,
    radius: usize,
) void {
    const output_count = output_side * output_side * output_side;
    @memset(output[0..output_count], 1);
    var remaining_true_count = output_count;

    var z: usize = 0;
    while (z < input_side and remaining_true_count != 0) : (z += 1) {
        var y: usize = 0;
        while (y < input_side and remaining_true_count != 0) : (y += 1) {
            const row = input_side * (y + input_side * z);
            const row_values = input[row..][0..input_side];
            var start: usize = 0;
            while (start < input_side and remaining_true_count != 0) {
                const offset = std.mem.indexOfScalar(u8, row_values[start..], 0) orelse break;
                const run_start = start + offset;
                const end_offset = std.mem.indexOfScalar(u8, row_values[run_start..], 1);
                const run_end = if (end_offset) |end| run_start + end else input_side;
                if (run_end - run_start >= MinInverseScatterRunLength) {
                    clearAffectedOutputRunExceptCenterRadiusN(output, output_side, radius, run_start, run_end, y, z, &remaining_true_count);
                } else {
                    var x = run_start;
                    while (x < run_end and remaining_true_count != 0) : (x += 1) {
                        clearAffectedOutputBoxExceptCenterRadiusN(output, output_side, radius, x, y, z, &remaining_true_count);
                    }
                }
                start = run_end;
            }
        }
    }
}

fn cellularAutomataSmooth3dBoolShrinkBytesToBool(
    output: []bool,
    input: []const bool,
    input_side: usize,
    output_side: usize,
    radius: usize,
    solid_threshold: u32,
) void {
    const input_bytes = boolSliceConstBytes(input);
    const plane = input_side * input_side;
    const window_side = radius * 2 + 1;
    if (radius == 2) {
        cellularAutomataSmooth3dBoolShrinkRadius2BytesToBoolVector(
            output,
            input_bytes,
            input_side,
            output_side,
            @intCast(solid_threshold),
        );
        return;
    }
    if (radius == 3) {
        cellularAutomataSmooth3dBoolShrinkRadius3BytesToBoolRows(
            output,
            input_bytes,
            input_side,
            output_side,
            solid_threshold,
        );
        return;
    }
    if (window_side >= 8) {
        cellularAutomataSmooth3dBoolShrinkBytesToBoolWideSwar(
            output,
            input_bytes,
            input_side,
            output_side,
            radius,
            window_side,
            solid_threshold,
        );
        return;
    }

    const center_offset = radius * plane + radius * input_side + radius;

    var out_index: usize = 0;
    var out_z: usize = 0;
    while (out_z < output_side) : (out_z += 1) {
        const z_window_base = out_z * plane;
        var out_y: usize = 0;
        while (out_y < output_side) : (out_y += 1) {
            const yz_window_base = z_window_base + out_y * input_side;
            var out_x: usize = 0;
            while (out_x < output_side) : (out_x += 1) {
                const window_base = yz_window_base + out_x;
                var count: u32 = 0;

                var window_z: usize = 0;
                while (window_z < window_side) : (window_z += 1) {
                    const plane_base = window_base + window_z * plane;
                    var window_y: usize = 0;
                    while (window_y < window_side) : (window_y += 1) {
                        const row_base = plane_base + window_y * input_side;
                        var window_x: usize = 0;
                        while (window_x < window_side) : (window_x += 1) {
                            count += input_bytes[row_base + window_x];
                        }
                    }
                }

                count -= input_bytes[window_base + center_offset];
                output[out_index] = count >= solid_threshold;
                out_index += 1;
            }
        }
    }
}

fn cellularAutomataSmooth3dBoolShrinkRadius3BytesToBoolRows(
    output: []bool,
    input_bytes: []const u8,
    input_side: usize,
    output_side: usize,
    solid_threshold: u32,
) void {
    const threshold: u16 = @intCast(solid_threshold);
    const plane = input_side * input_side;
    const row_offsets = [7]usize{ 0, input_side, 2 * input_side, 3 * input_side, 4 * input_side, 5 * input_side, 6 * input_side };
    const z_offsets = [7]usize{ 0, plane, 2 * plane, 3 * plane, 4 * plane, 5 * plane, 6 * plane };

    var out_index: usize = 0;
    var out_z: usize = 0;
    while (out_z < output_side) : (out_z += 1) {
        const z_window_base = out_z * plane;
        var out_y: usize = 0;
        while (out_y < output_side) : (out_y += 1) {
            const yz_window_base = z_window_base + out_y * input_side;
            var out_x: usize = 0;
            while (out_x < output_side) : (out_x += 1) {
                const window_ptr = input_bytes.ptr + yz_window_base + out_x;
                var count: u16 = 0;
                count += solidByteRadius3FullPlaneSum(window_ptr + z_offsets[0], &row_offsets);
                count += solidByteRadius3FullPlaneSum(window_ptr + z_offsets[1], &row_offsets);
                count += solidByteRadius3FullPlaneSum(window_ptr + z_offsets[2], &row_offsets);
                count += solidByteRadius3CenterPlaneSum(window_ptr + z_offsets[3], &row_offsets);
                count += solidByteRadius3FullPlaneSum(window_ptr + z_offsets[4], &row_offsets);
                count += solidByteRadius3FullPlaneSum(window_ptr + z_offsets[5], &row_offsets);
                count += solidByteRadius3FullPlaneSum(window_ptr + z_offsets[6], &row_offsets);

                output[out_index] = count >= threshold;
                out_index += 1;
            }
        }
    }
}

inline fn solidByteSpanSum7U16(pointer: [*]const u8) u16 {
    return @as(u16, pointer[0]) + pointer[1] + pointer[2] + pointer[3] + pointer[4] + pointer[5] + pointer[6];
}

inline fn solidByteRadius3FullPlaneSum(input_ptr: [*]const u8, row_offsets: *const [7]usize) u16 {
    var count: u16 = 0;
    inline for (0..7) |row| {
        count += solidByteSpanSum7U16(input_ptr + row_offsets[row]);
    }
    return count;
}

inline fn solidByteRadius3CenterPlaneSum(input_ptr: [*]const u8, row_offsets: *const [7]usize) u16 {
    var count: u16 = 0;
    inline for (0..3) |row| {
        count += solidByteSpanSum7U16(input_ptr + row_offsets[row]);
    }

    const center_row = input_ptr + row_offsets[3];
    count += @as(u16, center_row[0]) + center_row[1] + center_row[2] + center_row[4] + center_row[5] + center_row[6];

    inline for (4..7) |row| {
        count += solidByteSpanSum7U16(input_ptr + row_offsets[row]);
    }
    return count;
}

fn cellularAutomataSmooth3dBoolShrinkRadius2BytesToBoolVector(
    output: []bool,
    input_bytes: []const u8,
    input_side: usize,
    output_side: usize,
    solid_threshold: u8,
) void {
    const output_count = output_side * output_side * output_side;
    const output_bytes = boolSliceBytes(output[0..output_count]);
    const plane = input_side * input_side;
    const output_plane = output_side * output_side;
    const threshold_v = @as(U8x16, @splat(solid_threshold));
    const one_v = @as(U8x16, @splat(@as(u8, 1)));
    const zero_v = @as(U8x16, @splat(@as(u8, 0)));
    const full_vector_end = output_side & ~(ByteLaneCount - 1);

    var z: usize = 0;
    while (z < output_side) : (z += 1) {
        const z_base = z * plane;
        const output_z_base = z * output_plane;
        var y: usize = 0;
        while (y < output_side) : (y += 1) {
            const yz_base = z_base + y * input_side;
            const output_row = output_z_base + y * output_side;
            var x: usize = 0;
            while (x < full_vector_end) : (x += ByteLaneCount) {
                const window_base = yz_base + x;
                var count_v = @as(U8x16, @splat(@as(u8, 0)));
                var center_v = @as(U8x16, @splat(@as(u8, 0)));

                comptime var window_z: usize = 0;
                inline while (window_z < 5) : (window_z += 1) {
                    comptime var window_y: usize = 0;
                    inline while (window_y < 5) : (window_y += 1) {
                        const row_base = window_base + window_z * plane + window_y * input_side;
                        comptime var window_x: usize = 0;
                        inline while (window_x < 5) : (window_x += 1) {
                            const value_v = loadU8x16(input_bytes.ptr + row_base + window_x);
                            count_v +%= value_v;
                            if (window_z == 2 and window_y == 2 and window_x == 2) center_v = value_v;
                        }
                    }
                }

                storeU8x16(output_bytes.ptr + output_row + x, @select(u8, (count_v -% center_v) >= threshold_v, one_v, zero_v));
            }

            while (x < output_side) : (x += 1) {
                const window_base = yz_base + x;
                var count: u8 = 0;
                var center: u8 = 0;
                comptime var tail_z: usize = 0;
                inline while (tail_z < 5) : (tail_z += 1) {
                    comptime var tail_y: usize = 0;
                    inline while (tail_y < 5) : (tail_y += 1) {
                        const row_base = window_base + tail_z * plane + tail_y * input_side;
                        comptime var tail_x: usize = 0;
                        inline while (tail_x < 5) : (tail_x += 1) {
                            const value = input_bytes[row_base + tail_x];
                            count +%= value;
                            if (tail_z == 2 and tail_y == 2 and tail_x == 2) center = value;
                        }
                    }
                }

                output_bytes[output_row + x] = @intFromBool((count -% center) >= solid_threshold);
            }
        }
    }
}

inline fn loadU64Unaligned(pointer: [*]const u8) u64 {
    const value: *align(1) const u64 = @ptrCast(pointer);
    return value.*;
}

inline fn solidByteSpanSum(pointer: [*]const u8, len: usize) u32 {
    switch (len) {
        0 => return 0,
        1 => return pointer[0],
        2 => return pointer[0] + pointer[1],
        3 => return pointer[0] + pointer[1] + pointer[2],
        4 => return pointer[0] + pointer[1] + pointer[2] + pointer[3],
        5 => return pointer[0] + pointer[1] + pointer[2] + pointer[3] + pointer[4],
        6 => return pointer[0] + pointer[1] + pointer[2] + pointer[3] + pointer[4] + pointer[5],
        7 => return pointer[0] + pointer[1] + pointer[2] + pointer[3] + pointer[4] + pointer[5] + pointer[6],
        8 => return @intCast(@popCount(loadU64Unaligned(pointer))),
        10 => return @as(u32, @intCast(@popCount(loadU64Unaligned(pointer)))) + pointer[8] + pointer[9],
        11 => return @as(u32, @intCast(@popCount(loadU64Unaligned(pointer)))) + pointer[8] + pointer[9] + pointer[10],
        12 => return @as(u32, @intCast(@popCount(loadU64Unaligned(pointer)))) + pointer[8] + pointer[9] + pointer[10] + pointer[11],
        13 => return @as(u32, @intCast(@popCount(loadU64Unaligned(pointer)))) + pointer[8] + pointer[9] + pointer[10] + pointer[11] + pointer[12],
        14 => return @as(u32, @intCast(@popCount(loadU64Unaligned(pointer)))) + pointer[8] + pointer[9] + pointer[10] + pointer[11] + pointer[12] + pointer[13],
        16 => return @as(u32, @intCast(@popCount(loadU64Unaligned(pointer)))) + @as(u32, @intCast(@popCount(loadU64Unaligned(pointer + 8)))),
        else => {},
    }

    var count: u32 = 0;
    var index: usize = 0;
    while (index + 8 <= len) : (index += 8) {
        count += @intCast(@popCount(loadU64Unaligned(pointer + index)));
    }
    while (index < len) : (index += 1) {
        count += pointer[index];
    }
    return count;
}

inline fn solidByteRowsSum(
    input_bytes: [*]const u8,
    input_side: usize,
    row_start: usize,
    row_count: usize,
    window_side: usize,
) u32 {
    var count: u32 = 0;
    var row_base = row_start;
    var row: usize = 0;
    while (row < row_count) : ({
        row += 1;
        row_base += input_side;
    }) {
        count += solidByteSpanSum(input_bytes + row_base, window_side);
    }
    return count;
}

inline fn solidByteSpanSumLargeWindow(pointer: [*]const u8, len: usize) u32 {
    switch (len) {
        9 => return @as(u32, @intCast(@popCount(loadU64Unaligned(pointer)))) + pointer[8],
        15 => return @as(u32, @intCast(@popCount(loadU64Unaligned(pointer)))) +
            pointer[8] + pointer[9] + pointer[10] + pointer[11] + pointer[12] + pointer[13] + pointer[14],
        17 => return @as(u32, @intCast(@popCount(loadU64Unaligned(pointer)))) +
            @as(u32, @intCast(@popCount(loadU64Unaligned(pointer + 8)))) +
            pointer[16],
        else => return solidByteSpanSum(pointer, len),
    }
}

inline fn solidByteRowsSumLargeWindow(
    row_ptr_start: [*]const u8,
    input_side: usize,
    row_count: usize,
    window_side: usize,
) u32 {
    var count: u32 = 0;
    var row_ptr = row_ptr_start;
    var row: usize = 0;
    while (row < row_count) : ({
        row += 1;
        row_ptr += input_side;
    }) {
        count += solidByteSpanSumLargeWindow(row_ptr, window_side);
    }
    return count;
}

fn cellularAutomataSmooth3dBoolShrinkBytesToBoolWideSwar(
    output: []bool,
    input_bytes: []const u8,
    input_side: usize,
    output_side: usize,
    radius: usize,
    window_side: usize,
    solid_threshold: u32,
) void {
    if (solid_threshold <= window_side) {
        cellularAutomataSmooth3dBoolShrinkBytesToBoolWideSwarBoundedLow(
            output,
            input_bytes,
            input_side,
            output_side,
            radius,
            window_side,
            solid_threshold,
        );
        return;
    }

    if (radius >= 8) {
        cellularAutomataSmooth3dBoolShrinkBytesToBoolWideSwarLargeRadius(
            output,
            input_bytes,
            input_side,
            output_side,
            radius,
            window_side,
            solid_threshold,
        );
        return;
    }

    const plane = input_side * input_side;
    var out_index: usize = 0;
    var out_z: usize = 0;
    while (out_z < output_side) : (out_z += 1) {
        const z_window_base = out_z * plane;
        var out_y: usize = 0;
        while (out_y < output_side) : (out_y += 1) {
            const yz_window_base = z_window_base + out_y * input_side;
            var out_x: usize = 0;
            while (out_x < output_side) : (out_x += 1) {
                const window_base = yz_window_base + out_x;
                var count: u32 = 0;

                var window_z: usize = 0;
                while (window_z < radius) : (window_z += 1) {
                    count += solidByteRowsSum(input_bytes.ptr, input_side, window_base + window_z * plane, window_side, window_side);
                }

                const center_plane = window_base + radius * plane;
                count += solidByteRowsSum(input_bytes.ptr, input_side, center_plane, radius, window_side);
                count += solidByteSpanSum(input_bytes.ptr + center_plane + radius * input_side, radius);
                count += solidByteSpanSum(input_bytes.ptr + center_plane + radius * input_side + radius + 1, radius);
                count += solidByteRowsSum(input_bytes.ptr, input_side, center_plane + (radius + 1) * input_side, radius, window_side);

                window_z = radius + 1;
                while (window_z < window_side) : (window_z += 1) {
                    count += solidByteRowsSum(input_bytes.ptr, input_side, window_base + window_z * plane, window_side, window_side);
                }

                output[out_index] = count >= solid_threshold;
                out_index += 1;
            }
        }
    }
}

fn cellularAutomataSmooth3dBoolShrinkBytesToBoolWideSwarBoundedLow(
    output: []bool,
    input_bytes: []const u8,
    input_side: usize,
    output_side: usize,
    radius: usize,
    window_side: usize,
    solid_threshold: u32,
) void {
    const plane = input_side * input_side;
    var out_index: usize = 0;
    var out_z: usize = 0;
    while (out_z < output_side) : (out_z += 1) {
        const z_window_base = out_z * plane;
        var out_y: usize = 0;
        while (out_y < output_side) : (out_y += 1) {
            const yz_window_base = z_window_base + out_y * input_side;
            var out_x: usize = 0;
            while (out_x < output_side) : (out_x += 1) {
                const window_ptr = input_bytes.ptr + yz_window_base + out_x;
                var count: u32 = 0;
                var solid = false;

                var window_z: usize = 0;
                while (window_z < radius and !solid) : (window_z += 1) {
                    count += solidByteRowsSumLargeWindow(window_ptr + window_z * plane, input_side, window_side, window_side);
                    solid = count >= solid_threshold;
                }

                if (!solid) {
                    const center_plane = window_ptr + radius * plane;
                    count += solidByteRowsSumLargeWindow(center_plane, input_side, radius, window_side);
                    solid = count >= solid_threshold;

                    if (!solid) {
                        const center_row = center_plane + radius * input_side;
                        count += solidByteSpanSumLargeWindow(center_row, radius);
                        count += solidByteSpanSumLargeWindow(center_row + radius + 1, radius);
                        solid = count >= solid_threshold;
                    }

                    if (!solid) {
                        count += solidByteRowsSumLargeWindow(center_plane + (radius + 1) * input_side, input_side, radius, window_side);
                        solid = count >= solid_threshold;
                    }

                    window_z = radius + 1;
                    while (window_z < window_side and !solid) : (window_z += 1) {
                        count += solidByteRowsSumLargeWindow(window_ptr + window_z * plane, input_side, window_side, window_side);
                        solid = count >= solid_threshold;
                    }
                }

                output[out_index] = solid;
                out_index += 1;
            }
        }
    }
}

fn cellularAutomataSmooth3dBoolShrinkBytesToBoolWideSwarLargeRadius(
    output: []bool,
    input_bytes: []const u8,
    input_side: usize,
    output_side: usize,
    radius: usize,
    window_side: usize,
    solid_threshold: u32,
) void {
    const plane = input_side * input_side;
    var out_index: usize = 0;
    var out_z: usize = 0;
    while (out_z < output_side) : (out_z += 1) {
        const z_window_base = out_z * plane;
        var out_y: usize = 0;
        while (out_y < output_side) : (out_y += 1) {
            const yz_window_base = z_window_base + out_y * input_side;
            var out_x: usize = 0;
            while (out_x < output_side) : (out_x += 1) {
                const window_ptr = input_bytes.ptr + yz_window_base + out_x;
                var count: u32 = 0;

                var window_z: usize = 0;
                while (window_z < radius) : (window_z += 1) {
                    count += solidByteRowsSumLargeWindow(window_ptr + window_z * plane, input_side, window_side, window_side);
                }

                const center_plane = window_ptr + radius * plane;
                count += solidByteRowsSumLargeWindow(center_plane, input_side, radius, window_side);
                const center_row = center_plane + radius * input_side;
                count += solidByteSpanSumLargeWindow(center_row, radius);
                count += solidByteSpanSumLargeWindow(center_row + radius + 1, radius);
                count += solidByteRowsSumLargeWindow(center_plane + (radius + 1) * input_side, input_side, radius, window_side);

                window_z = radius + 1;
                while (window_z < window_side) : (window_z += 1) {
                    count += solidByteRowsSumLargeWindow(window_ptr + window_z * plane, input_side, window_side, window_side);
                }

                output[out_index] = count >= solid_threshold;
                out_index += 1;
            }
        }
    }
}

fn countSolidNeighborsShrinkRowSpan(
    input: []const bool,
    input_side: usize,
    input_x: usize,
    input_y: usize,
    input_z: usize,
    radius: usize,
) u32 {
    const plane = input_side * input_side;
    const x_start = input_x - radius;
    const x_end = input_x + radius;
    const y_start = input_y - radius;
    const y_end = input_y + radius;
    const z_start = input_z - radius;
    const z_end = input_z + radius;
    var count: u32 = 0;

    var z = z_start;
    while (z <= z_end) : (z += 1) {
        const plane_base = z * plane;
        var y = y_start;
        while (y <= y_end) : (y += 1) {
            const row_base = plane_base + y * input_side;
            var x = x_start;
            while (x <= x_end) : (x += 1) {
                count += @intFromBool(input[row_base + x]);
            }
        }
    }

    return count - @intFromBool(input[input_z * plane + input_y * input_side + input_x]);
}

fn anySolidNeighborShrinkRowSpan(
    input: []const bool,
    input_side: usize,
    input_x: usize,
    input_y: usize,
    input_z: usize,
    radius: usize,
) bool {
    const plane = input_side * input_side;
    const center = input_z * plane + input_y * input_side + input_x;
    const x_start = input_x - radius;
    const x_end = input_x + radius;
    const y_start = input_y - radius;
    const y_end = input_y + radius;
    const z_start = input_z - radius;
    const z_end = input_z + radius;

    var z = z_start;
    while (z <= z_end) : (z += 1) {
        const plane_base = z * plane;
        var y = y_start;
        while (y <= y_end) : (y += 1) {
            const row_base = plane_base + y * input_side;
            var x = x_start;
            while (x <= x_end) : (x += 1) {
                const index = row_base + x;
                if (index != center and input[index]) return true;
            }
        }
    }

    return false;
}

fn allSolidNeighborsShrinkRowSpan(
    input: []const bool,
    input_side: usize,
    input_x: usize,
    input_y: usize,
    input_z: usize,
    radius: usize,
) bool {
    const plane = input_side * input_side;
    const center = input_z * plane + input_y * input_side + input_x;
    const x_start = input_x - radius;
    const x_end = input_x + radius;
    const y_start = input_y - radius;
    const y_end = input_y + radius;
    const z_start = input_z - radius;
    const z_end = input_z + radius;

    var z = z_start;
    while (z <= z_end) : (z += 1) {
        const plane_base = z * plane;
        var y = y_start;
        while (y <= y_end) : (y += 1) {
            const row_base = plane_base + y * input_side;
            var x = x_start;
            while (x <= x_end) : (x += 1) {
                const index = row_base + x;
                if (index != center and !input[index]) return false;
            }
        }
    }

    return true;
}

const Chunk3dConstantMode = enum {
    none,
    all_false,
    all_true,
};

fn chunk3dConstantMode(options: GenerateOptions, fill_threshold: u32) Chunk3dConstantMode {
    if (options.iterations == 0) {
        if (fill_threshold == 0) return .all_false;
        if (fill_threshold == std.math.maxInt(u32)) return .all_true;
        return .none;
    }

    if (options.solid_threshold == 0) return .all_true;
    if (fill_threshold == 0) {
        return .all_false;
    }
    if (fill_threshold == std.math.maxInt(u32)) return .all_true;
    return .none;
}

fn cellularAutomataChunk3dConstantOutput(out_chunk: []bool, mode: Chunk3dConstantMode) void {
    switch (mode) {
        .all_false => @memset(out_chunk, false),
        .all_true => @memset(out_chunk, true),
        .none => unreachable,
    }
}

pub fn cellularAutomataChunk3d(allocator: std.mem.Allocator, out: []bool, options: GenerateOptions) Error!void {
    try validateGenerateOptions(options);
    const layout = try cellularAutomataChunk3dLayout(options);
    if (out.len < layout.chunk_count) return error.BufferTooSmall;
    const out_chunk = out[0..layout.chunk_count];

    const constant_mode = chunk3dConstantMode(options, layout.fill_threshold);
    if (constant_mode != .none) {
        cellularAutomataChunk3dConstantOutput(out_chunk, constant_mode);
        return;
    }

    if (options.iterations == 0) {
        cellularAutomataRandomSolidGrid3dBytesThresholdTrustedChunk(
            boolSliceBytes(out_chunk),
            options.chunk_size,
            layout.origin_x_bits,
            layout.origin_y_bits,
            layout.origin_z_bits,
            layout.origin_x_nowrap,
            layout.origin_y_nowrap,
            layout.origin_z_nowrap,
            options.seed,
            layout.fill_threshold,
        );
        return;
    }

    if (options.neighborhood_radius == 1) {
        return cellularAutomataChunk3dBytes(allocator, out_chunk, options, layout);
    }

    return cellularAutomataChunk3dBool(allocator, out_chunk, options, layout);
}

pub fn cellularAutomataChunk3dScratchByteCount(options: GenerateOptions) Error!usize {
    try validateGenerateOptions(options);
    if (options.iterations > 0 and options.solid_threshold == 0) return 0;
    if (chunk3dConstantMode(options, cellularAutomataFillThreshold(options.fill_percent)) != .none) return 0;
    return (try cellularAutomataChunk3dScratchShape(options)).buffer_count;
}

const ChunkScratchShape = struct {
    temp_count: usize,
    next_count: usize,
    buffer_count: usize,
};

fn cellularAutomataChunk3dScratchShape(options: GenerateOptions) Error!ChunkScratchShape {
    if (options.iterations == 0) return .{
        .temp_count = 0,
        .next_count = 0,
        .buffer_count = 0,
    };

    const padding = try cellularAutomataRequiredPadding(options.iterations, options.neighborhood_radius);
    const temp_side = try cellularAutomataTemporarySide(options.chunk_size, padding);
    const temp_count = try cellularAutomataCellCount(temp_side);
    const next_count = if (options.iterations > 1) blk: {
        const shrink_side = if (options.neighborhood_radius == 1)
            temp_side - 2
        else
            temp_side - 2 * options.neighborhood_radius;
        break :blk try cellularAutomataCellCount(shrink_side);
    } else 0;
    return .{
        .temp_count = temp_count,
        .next_count = next_count,
        .buffer_count = std.math.add(usize, temp_count, next_count) catch return error.CellCountOverflow,
    };
}

pub fn cellularAutomataChunk3dWithScratch(out: []bool, scratch: []u8, options: GenerateOptions) Error!void {
    try validateGenerateOptions(options);
    const layout = try cellularAutomataChunk3dLayout(options);
    if (out.len < layout.chunk_count) return error.BufferTooSmall;
    const out_chunk = out[0..layout.chunk_count];

    const constant_mode = chunk3dConstantMode(options, layout.fill_threshold);
    if (constant_mode != .none) {
        cellularAutomataChunk3dConstantOutput(out_chunk, constant_mode);
        return;
    }

    if (options.iterations == 0) {
        cellularAutomataRandomSolidGrid3dBytesThresholdTrustedChunk(
            boolSliceBytes(out_chunk),
            options.chunk_size,
            layout.origin_x_bits,
            layout.origin_y_bits,
            layout.origin_z_bits,
            layout.origin_x_nowrap,
            layout.origin_y_nowrap,
            layout.origin_z_nowrap,
            options.seed,
            layout.fill_threshold,
        );
        return;
    }

    if (options.neighborhood_radius == 1) {
        return cellularAutomataChunk3dBytesWithScratch(out_chunk, scratch, options, layout);
    }

    return cellularAutomataChunk3dBoolWithScratch(out_chunk, scratch, options, layout);
}

const ChunkLayout = struct {
    temp_side: usize,
    chunk_count: usize,
    temp_count: usize,
    origin_x_bits: u32,
    origin_y_bits: u32,
    origin_z_bits: u32,
    origin_x_nowrap: bool,
    origin_y_nowrap: bool,
    origin_z_nowrap: bool,
    fill_threshold: u32,
};

const ChunkAxisOriginBits = struct {
    bits: u32,
    nowrap_until_temp_end: bool,
};

fn chunkAxisOriginBits(chunk_coord: i64, chunk_size_i64: i64, padding_i64: i64, max_offset: usize) Error!ChunkAxisOriginBits {
    const world_start = if (chunk_coord == 0) 0 else try checkedMulI64(chunk_coord, chunk_size_i64);
    const origin = try checkedSubI64(world_start, padding_i64);
    _ = try addOffset(origin, max_offset);
    const bits = coordBits32(origin);
    const nowrap_until_temp_end = max_offset <= std.math.maxInt(u32) and
        bits <= std.math.maxInt(u32) - @as(u32, @intCast(max_offset));
    return .{
        .bits = bits,
        .nowrap_until_temp_end = nowrap_until_temp_end,
    };
}

inline fn cellularAutomataChunk3dLayout(options: GenerateOptions) Error!ChunkLayout {
    const padding = try cellularAutomataRequiredPadding(options.iterations, options.neighborhood_radius);
    const temp_side = try cellularAutomataTemporarySide(options.chunk_size, padding);
    const chunk_count = try cellularAutomataCellCount(options.chunk_size);
    const temp_count = try cellularAutomataCellCount(temp_side);

    const chunk_size_i64 = try usizeToI64(options.chunk_size);
    const padding_i64 = try usizeToI64(padding);
    const max_offset = temp_side - 1;
    const origin_x = try chunkAxisOriginBits(options.chunk_x, chunk_size_i64, padding_i64, max_offset);
    const origin_y = try chunkAxisOriginBits(options.chunk_y, chunk_size_i64, padding_i64, max_offset);
    const origin_z = try chunkAxisOriginBits(options.chunk_z, chunk_size_i64, padding_i64, max_offset);

    return .{
        .temp_side = temp_side,
        .chunk_count = chunk_count,
        .temp_count = temp_count,
        .origin_x_bits = origin_x.bits,
        .origin_y_bits = origin_y.bits,
        .origin_z_bits = origin_z.bits,
        .origin_x_nowrap = origin_x.nowrap_until_temp_end,
        .origin_y_nowrap = origin_y.nowrap_until_temp_end,
        .origin_z_nowrap = origin_z.nowrap_until_temp_end,
        .fill_threshold = cellularAutomataFillThreshold(options.fill_percent),
    };
}

fn cellularAutomataChunk3dBool(
    allocator: std.mem.Allocator,
    out_chunk: []bool,
    options: GenerateOptions,
    layout: ChunkLayout,
) Error!void {
    const next_count = if (options.iterations > 1) blk: {
        const shrink_side = layout.temp_side - 2 * options.neighborhood_radius;
        break :blk try cellularAutomataCellCount(shrink_side);
    } else 0;
    const buffer_count = std.math.add(usize, layout.temp_count, next_count) catch return error.CellCountOverflow;
    if (buffer_count <= MaxStackChunkScratchBytes) {
        return cellularAutomataChunk3dBoolWithStack(out_chunk, options, layout, buffer_count);
    }

    const current = try allocator.alloc(bool, layout.temp_count);
    defer allocator.free(current);
    const next: []bool = if (next_count == 0) &.{} else try allocator.alloc(bool, next_count);
    defer if (next.len != 0) allocator.free(next);

    return cellularAutomataChunk3dBoolWithBuffers(out_chunk, options, layout, current, next);
}

noinline fn cellularAutomataChunk3dBoolWithStack(
    out_chunk: []bool,
    options: GenerateOptions,
    layout: ChunkLayout,
    buffer_count: usize,
) Error!void {
    var stack_buffer: [MaxStackChunkScratchBytes]u8 = undefined;
    return cellularAutomataChunk3dBoolWithScratch(out_chunk, stack_buffer[0..buffer_count], options, layout);
}

fn cellularAutomataChunk3dBoolWithScratch(
    out_chunk: []bool,
    scratch: []u8,
    options: GenerateOptions,
    layout: ChunkLayout,
) Error!void {
    const next_count = if (options.iterations > 1) blk: {
        const shrink_side = layout.temp_side - 2 * options.neighborhood_radius;
        break :blk try cellularAutomataCellCount(shrink_side);
    } else 0;
    const buffer_count = std.math.add(usize, layout.temp_count, next_count) catch return error.CellCountOverflow;
    if (scratch.len < buffer_count) return error.BufferTooSmall;

    const current = boolSliceFromBytes(scratch[0..layout.temp_count]);
    const next: []bool = if (next_count == 0)
        &.{}
    else
        boolSliceFromBytes(scratch[layout.temp_count..buffer_count]);

    return cellularAutomataChunk3dBoolWithBuffers(out_chunk, options, layout, current, next);
}

fn cellularAutomataChunk3dBoolWithBuffers(
    out_chunk: []bool,
    options: GenerateOptions,
    layout: ChunkLayout,
    current: []bool,
    next: []bool,
) Error!void {
    if (current.len < layout.temp_count) return error.BufferTooSmall;
    const first_output_side = layout.temp_side - 2 * options.neighborhood_radius;
    const next_count = if (options.iterations > 1)
        try cellularAutomataCellCount(first_output_side)
    else
        0;
    if (next.len < next_count) return error.BufferTooSmall;

    cellularAutomataRandomSolidGrid3dBoolThresholdTrustedChunk(
        current[0..layout.temp_count],
        layout.temp_side,
        layout.origin_x_bits,
        layout.origin_y_bits,
        layout.origin_z_bits,
        layout.origin_x_nowrap,
        layout.origin_y_nowrap,
        layout.origin_z_nowrap,
        options.seed,
        layout.fill_threshold,
    );

    const iterations = options.iterations;
    const radius = options.neighborhood_radius;
    const radius_step = radius * 2;
    const threshold = options.solid_threshold;

    var current_active = current[0..layout.temp_count];
    var input_side = layout.temp_side;
    var iteration: usize = 0;
    while (iteration + 2 < iterations) {
        var output_side = input_side - radius_step;
        var output_count = try cellularAutomataCellCount(output_side);
        try cellularAutomataSmooth3dBoolShrinkToBool(
            next[0..output_count],
            current_active,
            input_side,
            output_side,
            radius,
            threshold,
        );

        iteration += 1;
        input_side = output_side;
        const next_active = next[0..output_count];

        output_side = input_side - radius_step;
        output_count = try cellularAutomataCellCount(output_side);
        try cellularAutomataSmooth3dBoolShrinkToBool(
            current[0..output_count],
            next_active,
            input_side,
            output_side,
            radius,
            threshold,
        );

        iteration += 1;
        input_side = output_side;
        current_active = current[0..output_count];
    }

    if (iteration + 1 < iterations) {
        const output_side = input_side - radius_step;
        const output_count = try cellularAutomataCellCount(output_side);
        try cellularAutomataSmooth3dBoolShrinkToBool(
            next[0..output_count],
            current_active,
            input_side,
            output_side,
            radius,
            threshold,
        );

        input_side = output_side;
        const final_output_side = input_side - radius_step;
        std.debug.assert(final_output_side == options.chunk_size);
        try cellularAutomataSmooth3dBoolShrinkToBool(
            out_chunk,
            next[0..output_count],
            input_side,
            final_output_side,
            radius,
            threshold,
        );
        return;
    }

    const final_output_side = input_side - radius_step;
    std.debug.assert(final_output_side == options.chunk_size);
    try cellularAutomataSmooth3dBoolShrinkToBool(
        out_chunk,
        current_active,
        input_side,
        final_output_side,
        radius,
        threshold,
    );
}

inline fn cellularAutomataChunk3dBytesBufferCount(temp_side: usize, temp_count: usize, iterations: usize) Error!usize {
    const next_count = if (iterations > 1)
        try cellularAutomataCellCount(temp_side - 2)
    else
        0;
    return std.math.add(usize, temp_count, next_count) catch error.CellCountOverflow;
}

fn cellularAutomataChunk3dBytes(
    allocator: std.mem.Allocator,
    out_chunk: []bool,
    options: GenerateOptions,
    layout: ChunkLayout,
) Error!void {
    const buffer_count = try cellularAutomataChunk3dBytesBufferCount(layout.temp_side, layout.temp_count, options.iterations);
    if (buffer_count <= MaxStackChunkScratchBytes) {
        return cellularAutomataChunk3dBytesWithStack(out_chunk, options, layout, buffer_count);
    }

    const buffers = try allocator.alloc(u8, buffer_count);
    defer allocator.free(buffers);
    return cellularAutomataChunk3dBytesWithBuffers(out_chunk, options, layout, buffers, buffer_count);
}

noinline fn cellularAutomataChunk3dBytesWithStack(
    out_chunk: []bool,
    options: GenerateOptions,
    layout: ChunkLayout,
    buffer_count: usize,
) Error!void {
    var stack_buffer: [MaxStackChunkScratchBytes]u8 = undefined;
    return cellularAutomataChunk3dBytesWithBuffers(out_chunk, options, layout, stack_buffer[0..buffer_count], buffer_count);
}

fn cellularAutomataChunk3dBytesWithScratch(
    out_chunk: []bool,
    scratch: []u8,
    options: GenerateOptions,
    layout: ChunkLayout,
) Error!void {
    const buffer_count = try cellularAutomataChunk3dBytesBufferCount(layout.temp_side, layout.temp_count, options.iterations);
    if (scratch.len < buffer_count) return error.BufferTooSmall;
    return cellularAutomataChunk3dBytesWithBuffers(out_chunk, options, layout, scratch[0..buffer_count], buffer_count);
}

fn cellularAutomataChunk3dBytesTryStableCopy(
    out_bytes: []u8,
    output: []const u8,
    input: []const u8,
    input_side: usize,
    output_side: usize,
    remaining_after_this_pass: usize,
    chunk_size: usize,
) bool {
    std.debug.assert(remaining_after_this_pass >= 2);
    std.debug.assert(output.len >= 4096);
    std.debug.assert(output_side == chunk_size + remaining_after_this_pass * 2);
    if (cellularAutomataBytesEqualCenteredCrop(output, input, input_side, output_side, 1)) {
        cellularAutomataCopyCenteredCropBytes(
            out_bytes,
            output,
            output_side,
            chunk_size,
            remaining_after_this_pass,
        );
        return true;
    }
    return false;
}

const Chunk3dBytesStableMinimumSide: usize = 16;

inline fn cellularAutomataChunk3dCube(side: usize) usize {
    return side * side * side;
}

inline fn cellularAutomataChunk3dBytesWithBuffers(
    out_chunk: []bool,
    options: GenerateOptions,
    layout: ChunkLayout,
    buffers: []u8,
    buffer_count: usize,
) Error!void {
    if (buffers.len < buffer_count) {
        @branchHint(.unlikely);
        return error.BufferTooSmall;
    }

    const iterations = options.iterations;
    const temp_side = layout.temp_side;
    const temp_count = layout.temp_count;
    const final_side_from_layout = temp_side - iterations * 2;
    const final_count = cellularAutomataChunk3dCube(final_side_from_layout);
    const threshold: u32 = options.solid_threshold;

    std.debug.assert(iterations >= 1);
    std.debug.assert(temp_side >= 3);
    std.debug.assert(layout.chunk_count <= out_chunk.len);
    std.debug.assert(layout.temp_count == cellularAutomataChunk3dCube(temp_side));
    std.debug.assert(layout.chunk_count == final_count);
    std.debug.assert(final_side_from_layout == options.chunk_size);
    std.debug.assert(final_side_from_layout > 0);
    std.debug.assert(threshold <= 26);

    const out_bool = out_chunk[0..layout.chunk_count];

    std.debug.assert(buffer_count >= temp_count);
    const current_buffer = buffers[0..temp_count];

    {
        const seed = options.seed;
        const fill_threshold = layout.fill_threshold;
        const origin_x_bits = layout.origin_x_bits;
        const origin_y_bits = layout.origin_y_bits;
        const origin_z_bits = layout.origin_z_bits;
        const origin_x_nowrap = layout.origin_x_nowrap;
        const origin_y_nowrap = layout.origin_y_nowrap;
        const origin_z_nowrap = layout.origin_z_nowrap;

        cellularAutomataRandomSolidGrid3dBytesThresholdTrustedChunk(
            current_buffer,
            temp_side,
            origin_x_bits,
            origin_y_bits,
            origin_z_bits,
            origin_x_nowrap,
            origin_y_nowrap,
            origin_z_nowrap,
            seed,
            fill_threshold,
        );
    }

    if (iterations == 1) {
        std.debug.assert(final_side_from_layout == temp_side - 2);
        std.debug.assert(buffer_count == temp_count);
        const out_bytes = boolSliceBytes(out_bool);
        cellularAutomataSmooth3dBytesShrinkToBytesTrusted(out_bytes, current_buffer, temp_side, final_side_from_layout, threshold);
        return;
    }

    std.debug.assert(buffer_count > temp_count);
    const next_capacity = buffer_count - temp_count;
    const first_output_side = temp_side - 2;
    std.debug.assert(next_capacity >= cellularAutomataChunk3dCube(first_output_side));
    const next_buffer = buffers[temp_count..buffer_count];

    var src: []const u8 = current_buffer;
    var input_side = temp_side;
    var iteration: usize = 0;
    while (iteration + 2 < iterations) {
        const output_side_1 = input_side - 2;
        const output_count_1 = cellularAutomataChunk3dCube(output_side_1);
        const dst1 = next_buffer[0..output_count_1];
        cellularAutomataSmooth3dBytesShrinkToBytesTrusted(dst1, src, input_side, output_side_1, threshold);

        const iteration_after_first = iteration + 1;
        const can_stable_time_1 = iteration_after_first + 2 <= iterations;
        const can_stable_size_1 = output_side_1 >= Chunk3dBytesStableMinimumSide;
        if (can_stable_time_1 and can_stable_size_1) {
            const out_bytes = boolSliceBytes(out_bool);
            if (cellularAutomataChunk3dBytesTryStableCopy(out_bytes, dst1, src, input_side, output_side_1, iterations - iteration_after_first, final_side_from_layout)) {
                @branchHint(.unlikely);
                return;
            }
        }

        const output_side_2 = output_side_1 - 2;
        const output_count_2 = cellularAutomataChunk3dCube(output_side_2);
        const current_out = current_buffer[0..output_count_2];
        cellularAutomataSmooth3dBytesShrinkToBytesTrusted(current_out, dst1, output_side_1, output_side_2, threshold);

        const iteration_after_second = iteration + 2;
        const can_stable_time_2 = iteration_after_second + 2 <= iterations;
        const can_stable_size_2 = output_side_2 >= Chunk3dBytesStableMinimumSide;
        if (can_stable_time_2 and can_stable_size_2) {
            const out_bytes = boolSliceBytes(out_bool);
            if (cellularAutomataChunk3dBytesTryStableCopy(out_bytes, current_out, dst1, output_side_1, output_side_2, iterations - iteration_after_second, final_side_from_layout)) {
                @branchHint(.unlikely);
                return;
            }
        }

        iteration = iteration_after_second;
        input_side = output_side_2;
        src = current_out;
    }

    std.debug.assert(iterations - iteration <= 2);
    switch (iterations - iteration) {
        1 => {
            std.debug.assert(input_side == final_side_from_layout + 2);
            const out_bytes = boolSliceBytes(out_bool);
            cellularAutomataSmooth3dBytesShrinkToBytesTrusted(out_bytes, src, input_side, final_side_from_layout, threshold);
        },
        2 => {
            const pre_final_side = final_side_from_layout + 2;
            std.debug.assert(input_side == pre_final_side + 2);
            const pre_final_count = cellularAutomataChunk3dCube(pre_final_side);
            const pre_final = next_buffer[0..pre_final_count];
            cellularAutomataSmooth3dBytesShrinkToBytesTrusted(pre_final, src, input_side, pre_final_side, threshold);

            const out_bytes = boolSliceBytes(out_bool);
            cellularAutomataSmooth3dBytesShrinkToBytesTrusted(out_bytes, pre_final, pre_final_side, final_side_from_layout, threshold);
        },
        else => unreachable,
    }
}

fn validateGenerateOptions(options: GenerateOptions) Error!void {
    if (options.chunk_size == 0) return error.InvalidChunkSize;
    try validateFillPercent(options.fill_percent);
    if (options.iterations > 0) {
        try validateSmoothOptions(options.chunk_size, .{
            .neighborhood_radius = options.neighborhood_radius,
            .solid_threshold = options.solid_threshold,
            .boundary_is_solid = options.boundary_is_solid,
        });
    }
}

fn validateFillPercent(fill_percent: f32) Error!void {
    if (!std.math.isFinite(fill_percent) or fill_percent < 0.0 or fill_percent > 1.0) {
        return error.InvalidFillPercent;
    }
}

fn validateSmoothOptions(side: usize, options: SmoothOptions) Error!void {
    if (side == 0) return error.InvalidGridSide;
    if (options.neighborhood_radius == 0) return error.InvalidNeighborhoodRadius;
    _ = try usizeToIsize(side);
    _ = try usizeToIsize(options.neighborhood_radius);

    const max_neighbors = try maxNeighborCount(options.neighborhood_radius);
    if (options.solid_threshold > max_neighbors) return error.InvalidThreshold;
}

fn maxNeighborCount(radius: usize) Error!u32 {
    const doubled_radius = std.math.mul(usize, radius, 2) catch return error.CellCountOverflow;
    const side = std.math.add(usize, doubled_radius, 1) catch return error.CellCountOverflow;
    const count = try cellularAutomataCellCount(side);
    const neighbor_count = count - 1;
    if (neighbor_count > std.math.maxInt(u32)) return error.CellCountOverflow;
    return @intCast(neighbor_count);
}

fn countSolidNeighborsUnchecked(
    grid: []const bool,
    side: usize,
    x: usize,
    y: usize,
    z: usize,
    radius: isize,
    boundary_is_solid: bool,
) u32 {
    const side_i: isize = @intCast(side);
    const x_i: isize = @intCast(x);
    const y_i: isize = @intCast(y);
    const z_i: isize = @intCast(z);
    var count: u32 = 0;

    var dz = -radius;
    while (dz <= radius) : (dz += 1) {
        var dy = -radius;
        while (dy <= radius) : (dy += 1) {
            var dx = -radius;
            while (dx <= radius) : (dx += 1) {
                if (dx == 0 and dy == 0 and dz == 0) continue;

                const nx = x_i + dx;
                const ny = y_i + dy;
                const nz = z_i + dz;
                if (nx < 0 or ny < 0 or nz < 0 or nx >= side_i or ny >= side_i or nz >= side_i) {
                    if (boundary_is_solid) count += 1;
                } else if (grid[cellularAutomataIndex3d(@intCast(nx), @intCast(ny), @intCast(nz), side)]) {
                    count += 1;
                }
            }
        }
    }

    return count;
}

fn countSolidNeighborsByteUnchecked(
    grid: []const u8,
    side: usize,
    x: usize,
    y: usize,
    z: usize,
    boundary_is_solid: bool,
) u8 {
    const side_i: isize = @intCast(side);
    const x_i: isize = @intCast(x);
    const y_i: isize = @intCast(y);
    const z_i: isize = @intCast(z);
    var count: u8 = 0;

    var dz: isize = -1;
    while (dz <= 1) : (dz += 1) {
        var dy: isize = -1;
        while (dy <= 1) : (dy += 1) {
            var dx: isize = -1;
            while (dx <= 1) : (dx += 1) {
                if (dx == 0 and dy == 0 and dz == 0) continue;

                const nx = x_i + dx;
                const ny = y_i + dy;
                const nz = z_i + dz;
                if (nx < 0 or ny < 0 or nz < 0 or nx >= side_i or ny >= side_i or nz >= side_i) {
                    if (boundary_is_solid) count += 1;
                } else {
                    count += grid[cellularAutomataIndex3d(@intCast(nx), @intCast(ny), @intCast(nz), side)];
                }
            }
        }
    }

    return count;
}

fn addOffset(origin: i64, offset: usize) Error!i64 {
    return std.math.add(i64, origin, try usizeToI64(offset)) catch error.CoordinateOverflow;
}

fn checkedSubI64(a: i64, b: i64) Error!i64 {
    return std.math.sub(i64, a, b) catch error.CoordinateOverflow;
}

fn checkedMulI64(a: i64, b: i64) Error!i64 {
    return std.math.mul(i64, a, b) catch error.CoordinateOverflow;
}

fn usizeToI64(value: usize) Error!i64 {
    if (value > @as(usize, @intCast(std.math.maxInt(i64)))) return error.CoordinateOverflow;
    return @intCast(value);
}

fn usizeToIsize(value: usize) Error!isize {
    if (value > @as(usize, @intCast(std.math.maxInt(isize)))) return error.CellCountOverflow;
    return @intCast(value);
}

test "required padding is iterations times neighborhood radius" {
    try std.testing.expectEqual(@as(usize, 5), try cellularAutomataRequiredPadding(5, 1));
    try std.testing.expectEqual(@as(usize, 8), try cellularAutomataRequiredPadding(4, 2));
    try std.testing.expectEqual(@as(usize, 0), try cellularAutomataRequiredPadding(0, 2));
}

test "random solid is deterministic for global coordinates" {
    const a = cellularAutomataRandomSolid(100, 20, 50, 1337, 0.48);
    const b = cellularAutomataRandomSolid(100, 20, 50, 1337, 0.48);
    try std.testing.expectEqual(a, b);
    try std.testing.expect(!cellularAutomataRandomSolid(-3, 4, 5, 1337, 0.0));
    try std.testing.expect(cellularAutomataRandomSolid(-3, 4, 5, 1337, 1.0));
}

test "count solid neighbors supports solid temporary boundaries" {
    const side: usize = 3;
    var grid = [_]bool{false} ** (side * side * side);
    const count = try cellularAutomataSolidNeighborCount(grid[0..], side, 0, 0, 0, .{
        .neighborhood_radius = 1,
        .solid_threshold = 14,
        .boundary_is_solid = true,
    });
    try std.testing.expectEqual(@as(u32, 19), count);
}

test "byte random fill matches scalar bool fill" {
    const Side: usize = 19;
    const Count: usize = Side * Side * Side;
    const origin = GridOrigin{ .x = -37, .y = 11, .z = -5 };
    const seed: u32 = 42;
    const fill_percent: f32 = 0.49;

    var bool_grid = [_]bool{false} ** Count;
    var byte_grid = [_]u8{0} ** Count;
    try cellularAutomataRandomSolidGrid3d(bool_grid[0..], Side, origin, seed, fill_percent);
    try cellularAutomataRandomSolidGrid3dBytes(byte_grid[0..], Side, origin, seed, fill_percent);

    for (bool_grid, byte_grid) |solid, byte| {
        try std.testing.expectEqual(solid, byte != 0);
    }
}

test "trusted bool random fill matches scalar bool fill" {
    const Side: usize = 19;
    const Count: usize = Side * Side * Side;
    const origin = GridOrigin{ .x = -37, .y = 11, .z = -5 };
    const seed: u32 = 42;
    const fill_percent: f32 = 0.49;
    var expected = [_]bool{false} ** Count;
    var trusted = [_]bool{false} ** Count;

    try cellularAutomataRandomSolidGrid3d(expected[0..], Side, origin, seed, fill_percent);
    cellularAutomataRandomSolidGrid3dBoolThresholdTrusted(
        trusted[0..],
        Side,
        coordBits32(origin.x),
        coordBits32(origin.y),
        coordBits32(origin.z),
        seed,
        cellularAutomataFillThreshold(fill_percent),
    );

    try std.testing.expectEqualSlices(bool, expected[0..], trusted[0..]);
}

test "chunk trusted random fill nowrap and wrap fallback match baseline" {
    const Side: usize = 18;
    const Count: usize = Side * Side * Side;
    const threshold: u32 = 0x8123_4567;
    var baseline = [_]u8{0} ** Count;
    var candidate = [_]u8{0} ** Count;

    cellularAutomataRandomSolidGrid3dBytesThresholdTrusted(baseline[0..], Side, 10, 20, 30, 12345, threshold);
    cellularAutomataRandomSolidGrid3dBytesThresholdTrustedChunk(candidate[0..], Side, 10, 20, 30, true, true, true, 12345, threshold);
    try std.testing.expectEqualSlices(u8, baseline[0..], candidate[0..]);

    @memset(baseline[0..], 0);
    @memset(candidate[0..], 0);
    const wrapping_x = std.math.maxInt(u32) - 8;
    cellularAutomataRandomSolidGrid3dBytesThresholdTrusted(baseline[0..], Side, wrapping_x, 20, 30, 12345, threshold);
    cellularAutomataRandomSolidGrid3dBytesThresholdTrustedChunk(candidate[0..], Side, wrapping_x, 20, 30, false, true, true, 12345, threshold);
    try std.testing.expectEqualSlices(u8, baseline[0..], candidate[0..]);
}

test "fill sweep matches scalar bool fill candidates" {
    const Side: usize = 9;
    const Count: usize = Side * Side * Side;
    const fill_percents = [_]f32{ 0.0, 0.42, 0.57, 1.0 };
    const origin = GridOrigin{ .x = -37, .y = 11, .z = -5 };
    const seed: u32 = 42;

    var expected = [_]bool{false} ** Count;
    var sweep = [_]bool{false} ** (Count * fill_percents.len);
    try cellularAutomataRandomSolidGrid3dFillSweep(sweep[0..], Side, origin, seed, fill_percents[0..]);

    for (fill_percents, 0..) |fill_percent, candidate| {
        try cellularAutomataRandomSolidGrid3d(expected[0..], Side, origin, seed, fill_percent);
        for (expected, sweep[candidate * Count ..][0..Count]) |expected_solid, sweep_solid| {
            try std.testing.expectEqual(expected_solid, sweep_solid);
        }
    }
}

fn expectByteSmoothMatchesBool(comptime boundary_is_solid: bool) !void {
    const Side: usize = 19;
    const Count: usize = Side * Side * Side;
    var bool_input = [_]bool{false} ** Count;
    var bool_output = [_]bool{false} ** Count;
    var byte_input = [_]u8{0} ** Count;
    var byte_output = [_]u8{0} ** Count;

    for (0..Count) |index| {
        const x = index % Side;
        const yz = index / Side;
        const y = yz % Side;
        const z = yz / Side;
        const solid = ((x * 17 + y * 31 + z * 43 + index * 7) & 3) != 0;
        bool_input[index] = solid;
        byte_input[index] = if (solid) 1 else 0;
    }

    try cellularAutomataSmooth3d(bool_output[0..], bool_input[0..], Side, .{
        .neighborhood_radius = 1,
        .solid_threshold = 14,
        .boundary_is_solid = boundary_is_solid,
    });
    try cellularAutomataSmooth3dBytes(byte_output[0..], byte_input[0..], Side, .{
        .neighborhood_radius = 1,
        .solid_threshold = 14,
        .boundary_is_solid = boundary_is_solid,
    });

    for (bool_output, byte_output) |solid, byte| {
        try std.testing.expectEqual(solid, byte != 0);
    }
}

test "byte radius one smoothing matches scalar bool smoothing" {
    try expectByteSmoothMatchesBool(true);
    try expectByteSmoothMatchesBool(false);
}

test "byte shrink thresholds match generic neighbor count" {
    const InputSide: usize = 19;
    const OutputSide: usize = InputSide - 2;
    const InputCount: usize = InputSide * InputSide * InputSide;
    const OutputCount: usize = OutputSide * OutputSide * OutputSide;
    const thresholds = [_]u32{ 0, 1, 14, 26 };
    var input = [_]u8{0} ** InputCount;
    var output = [_]u8{0} ** OutputCount;

    for (0..InputCount) |index| {
        const x = index % InputSide;
        const yz = index / InputSide;
        const y = yz % InputSide;
        const z = yz / InputSide;
        input[index] = if (((x * 13 + y * 29 + z * 37 + index * 5) & 7) < 4) 1 else 0;
    }

    for (thresholds) |threshold| {
        @memset(output[0..], 0);
        try cellularAutomataSmooth3dBytesShrinkToBytes(output[0..], input[0..], InputSide, OutputSide, threshold);

        var z: usize = 1;
        while (z + 1 < InputSide) : (z += 1) {
            var y: usize = 1;
            while (y + 1 < InputSide) : (y += 1) {
                var x: usize = 1;
                while (x + 1 < InputSide) : (x += 1) {
                    const output_index = cellularAutomataIndex3d(x - 1, y - 1, z - 1, OutputSide);
                    const expected = countSolidNeighborsByteUnchecked(input[0..], InputSide, x, y, z, false) >= threshold;
                    try std.testing.expectEqual(expected, output[output_index] != 0);
                }
            }
        }
    }
}

test "byte shrink global count shortcuts produce constant output" {
    const InputSide: usize = 9;
    const OutputSide: usize = InputSide - 2;
    const InputCount: usize = InputSide * InputSide * InputSide;
    const OutputCount: usize = OutputSide * OutputSide * OutputSide;
    const threshold: u32 = 14;
    var input = [_]u8{0} ** InputCount;
    var output = [_]u8{1} ** OutputCount;

    for (0..13) |index| input[index * 3] = 1;
    try cellularAutomataSmooth3dBytesShrinkToBytes(output[0..], input[0..], InputSide, OutputSide, threshold);
    for (output) |solid| try std.testing.expectEqual(@as(u8, 0), solid);

    @memset(input[0..], 1);
    @memset(output[0..], 0);
    for (0..12) |index| input[index * 5] = 0;
    try cellularAutomataSmooth3dBytesShrinkToBytes(output[0..], input[0..], InputSide, OutputSide, threshold);
    for (output) |solid| try std.testing.expectEqual(@as(u8, 1), solid);
}

test "byte centered crop helpers match expected interior" {
    const InputSide: usize = 5;
    const OutputSide: usize = 3;
    const InputCount: usize = InputSide * InputSide * InputSide;
    const OutputCount: usize = OutputSide * OutputSide * OutputSide;
    var input = [_]u8{0} ** InputCount;
    var output = [_]u8{0} ** OutputCount;
    var out_bools = [_]bool{false} ** OutputCount;

    for (0..InputCount) |index| {
        input[index] = @intFromBool((index * 17 + 3) % 5 < 2);
    }

    var z: usize = 0;
    while (z < OutputSide) : (z += 1) {
        var y: usize = 0;
        while (y < OutputSide) : (y += 1) {
            const output_row = cellularAutomataIndex3d(0, y, z, OutputSide);
            const input_row = cellularAutomataIndex3d(1, y + 1, z + 1, InputSide);
            @memcpy(output[output_row..][0..OutputSide], input[input_row..][0..OutputSide]);
        }
    }

    try std.testing.expect(cellularAutomataBytesEqualCenteredCrop(output[0..], input[0..], InputSide, OutputSide, 1));
    output[0] ^= 1;
    try std.testing.expect(!cellularAutomataBytesEqualCenteredCrop(output[0..], input[0..], InputSide, OutputSide, 1));
    output[0] ^= 1;

    cellularAutomataCopyCenteredCropBytesToOut(out_bools[0..], input[0..], InputSide, OutputSide, 1);
    try std.testing.expectEqualSlices(u8, output[0..], boolSliceConstBytes(out_bools[0..]));
}

test "bool shrink threshold extremes and mid path match row span reference" {
    const Radius: usize = 2;
    const InputSide: usize = 8;
    const OutputSide: usize = InputSide - 2 * Radius;
    const InputCount: usize = InputSide * InputSide * InputSide;
    const OutputCount: usize = OutputSide * OutputSide * OutputSide;
    const max_neighbors = try maxNeighborCount(Radius);
    const thresholds = [_]u32{ 0, 1, 60, max_neighbors };
    var input = [_]bool{false} ** InputCount;
    var output = [_]bool{false} ** OutputCount;

    for (0..InputCount) |index| {
        const x = index % InputSide;
        const yz = index / InputSide;
        const y = yz % InputSide;
        const z = yz / InputSide;
        input[index] = ((x * 11 + y * 23 + z * 41 + index * 3) & 5) != 0;
    }

    for (thresholds) |threshold| {
        @memset(output[0..], false);
        try cellularAutomataSmooth3dBoolShrinkToBool(output[0..], input[0..], InputSide, OutputSide, Radius, threshold);

        var z: usize = 0;
        while (z < OutputSide) : (z += 1) {
            var y: usize = 0;
            while (y < OutputSide) : (y += 1) {
                const output_row = OutputSide * (y + OutputSide * z);
                var x: usize = 0;
                while (x < OutputSide) : (x += 1) {
                    const expected = if (threshold == 0)
                        true
                    else
                        countSolidNeighborsShrinkRowSpan(input[0..], InputSide, x + Radius, y + Radius, z + Radius, Radius) >= threshold;
                    try std.testing.expectEqual(expected, output[output_row + x]);
                }
            }
        }
    }
}

test "bool shrink global count shortcuts produce constant output" {
    const Radius: usize = 4;
    const InputSide: usize = 12;
    const OutputSide: usize = InputSide - 2 * Radius;
    const InputCount: usize = InputSide * InputSide * InputSide;
    const OutputCount: usize = OutputSide * OutputSide * OutputSide;
    const max_neighbors = try maxNeighborCount(Radius);
    var input = [_]bool{false} ** InputCount;
    var output = [_]bool{true} ** OutputCount;

    const false_threshold: u32 = 64;
    for (0..63) |index| input[index * 7] = true;
    try cellularAutomataSmooth3dBoolShrinkToBool(output[0..], input[0..], InputSide, OutputSide, Radius, false_threshold);
    for (output) |solid| try std.testing.expect(!solid);

    @memset(input[0..], true);
    @memset(output[0..], false);
    const true_threshold = max_neighbors - 5;
    for (0..5) |index| input[index * 11] = false;
    try cellularAutomataSmooth3dBoolShrinkToBool(output[0..], input[0..], InputSide, OutputSide, Radius, true_threshold);
    for (output) |solid| try std.testing.expect(solid);
}

test "bool shrink threshold inverse paths handle all empty and all full inputs" {
    const Radius: usize = 4;
    const InputSide: usize = 12;
    const OutputSide: usize = InputSide - 2 * Radius;
    const InputCount: usize = InputSide * InputSide * InputSide;
    const OutputCount: usize = OutputSide * OutputSide * OutputSide;
    const max_neighbors = try maxNeighborCount(Radius);

    var input = [_]bool{false} ** InputCount;
    var output = [_]bool{true} ** OutputCount;

    try cellularAutomataSmooth3dBoolShrinkToBool(output[0..], input[0..], InputSide, OutputSide, Radius, 1);
    for (output) |solid| try std.testing.expect(!solid);

    @memset(input[0..], true);
    @memset(output[0..], false);
    try cellularAutomataSmooth3dBoolShrinkToBool(output[0..], input[0..], InputSide, OutputSide, Radius, max_neighbors);
    for (output) |solid| try std.testing.expect(solid);
}

test "bool shrink inverse paths match clustered run reference" {
    const Radius: usize = 4;
    const InputSide: usize = 14;
    const OutputSide: usize = InputSide - 2 * Radius;
    const InputCount: usize = InputSide * InputSide * InputSide;
    const OutputCount: usize = OutputSide * OutputSide * OutputSide;
    const max_neighbors = try maxNeighborCount(Radius);

    var input = [_]bool{false} ** InputCount;
    var output = [_]bool{false} ** OutputCount;

    for (2..5) |x| input[cellularAutomataIndex3d(x, 4, 3, InputSide)] = true;
    for (4..8) |x| input[cellularAutomataIndex3d(x, 6, 8, InputSide)] = true;

    try cellularAutomataSmooth3dBoolShrinkToBool(output[0..], input[0..], InputSide, OutputSide, Radius, 1);

    var z: usize = 0;
    while (z < OutputSide) : (z += 1) {
        var y: usize = 0;
        while (y < OutputSide) : (y += 1) {
            const output_row = OutputSide * (y + OutputSide * z);
            var x: usize = 0;
            while (x < OutputSide) : (x += 1) {
                const expected = countSolidNeighborsShrinkRowSpan(input[0..], InputSide, x + Radius, y + Radius, z + Radius, Radius) >= 1;
                try std.testing.expectEqual(expected, output[output_row + x]);
            }
        }
    }

    @memset(input[0..], true);
    @memset(output[0..], false);
    for (2..5) |x| input[cellularAutomataIndex3d(x, 4, 3, InputSide)] = false;
    for (4..8) |x| input[cellularAutomataIndex3d(x, 6, 8, InputSide)] = false;

    try cellularAutomataSmooth3dBoolShrinkToBool(output[0..], input[0..], InputSide, OutputSide, Radius, max_neighbors);

    z = 0;
    while (z < OutputSide) : (z += 1) {
        var y: usize = 0;
        while (y < OutputSide) : (y += 1) {
            const output_row = OutputSide * (y + OutputSide * z);
            var x: usize = 0;
            while (x < OutputSide) : (x += 1) {
                const expected = countSolidNeighborsShrinkRowSpan(input[0..], InputSide, x + Radius, y + Radius, z + Radius, Radius) >= max_neighbors;
                try std.testing.expectEqual(expected, output[output_row + x]);
            }
        }
    }
}

test "bool shrink radius two vector path matches row span reference" {
    const Radius: usize = 2;
    const InputSide: usize = 22;
    const OutputSide: usize = InputSide - 2 * Radius;
    const InputCount: usize = InputSide * InputSide * InputSide;
    const OutputCount: usize = OutputSide * OutputSide * OutputSide;
    const threshold: u32 = 60;
    var input = [_]bool{false} ** InputCount;
    var output = [_]bool{false} ** OutputCount;

    for (0..InputCount) |index| {
        const x = index % InputSide;
        const yz = index / InputSide;
        const y = yz % InputSide;
        const z = yz / InputSide;
        input[index] = ((x * 11 + y * 23 + z * 41 + index * 3) & 7) < 4;
    }

    try cellularAutomataSmooth3dBoolShrinkToBool(output[0..], input[0..], InputSide, OutputSide, Radius, threshold);

    var z: usize = 0;
    while (z < OutputSide) : (z += 1) {
        var y: usize = 0;
        while (y < OutputSide) : (y += 1) {
            const output_row = OutputSide * (y + OutputSide * z);
            var x: usize = 0;
            while (x < OutputSide) : (x += 1) {
                const expected = countSolidNeighborsShrinkRowSpan(input[0..], InputSide, x + Radius, y + Radius, z + Radius, Radius) >= threshold;
                try std.testing.expectEqual(expected, output[output_row + x]);
            }
        }
    }
}

test "bool shrink radius three row span path matches row span reference" {
    const Radius: usize = 3;
    const InputSide: usize = 17;
    const OutputSide: usize = InputSide - 2 * Radius;
    const InputCount: usize = InputSide * InputSide * InputSide;
    const OutputCount: usize = OutputSide * OutputSide * OutputSide;
    const threshold: u32 = 170;
    var input = [_]bool{false} ** InputCount;
    var output = [_]bool{false} ** OutputCount;

    for (0..InputCount) |index| {
        const x = index % InputSide;
        const yz = index / InputSide;
        const y = yz % InputSide;
        const z = yz / InputSide;
        input[index] = ((x * 13 + y * 19 + z * 29 + index * 5) & 7) < 4;
    }

    try cellularAutomataSmooth3dBoolShrinkToBool(output[0..], input[0..], InputSide, OutputSide, Radius, threshold);

    var z: usize = 0;
    while (z < OutputSide) : (z += 1) {
        var y: usize = 0;
        while (y < OutputSide) : (y += 1) {
            const output_row = OutputSide * (y + OutputSide * z);
            var x: usize = 0;
            while (x < OutputSide) : (x += 1) {
                const expected = countSolidNeighborsShrinkRowSpan(input[0..], InputSide, x + Radius, y + Radius, z + Radius, Radius) >= threshold;
                try std.testing.expectEqual(expected, output[output_row + x]);
            }
        }
    }
}

test "bool shrink wide generic path matches row span reference" {
    const Radius: usize = 4;
    const InputSide: usize = 12;
    const OutputSide: usize = InputSide - 2 * Radius;
    const InputCount: usize = InputSide * InputSide * InputSide;
    const OutputCount: usize = OutputSide * OutputSide * OutputSide;
    const threshold: u32 = 360;
    var input = [_]bool{false} ** InputCount;
    var output = [_]bool{false} ** OutputCount;

    for (0..InputCount) |index| {
        const x = index % InputSide;
        const yz = index / InputSide;
        const y = yz % InputSide;
        const z = yz / InputSide;
        input[index] = ((x * 11 + y * 23 + z * 41 + index * 3) & 7) < 4;
    }

    try cellularAutomataSmooth3dBoolShrinkToBool(output[0..], input[0..], InputSide, OutputSide, Radius, threshold);

    var z: usize = 0;
    while (z < OutputSide) : (z += 1) {
        var y: usize = 0;
        while (y < OutputSide) : (y += 1) {
            const output_row = OutputSide * (y + OutputSide * z);
            var x: usize = 0;
            while (x < OutputSide) : (x += 1) {
                const expected = countSolidNeighborsShrinkRowSpan(input[0..], InputSide, x + Radius, y + Radius, z + Radius, Radius) >= threshold;
                try std.testing.expectEqual(expected, output[output_row + x]);
            }
        }
    }
}

test "bool shrink wide bounded low path matches row span reference" {
    const cases = [_]struct {
        radius: usize,
        input_side: usize,
        threshold: u32,
    }{
        .{ .radius = 4, .input_side = 18, .threshold = 4 },
        .{ .radius = 8, .input_side = 19, .threshold = 8 },
    };

    inline for (cases) |case| {
        const OutputSide: usize = case.input_side - 2 * case.radius;
        const InputCount: usize = case.input_side * case.input_side * case.input_side;
        const OutputCount: usize = OutputSide * OutputSide * OutputSide;
        var input = [_]bool{false} ** InputCount;
        var output = [_]bool{false} ** OutputCount;

        for (0..InputCount) |index| {
            const x = index % case.input_side;
            const yz = index / case.input_side;
            const y = yz % case.input_side;
            const z = yz / case.input_side;
            input[index] = ((x * 13 + y * 19 + z * 29 + index * 5) & 15) < 8;
        }

        try cellularAutomataSmooth3dBoolShrinkToBool(output[0..], input[0..], case.input_side, OutputSide, case.radius, case.threshold);

        var z: usize = 0;
        while (z < OutputSide) : (z += 1) {
            var y: usize = 0;
            while (y < OutputSide) : (y += 1) {
                const output_row = OutputSide * (y + OutputSide * z);
                var x: usize = 0;
                while (x < OutputSide) : (x += 1) {
                    const expected = countSolidNeighborsShrinkRowSpan(input[0..], case.input_side, x + case.radius, y + case.radius, z + case.radius, case.radius) >= case.threshold;
                    try std.testing.expectEqual(expected, output[output_row + x]);
                }
            }
        }
    }
}

test "bool shrink wide large radius path matches row span reference" {
    const Radius: usize = 8;
    const InputSide: usize = 19;
    const OutputSide: usize = InputSide - 2 * Radius;
    const InputCount: usize = InputSide * InputSide * InputSide;
    const OutputCount: usize = OutputSide * OutputSide * OutputSide;
    const threshold: u32 = 2450;
    var input = [_]bool{false} ** InputCount;
    var output = [_]bool{false} ** OutputCount;

    for (0..InputCount) |index| {
        const x = index % InputSide;
        const yz = index / InputSide;
        const y = yz % InputSide;
        const z = yz / InputSide;
        input[index] = ((x * 11 + y * 23 + z * 41 + index * 3) & 7) < 4;
    }

    try cellularAutomataSmooth3dBoolShrinkToBool(output[0..], input[0..], InputSide, OutputSide, Radius, threshold);

    var z: usize = 0;
    while (z < OutputSide) : (z += 1) {
        var y: usize = 0;
        while (y < OutputSide) : (y += 1) {
            const output_row = OutputSide * (y + OutputSide * z);
            var x: usize = 0;
            while (x < OutputSide) : (x += 1) {
                const expected = countSolidNeighborsShrinkRowSpan(input[0..], InputSide, x + Radius, y + Radius, z + Radius, Radius) >= threshold;
                try std.testing.expectEqual(expected, output[output_row + x]);
            }
        }
    }
}

test "bool shrink sparse threshold one inverse path matches row span reference" {
    const Radius: usize = 4;
    const InputSide: usize = 16;
    const OutputSide: usize = InputSide - 2 * Radius;
    const InputCount: usize = InputSide * InputSide * InputSide;
    const OutputCount: usize = OutputSide * OutputSide * OutputSide;
    const threshold: u32 = 1;
    var input = [_]bool{false} ** InputCount;
    var output = [_]bool{false} ** OutputCount;

    input[cellularAutomataIndex3d(2, 3, 4, InputSide)] = true;
    input[cellularAutomataIndex3d(9, 5, 7, InputSide)] = true;
    input[cellularAutomataIndex3d(12, 13, 10, InputSide)] = true;

    try cellularAutomataSmooth3dBoolShrinkToBool(output[0..], input[0..], InputSide, OutputSide, Radius, threshold);

    var z: usize = 0;
    while (z < OutputSide) : (z += 1) {
        var y: usize = 0;
        while (y < OutputSide) : (y += 1) {
            const output_row = OutputSide * (y + OutputSide * z);
            var x: usize = 0;
            while (x < OutputSide) : (x += 1) {
                const expected = countSolidNeighborsShrinkRowSpan(input[0..], InputSide, x + Radius, y + Radius, z + Radius, Radius) >= threshold;
                try std.testing.expectEqual(expected, output[output_row + x]);
            }
        }
    }
}

test "bool shrink dense max threshold inverse path matches row span reference" {
    const Radius: usize = 4;
    const InputSide: usize = 16;
    const OutputSide: usize = InputSide - 2 * Radius;
    const InputCount: usize = InputSide * InputSide * InputSide;
    const OutputCount: usize = OutputSide * OutputSide * OutputSide;
    const threshold: u32 = try maxNeighborCount(Radius);
    var input = [_]bool{true} ** InputCount;
    var output = [_]bool{false} ** OutputCount;

    input[cellularAutomataIndex3d(2, 3, 4, InputSide)] = false;
    input[cellularAutomataIndex3d(9, 5, 7, InputSide)] = false;
    input[cellularAutomataIndex3d(12, 13, 10, InputSide)] = false;

    try cellularAutomataSmooth3dBoolShrinkToBool(output[0..], input[0..], InputSide, OutputSide, Radius, threshold);

    var z: usize = 0;
    while (z < OutputSide) : (z += 1) {
        var y: usize = 0;
        while (y < OutputSide) : (y += 1) {
            const output_row = OutputSide * (y + OutputSide * z);
            var x: usize = 0;
            while (x < OutputSide) : (x += 1) {
                const expected = countSolidNeighborsShrinkRowSpan(input[0..], InputSide, x + Radius, y + Radius, z + Radius, Radius) >= threshold;
                try std.testing.expectEqual(expected, output[output_row + x]);
            }
        }
    }
}

test "solid byte span sum fixed cases match scalar sum" {
    var input: [32]u8 = undefined;
    for (input[0..], 0..) |*value, index| {
        value.* = @intFromBool((index * 17 + 5) % 7 < 3);
    }

    const lengths = [_]usize{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 };
    for (lengths) |len| {
        var expected: u32 = 0;
        for (input[0..len]) |value| {
            expected += value;
        }
        try std.testing.expectEqual(expected, solidByteSpanSum(input[0..].ptr, len));
        try std.testing.expectEqual(expected, solidByteSpanSumLargeWindow(input[0..].ptr, len));
    }
}

fn expectPreparedSmoothMatchesPublic(comptime Side: usize, comptime boundary_is_solid: bool) !void {
    const Count: usize = Side * Side * Side;
    var input = [_]bool{false} ** Count;
    var public_output = [_]bool{false} ** Count;
    var prepared_output = [_]bool{false} ** Count;

    for (0..Count) |index| {
        const x = index % Side;
        const yz = index / Side;
        const y = yz % Side;
        const z = yz / Side;
        input[index] = ((x * 17 + y * 31 + z * 43 + index * 7) & 3) != 0;
    }

    const options = SmoothOptions{
        .neighborhood_radius = 1,
        .solid_threshold = 14,
        .boundary_is_solid = boundary_is_solid,
    };
    const plan = try PreparedSmoothPlan.init(Side, options);
    try cellularAutomataSmooth3d(public_output[0..], input[0..], Side, options);
    try plan.smooth(prepared_output[0..], input[0..]);

    for (public_output, prepared_output) |public_solid, prepared_solid| {
        try std.testing.expectEqual(public_solid, prepared_solid);
    }
}

test "prepared smooth plan matches public smoothing" {
    try expectPreparedSmoothMatchesPublic(19, true);
    try expectPreparedSmoothMatchesPublic(19, false);
    try expectPreparedSmoothMatchesPublic(2, true);
    try expectPreparedSmoothMatchesPublic(2, false);
}

fn expectThresholdSweepMatchesPublic(comptime boundary_is_solid: bool) !void {
    const Side: usize = 11;
    const Count: usize = Side * Side * Side;
    const thresholds = [_]u32{ 0, 10, 14, 26 };
    var input = [_]bool{false} ** Count;
    var expected = [_]bool{false} ** Count;
    var sweep = [_]bool{false} ** (Count * thresholds.len);

    for (0..Count) |index| {
        const x = index % Side;
        const yz = index / Side;
        const y = yz % Side;
        const z = yz / Side;
        input[index] = ((x * 17 + y * 31 + z * 43 + index * 7) & 3) != 0;
    }

    try cellularAutomataSmooth3dRadius1ThresholdSweep(sweep[0..], input[0..], Side, thresholds[0..], boundary_is_solid);
    for (thresholds, 0..) |threshold, candidate| {
        try cellularAutomataSmooth3d(expected[0..], input[0..], Side, .{
            .neighborhood_radius = 1,
            .solid_threshold = threshold,
            .boundary_is_solid = boundary_is_solid,
        });
        for (expected, sweep[candidate * Count ..][0..Count]) |expected_solid, sweep_solid| {
            try std.testing.expectEqual(expected_solid, sweep_solid);
        }
    }
}

test "threshold sweep matches public smoothing candidates" {
    try expectThresholdSweepMatchesPublic(true);
    try expectThresholdSweepMatchesPublic(false);
}

fn expectOffsetSmoothMatchesPublic(comptime Side: usize, comptime boundary_is_solid: bool) !void {
    const Count: usize = Side * Side * Side;
    var input = [_]bool{false} ** Count;
    var public_output = [_]bool{false} ** Count;
    var offset_output = [_]bool{false} ** Count;

    for (0..Count) |index| {
        const x = index % Side;
        const yz = index / Side;
        const y = yz % Side;
        const z = yz / Side;
        input[index] = ((x * 19 + y * 23 + z * 37 + index * 11) & 7) < 5;
    }

    const options = SmoothOptions{
        .neighborhood_radius = 1,
        .solid_threshold = 14,
        .boundary_is_solid = boundary_is_solid,
    };
    var plan = try OffsetSmoothPlan.init(std.testing.allocator, Side, options);
    defer plan.deinit();

    try cellularAutomataSmooth3d(public_output[0..], input[0..], Side, options);
    try plan.smooth(offset_output[0..], input[0..]);

    for (public_output, offset_output) |public_solid, offset_solid| {
        try std.testing.expectEqual(public_solid, offset_solid);
    }
}

test "offset smooth plan matches public smoothing" {
    try expectOffsetSmoothMatchesPublic(19, true);
    try expectOffsetSmoothMatchesPublic(19, false);
}

test "independent halo chunks match a larger smoothed world" {
    const allocator = std.testing.allocator;
    const chunk_size: usize = 4;
    const iterations: usize = 2;
    const radius: usize = 1;
    const padding = try cellularAutomataRequiredPadding(iterations, radius);
    const seed: u32 = 99;
    const fill_percent: f32 = 0.51;
    const threshold: u32 = 14;

    const chunk0 = try allocator.alloc(bool, try cellularAutomataCellCount(chunk_size));
    defer allocator.free(chunk0);
    const chunk1 = try allocator.alloc(bool, try cellularAutomataCellCount(chunk_size));
    defer allocator.free(chunk1);

    try cellularAutomataChunk3d(allocator, chunk0, .{
        .chunk_size = chunk_size,
        .chunk_x = 0,
        .chunk_y = 0,
        .chunk_z = 0,
        .seed = seed,
        .fill_percent = fill_percent,
        .iterations = iterations,
        .neighborhood_radius = radius,
        .solid_threshold = threshold,
    });

    try cellularAutomataChunk3d(allocator, chunk1, .{
        .chunk_size = chunk_size,
        .chunk_x = 1,
        .chunk_y = 0,
        .chunk_z = 0,
        .seed = seed,
        .fill_percent = fill_percent,
        .iterations = iterations,
        .neighborhood_radius = radius,
        .solid_threshold = threshold,
    });

    const combined_side = try cellularAutomataTemporarySide(chunk_size * 2, padding);
    const current = try allocator.alloc(bool, try cellularAutomataCellCount(combined_side));
    defer allocator.free(current);
    const next = try allocator.alloc(bool, try cellularAutomataCellCount(combined_side));
    defer allocator.free(next);

    const padding_i64 = try usizeToI64(padding);
    try cellularAutomataRandomSolidGrid3d(current, combined_side, .{
        .x = -padding_i64,
        .y = -padding_i64,
        .z = -padding_i64,
    }, seed, fill_percent);

    var input = current;
    var output = next;
    var iteration: usize = 0;
    while (iteration < iterations) : (iteration += 1) {
        try cellularAutomataSmooth3d(output, input, combined_side, .{
            .neighborhood_radius = radius,
            .solid_threshold = threshold,
        });
        const swap = input;
        input = output;
        output = swap;
    }

    var z: usize = 0;
    while (z < chunk_size) : (z += 1) {
        var y: usize = 0;
        while (y < chunk_size) : (y += 1) {
            var x: usize = 0;
            while (x < chunk_size) : (x += 1) {
                const local_index = cellularAutomataIndex3d(x, y, z, chunk_size);
                try std.testing.expectEqual(
                    input[cellularAutomataIndex3d(x + padding, y + padding, z + padding, combined_side)],
                    chunk0[local_index],
                );
                try std.testing.expectEqual(
                    input[cellularAutomataIndex3d(x + padding + chunk_size, y + padding, z + padding, combined_side)],
                    chunk1[local_index],
                );
            }
        }
    }
}

test "scratch chunk generation matches allocator generation" {
    const allocator = std.testing.allocator;
    const options = GenerateOptions{
        .chunk_size = 8,
        .chunk_x = -2,
        .chunk_y = 3,
        .chunk_z = 5,
        .seed = 1234,
        .fill_percent = 0.49,
        .iterations = 3,
        .neighborhood_radius = 1,
        .solid_threshold = 14,
    };
    const count = try cellularAutomataCellCount(options.chunk_size);

    const allocator_chunk = try allocator.alloc(bool, count);
    defer allocator.free(allocator_chunk);
    const scratch_chunk = try allocator.alloc(bool, count);
    defer allocator.free(scratch_chunk);
    const owned_scratch_chunk = try allocator.alloc(bool, count);
    defer allocator.free(owned_scratch_chunk);

    const scratch_size = try cellularAutomataChunk3dScratchByteCount(options);
    const scratch = try allocator.alloc(u8, scratch_size);
    defer allocator.free(scratch);

    try cellularAutomataChunk3d(allocator, allocator_chunk, options);
    try cellularAutomataChunk3dWithScratch(scratch_chunk, scratch, options);
    try std.testing.expectEqualSlices(bool, allocator_chunk, scratch_chunk);

    var owned_scratch = try CellularAutomataChunkScratch.init(allocator, options);
    defer owned_scratch.deinit();
    try owned_scratch.generateChunk3d(owned_scratch_chunk, options);
    try std.testing.expectEqualSlices(bool, allocator_chunk, owned_scratch_chunk);

    try std.testing.expectError(
        error.BufferTooSmall,
        cellularAutomataChunk3dWithScratch(scratch_chunk, scratch[0 .. scratch_size - 1], options),
    );
}

test "random-only chunks do not require scratch storage" {
    const allocator = std.testing.allocator;
    const options = GenerateOptions{
        .chunk_size = 8,
        .chunk_x = -3,
        .chunk_y = 2,
        .chunk_z = 4,
        .seed = 77,
        .fill_percent = 0.49,
        .iterations = 0,
    };
    const count = try cellularAutomataCellCount(options.chunk_size);
    const allocator_chunk = try allocator.alloc(bool, count);
    defer allocator.free(allocator_chunk);
    const scratch_chunk = try allocator.alloc(bool, count);
    defer allocator.free(scratch_chunk);
    const owned_scratch_chunk = try allocator.alloc(bool, count);
    defer allocator.free(owned_scratch_chunk);

    try std.testing.expectEqual(@as(usize, 0), try cellularAutomataChunk3dScratchByteCount(options));
    try cellularAutomataChunk3d(allocator, allocator_chunk, options);
    try cellularAutomataChunk3dWithScratch(scratch_chunk, &.{}, options);
    try std.testing.expectEqualSlices(bool, allocator_chunk, scratch_chunk);

    var owned_scratch = try CellularAutomataChunkScratch.init(allocator, options);
    defer owned_scratch.deinit();
    try std.testing.expectEqual(@as(usize, 0), owned_scratch.buffer.len);
    try owned_scratch.generateChunk3d(owned_scratch_chunk, options);
    try std.testing.expectEqualSlices(bool, allocator_chunk, owned_scratch_chunk);
}

test "constant-output smoothed chunks do not require scratch storage" {
    const allocator = std.testing.allocator;
    const cases = [_]struct {
        options: GenerateOptions,
        expected: bool,
    }{
        .{
            .options = .{
                .chunk_size = 8,
                .chunk_x = -2,
                .chunk_y = 3,
                .chunk_z = 5,
                .seed = 1234,
                .fill_percent = 0.0,
                .iterations = 3,
                .neighborhood_radius = 1,
                .solid_threshold = 14,
            },
            .expected = false,
        },
        .{
            .options = .{
                .chunk_size = 8,
                .chunk_x = 4,
                .chunk_y = -1,
                .chunk_z = 2,
                .seed = 9876,
                .fill_percent = 0.0,
                .iterations = 2,
                .neighborhood_radius = 2,
                .solid_threshold = 0,
            },
            .expected = true,
        },
        .{
            .options = .{
                .chunk_size = 8,
                .chunk_x = 6,
                .chunk_y = -4,
                .chunk_z = 1,
                .seed = 2468,
                .fill_percent = 0.49,
                .iterations = 3,
                .neighborhood_radius = 1,
                .solid_threshold = 0,
            },
            .expected = true,
        },
        .{
            .options = .{
                .chunk_size = 6,
                .chunk_x = -5,
                .chunk_y = 7,
                .chunk_z = -3,
                .seed = 555,
                .fill_percent = 1.0,
                .iterations = 2,
                .neighborhood_radius = 4,
                .solid_threshold = 100,
            },
            .expected = true,
        },
    };

    for (cases) |case| {
        const count = try cellularAutomataCellCount(case.options.chunk_size);
        const allocator_chunk = try allocator.alloc(bool, count);
        defer allocator.free(allocator_chunk);
        const scratch_chunk = try allocator.alloc(bool, count);
        defer allocator.free(scratch_chunk);

        try std.testing.expectEqual(@as(usize, 0), try cellularAutomataChunk3dScratchByteCount(case.options));
        try cellularAutomataChunk3d(allocator, allocator_chunk, case.options);
        try cellularAutomataChunk3dWithScratch(scratch_chunk, &.{}, case.options);
        try std.testing.expectEqualSlices(bool, allocator_chunk, scratch_chunk);
        for (allocator_chunk) |solid| {
            try std.testing.expectEqual(case.expected, solid);
        }
    }
}

test "scratch chunk generation supports generic radius path" {
    const allocator = std.testing.allocator;
    const options = GenerateOptions{
        .chunk_size = 4,
        .chunk_x = 1,
        .chunk_y = -1,
        .chunk_z = 2,
        .seed = 5678,
        .fill_percent = 0.52,
        .iterations = 2,
        .neighborhood_radius = 2,
        .solid_threshold = 45,
    };
    const count = try cellularAutomataCellCount(options.chunk_size);

    const allocator_chunk = try allocator.alloc(bool, count);
    defer allocator.free(allocator_chunk);
    const scratch_chunk = try allocator.alloc(bool, count);
    defer allocator.free(scratch_chunk);

    const scratch_size = try cellularAutomataChunk3dScratchByteCount(options);
    const scratch = try allocator.alloc(u8, scratch_size);
    defer allocator.free(scratch);

    try cellularAutomataChunk3d(allocator, allocator_chunk, options);
    try cellularAutomataChunk3dWithScratch(scratch_chunk, scratch, options);
    try std.testing.expectEqualSlices(bool, allocator_chunk, scratch_chunk);
}

test "single-iteration generic radius path uses only current scratch" {
    const allocator = std.testing.allocator;
    const options = GenerateOptions{
        .chunk_size = 8,
        .chunk_x = 1,
        .chunk_y = -2,
        .chunk_z = 3,
        .seed = 9012,
        .fill_percent = 0.52,
        .iterations = 1,
        .neighborhood_radius = 2,
        .solid_threshold = 60,
    };
    const padding = try cellularAutomataRequiredPadding(options.iterations, options.neighborhood_radius);
    const temp_side = try cellularAutomataTemporarySide(options.chunk_size, padding);
    const temp_count = try cellularAutomataCellCount(temp_side);
    const count = try cellularAutomataCellCount(options.chunk_size);

    try std.testing.expectEqual(temp_count, try cellularAutomataChunk3dScratchByteCount(options));

    const allocator_chunk = try allocator.alloc(bool, count);
    defer allocator.free(allocator_chunk);
    const scratch_chunk = try allocator.alloc(bool, count);
    defer allocator.free(scratch_chunk);
    const scratch = try allocator.alloc(u8, temp_count);
    defer allocator.free(scratch);

    try cellularAutomataChunk3d(allocator, allocator_chunk, options);
    try cellularAutomataChunk3dWithScratch(scratch_chunk, scratch, options);
    try std.testing.expectEqualSlices(bool, allocator_chunk, scratch_chunk);
}

test "generate chunk rejects undersized output" {
    var buffer: [7]bool = undefined;
    try std.testing.expectError(error.BufferTooSmall, cellularAutomataChunk3d(std.testing.allocator, buffer[0..], .{
        .chunk_size = 2,
        .iterations = 0,
    }));
}
