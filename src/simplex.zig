const std = @import("std");
const common = @import("common.zig");

comptime {
    @setFloatMode(.optimized);
}

const prime_x: u32 = 501_125_321;
const prime_y: u32 = 1_136_930_381;
const prime_z: u32 = 1_720_413_743;

const f3: f32 = 1.0 / 3.0;
const g3: f32 = 1.0 / 6.0;
const g3_drift_1: f32 = 0.1640625;
const g3_drift_2: f32 = 0.328125;
const f2: f32 = 0.366_025_403_784_438_6;
const g2: f32 = 0.211_324_865_405_187_13;

const simplex_falloff_shift_2d: f32 = 1.0 / 6.0;
const simplex_falloff_shift_3d: f32 = 0.2;
const simplex_falloff_radius_2d_shifted_square: f32 = 0.5 - simplex_falloff_shift_2d;
const simplex_falloff_radius_3d_shifted_square: f32 = 0.6 - simplex_falloff_shift_3d;
const simplex_falloff_radius_2d_shifted_square_f16: f16 = @floatCast(simplex_falloff_radius_2d_shifted_square);
const simplex_falloff_radius_3d_shifted_square_f16: f16 = @floatCast(simplex_falloff_radius_3d_shifted_square);
const simplex_output_scale_2d_shifted_square: f32 = 39.375;
const simplex_output_scale_3d_shifted_square: f32 = 25.92;
const F32x8 = common.F32x8;
const F16x8 = common.F16x8;
const I32x8 = common.I32x8;
const U16x8 = common.U16x8;
const U32x8 = common.U32x8;

pub const simplex_noise_chunk_16_3d_count: usize = 16 * 16 * 16;
pub const simplex_noise_chunk_32_3d_count: usize = 32 * 32 * 32;
pub const simplex_noise_chunk_64_3d_count: usize = 64 * 64 * 64;

const Grad3 = struct {
    x: f32,
    y: f32,
    z: f32,
};

const grad3_table = [_]Grad3{
    .{ .x = 1.0, .y = 1.0, .z = 0.0 },
    .{ .x = -1.0, .y = 1.0, .z = 0.0 },
    .{ .x = 1.0, .y = -1.0, .z = 0.0 },
    .{ .x = -1.0, .y = -1.0, .z = 0.0 },
    .{ .x = 1.0, .y = 0.0, .z = 1.0 },
    .{ .x = -1.0, .y = 0.0, .z = 1.0 },
    .{ .x = 1.0, .y = 0.0, .z = -1.0 },
    .{ .x = -1.0, .y = 0.0, .z = -1.0 },
    .{ .x = 0.0, .y = 1.0, .z = 1.0 },
    .{ .x = 0.0, .y = -1.0, .z = 1.0 },
    .{ .x = 0.0, .y = 1.0, .z = -1.0 },
    .{ .x = 0.0, .y = -1.0, .z = -1.0 },
    .{ .x = 1.0, .y = 1.0, .z = 0.0 },
    .{ .x = 0.0, .y = -1.0, .z = 1.0 },
    .{ .x = -1.0, .y = 1.0, .z = 0.0 },
    .{ .x = 0.0, .y = -1.0, .z = -1.0 },
};

inline fn primeIf(mask: u32, prime: u32) u32 {
    return prime & (0 -% mask);
}

// Deliberately weak hash.
// Only used to select one of 16 simplex gradients.
// Visually validated for terrain/cave noise; do not replace with a stronger
// hash unless it improves artifacts enough to justify the hot-path cost.
inline fn hash3(seed: u32, x_primed: u32, y_primed: u32, z_primed: u32) u32 {
    var h = seed ^ x_primed ^ y_primed ^ z_primed;
    h ^= h >> 16;
    return h;
}

inline fn hash2(seed: u32, x_primed: u32, y_primed: u32) u32 {
    var h = seed ^ x_primed ^ y_primed;
    h ^= h >> 16;
    return h;
}

inline fn hash3x8Seeded(seed_v: U32x8, x_primed: U32x8, y_primed: U32x8, z_primed: U32x8) U32x8 {
    var h = seed_v ^ x_primed ^ y_primed ^ z_primed;
    h ^= h >> @as(U32x8, @splat(16));
    return h;
}

inline fn hash2x8(seed: u32, x_primed: U32x8, y_primed: U32x8) U32x8 {
    var h = @as(U32x8, @splat(seed)) ^ x_primed ^ y_primed;
    h ^= h >> @as(U32x8, @splat(16));
    return h;
}

inline fn store8(out: []f32, index: usize, value: F32x8) void {
    out[index + 0] = value[0];
    out[index + 1] = value[1];
    out[index + 2] = value[2];
    out[index + 3] = value[3];
    out[index + 4] = value[4];
    out[index + 5] = value[5];
    out[index + 6] = value[6];
    out[index + 7] = value[7];
}

inline fn loadF32x8(pointer: [*]const f32) F32x8 {
    const vector_pointer: *align(4) const F32x8 = @ptrCast(pointer);
    return vector_pointer.*;
}

inline fn floorToI32x8(value: F32x8) I32x8 {
    return @intFromFloat(@floor(value));
}

inline fn truncToI32x8(value: F32x8) I32x8 {
    return @intFromFloat(value);
}

inline fn debugAssertNonnegativeSimplexDomain8(x: F32x8, y: F32x8, z: F32x8, skew: F32x8) void {
    if (std.debug.runtime_safety) {
        const zero = @as(F32x8, @splat(0.0));
        std.debug.assert(@reduce(.And, (x + skew >= zero) & (y + skew >= zero) & (z + skew >= zero)));
    }
}

inline fn debugAssertNonnegativeSimplexDomainScalar(x: f32, y: f32, z: f32) void {
    if (std.debug.runtime_safety) {
        const skew = (x + y + z) * f3;
        std.debug.assert(x + skew >= 0.0 and y + skew >= 0.0 and z + skew >= 0.0);
    }
}

inline fn xorSign16(value: F16x8, sign_mask: U16x8) F16x8 {
    const bits: U16x8 = @bitCast(value);
    return @bitCast(bits ^ sign_mask);
}

inline fn grad16Dot2dTable32(h: u32, x: f32, y: f32) f32 {
    const gradient = grad3_table[h & 15];
    return @mulAdd(f32, x, gradient.x, y * gradient.y);
}

inline fn grad16Dot3dTable32(h: u32, x: f32, y: f32, z: f32) f32 {
    const gradient = grad3_table[h & 15];
    return @mulAdd(f32, x, gradient.x, @mulAdd(f32, y, gradient.y, z * gradient.z));
}

inline fn grad16Dot2d8Nibble16(h: U32x8, x: F16x8, y: F16x8) F16x8 {
    const u = @select(f16, h < @as(U32x8, @splat(8)), x, y);

    const zero = @as(F16x8, @splat(@as(f16, 0.0)));
    const v_y_or_zero = @select(f16, h < @as(U32x8, @splat(4)), y, zero);
    const use_x_for_v = (h & @as(U32x8, @splat(13))) == @as(U32x8, @splat(12));
    const v = @select(f16, use_x_for_v, x, v_y_or_zero);

    const sign_u: U16x8 = @truncate(h << @as(U32x8, @splat(15)));
    const sign_v: U16x8 = @truncate((h << @as(U32x8, @splat(14))) & @as(U32x8, @splat(0x8000)));
    return xorSign16(u, sign_u) + xorSign16(v, sign_v);
}

inline fn grad16Dot3d8Nibble16(h: U16x8, x: F16x8, y: F16x8, z: F16x8) F16x8 {
    const sign_u = h << @as(U16x8, @splat(15));
    const sign_v = (h << @as(U16x8, @splat(14))) & @as(U16x8, @splat(0x8000));

    const u = @select(f16, h < @as(U16x8, @splat(8)), x, y);

    const y_or_z = @select(f16, h < @as(U16x8, @splat(4)), y, z);
    const use_x_for_v = (h & @as(U16x8, @splat(13))) == @as(U16x8, @splat(12));
    const v = @select(f16, use_x_for_v, x, y_or_z);

    return xorSign16(u, sign_u) + xorSign16(v, sign_v);
}

inline fn simplexCornerSample2d8F16Preconverted(h: U32x8, x: F16x8, y: F16x8) F16x8 {
    var t = @mulAdd(F16x8, x, -x, @as(F16x8, @splat(@as(f16, 0.5))));
    t = @max(@mulAdd(F16x8, y, -y, t), @as(F16x8, @splat(@as(f16, 0.0))));

    const t2 = t * t;
    return (t2 * t2) * grad16Dot2d8Nibble16(h, x, y);
}

inline fn simplexCornerSample2d8F16PreconvertedShiftedSquare(h: U32x8, x: F16x8, y: F16x8) F16x8 {
    var t = @mulAdd(F16x8, x, -x, @as(F16x8, @splat(simplex_falloff_radius_2d_shifted_square_f16)));
    t = @max(@mulAdd(F16x8, y, -y, t), @as(F16x8, @splat(@as(f16, 0.0))));

    return (t * t) * grad16Dot2d8Nibble16(h, x, y);
}

inline fn simplexCornerSample2d8F16FromF32(h: U32x8, x: F32x8, y: F32x8) F32x8 {
    return @floatCast(simplexCornerSample2d8F16Preconverted(
        h,
        @as(F16x8, @floatCast(x)),
        @as(F16x8, @floatCast(y)),
    ));
}

inline fn simplexCornerSample2d8F16FromF32ShiftedSquare(h: U32x8, x: F32x8, y: F32x8) F32x8 {
    return @floatCast(simplexCornerSample2d8F16PreconvertedShiftedSquare(
        h,
        @as(F16x8, @floatCast(x)),
        @as(F16x8, @floatCast(y)),
    ));
}

inline fn simplexCornerSample2dF32(h: u32, x: f32, y: f32) f32 {
    var t = @mulAdd(f32, x, -x, 0.5);
    t = @max(@mulAdd(f32, y, -y, t), 0.0);

    const t2 = t * t;
    return (t2 * t2) * grad16Dot2dTable32(h, x, y);
}

inline fn simplexCornerSample2dF32ShiftedSquare(h: u32, x: f32, y: f32) f32 {
    var t = @mulAdd(f32, x, -x, simplex_falloff_radius_2d_shifted_square);
    t = @max(@mulAdd(f32, y, -y, t), 0.0);

    return (t * t) * grad16Dot2dTable32(h, x, y);
}

inline fn simplexCornerSample3d8F16Preconverted(
    seed_v: U32x8,
    x_primed: U32x8,
    y_primed: U32x8,
    z_primed: U32x8,
    x: F16x8,
    y: F16x8,
    z: F16x8,
) F16x8 {
    var t = @mulAdd(F16x8, x, -x, @as(F16x8, @splat(@as(f16, 0.6))));
    t = @mulAdd(F16x8, y, -y, t);
    t = @max(@mulAdd(F16x8, z, -z, t), @as(F16x8, @splat(@as(f16, 0.0))));

    const h: U16x8 = @truncate(hash3x8Seeded(seed_v, x_primed, y_primed, z_primed) & @as(U32x8, @splat(15)));
    const t2 = t * t;
    return (t2 * t2) * grad16Dot3d8Nibble16(h, x, y, z);
}

inline fn simplexCornerSample3d8F16PreconvertedShiftedSquare(
    seed_v: U32x8,
    x_primed: U32x8,
    y_primed: U32x8,
    z_primed: U32x8,
    x: F16x8,
    y: F16x8,
    z: F16x8,
) F16x8 {
    var t = @mulAdd(F16x8, x, -x, @as(F16x8, @splat(simplex_falloff_radius_3d_shifted_square_f16)));
    t = @mulAdd(F16x8, y, -y, t);
    t = @max(@mulAdd(F16x8, z, -z, t), @as(F16x8, @splat(@as(f16, 0.0))));

    const h: U16x8 = @truncate(hash3x8Seeded(seed_v, x_primed, y_primed, z_primed) & @as(U32x8, @splat(15)));
    return (t * t) * grad16Dot3d8Nibble16(h, x, y, z);
}

inline fn simplexCornerSample3d8F16FromF32(
    seed_v: U32x8,
    x_primed: U32x8,
    y_primed: U32x8,
    z_primed: U32x8,
    x: F32x8,
    y: F32x8,
    z: F32x8,
) F32x8 {
    return @floatCast(simplexCornerSample3d8F16Preconverted(
        seed_v,
        x_primed,
        y_primed,
        z_primed,
        @as(F16x8, @floatCast(x)),
        @as(F16x8, @floatCast(y)),
        @as(F16x8, @floatCast(z)),
    ));
}

inline fn simplexCornerSample3d8F16FromF32ShiftedSquare(
    seed_v: U32x8,
    x_primed: U32x8,
    y_primed: U32x8,
    z_primed: U32x8,
    x: F32x8,
    y: F32x8,
    z: F32x8,
) F32x8 {
    return @floatCast(simplexCornerSample3d8F16PreconvertedShiftedSquare(
        seed_v,
        x_primed,
        y_primed,
        z_primed,
        @as(F16x8, @floatCast(x)),
        @as(F16x8, @floatCast(y)),
        @as(F16x8, @floatCast(z)),
    ));
}

inline fn simplexCornerSample3dF32(seed: u32, x_primed: u32, y_primed: u32, z_primed: u32, x: f32, y: f32, z: f32) f32 {
    var t = @mulAdd(f32, x, -x, 0.6);
    t = @mulAdd(f32, y, -y, t);
    t = @max(@mulAdd(f32, z, -z, t), 0.0);

    const h = hash3(seed, x_primed, y_primed, z_primed);
    const t2 = t * t;
    return (t2 * t2) * grad16Dot3dTable32(h, x, y, z);
}

inline fn simplexCornerSample3dF32ShiftedSquare(seed: u32, x_primed: u32, y_primed: u32, z_primed: u32, x: f32, y: f32, z: f32) f32 {
    var t = @mulAdd(f32, x, -x, simplex_falloff_radius_3d_shifted_square);
    t = @mulAdd(f32, y, -y, t);
    t = @max(@mulAdd(f32, z, -z, t), 0.0);

    const h = hash3(seed, x_primed, y_primed, z_primed);
    return (t * t) * grad16Dot3dTable32(h, x, y, z);
}

pub fn simplexNoise2d(seed: u32, x: f32, y: f32) f32 {
    const s = (x + y) * f2;

    const i: i32 = @floor(x + s);
    const j: i32 = @floor(y + s);

    const iu: u32 = @bitCast(i);
    const ju: u32 = @bitCast(j);

    const x0p = iu *% prime_x;
    const y0p = ju *% prime_y;

    const t = @as(f32, @floatFromInt(i + j)) * g2;

    const cell_x = @as(f32, @floatFromInt(i)) - t;
    const cell_y = @as(f32, @floatFromInt(j)) - t;

    const x0 = x - cell_x;
    const y0 = y - cell_y;

    const x_ge_y: u32 = @intFromBool(x0 >= y0);
    const y_gt_x = 1 - x_ge_y;

    const x1p = x0p +% primeIf(x_ge_y, prime_x);
    const y1p = y0p +% primeIf(y_gt_x, prime_y);

    const x2p = x0p +% prime_x;
    const y2p = y0p +% prime_y;

    const x1 = x0 + (if (x_ge_y != 0) g2 - 1.0 else g2);
    const y1 = y0 + (if (x_ge_y != 0) g2 else g2 - 1.0);
    const x2 = x0 + 2.0 * g2 - 1.0;
    const y2 = y0 + 2.0 * g2 - 1.0;

    const h0 = hash2(seed, x0p, y0p) & 15;
    const h1 = hash2(seed, x1p, y1p) & 15;
    const h2 = hash2(seed, x2p, y2p) & 15;

    const n =
        simplexCornerSample2dF32(h0, x0, y0) +
        simplexCornerSample2dF32(h1, x1, y1) +
        simplexCornerSample2dF32(h2, x2, y2);

    return 70.0 * n;
}

pub fn simplexNoise2dShiftedSquare(seed: u32, x: f32, y: f32) f32 {
    const s = (x + y) * f2;

    const i: i32 = @floor(x + s);
    const j: i32 = @floor(y + s);

    const iu: u32 = @bitCast(i);
    const ju: u32 = @bitCast(j);

    const x0p = iu *% prime_x;
    const y0p = ju *% prime_y;

    const t = @as(f32, @floatFromInt(i + j)) * g2;

    const cell_x = @as(f32, @floatFromInt(i)) - t;
    const cell_y = @as(f32, @floatFromInt(j)) - t;

    const x0 = x - cell_x;
    const y0 = y - cell_y;

    const x_ge_y: u32 = @intFromBool(x0 >= y0);
    const y_gt_x = 1 - x_ge_y;

    const x1p = x0p +% primeIf(x_ge_y, prime_x);
    const y1p = y0p +% primeIf(y_gt_x, prime_y);

    const x2p = x0p +% prime_x;
    const y2p = y0p +% prime_y;

    const x1 = x0 + (if (x_ge_y != 0) g2 - 1.0 else g2);
    const y1 = y0 + (if (x_ge_y != 0) g2 else g2 - 1.0);
    const x2 = x0 + 2.0 * g2 - 1.0;
    const y2 = y0 + 2.0 * g2 - 1.0;

    const h0 = hash2(seed, x0p, y0p) & 15;
    const h1 = hash2(seed, x1p, y1p) & 15;
    const h2 = hash2(seed, x2p, y2p) & 15;

    const n =
        simplexCornerSample2dF32ShiftedSquare(h0, x0, y0) +
        simplexCornerSample2dF32ShiftedSquare(h1, x1, y1) +
        simplexCornerSample2dF32ShiftedSquare(h2, x2, y2);

    return simplex_output_scale_2d_shifted_square * n;
}

inline fn simplexNoise2d8(seed: u32, x: F32x8, y: F32x8) F32x8 {
    const s = (x + y) * @as(F32x8, @splat(f2));
    const i = floorToI32x8(x + s);
    const j = floorToI32x8(y + s);

    const x0p = @as(U32x8, @bitCast(i)) *% @as(U32x8, @splat(prime_x));
    const y0p = @as(U32x8, @bitCast(j)) *% @as(U32x8, @splat(prime_y));

    const t = @as(F32x8, @floatFromInt(i +% j)) * @as(F32x8, @splat(g2));
    const cell_x = @as(F32x8, @floatFromInt(i)) - t;
    const cell_y = @as(F32x8, @floatFromInt(j)) - t;

    const x0 = x - cell_x;
    const y0 = y - cell_y;
    const x_ge_y = x0 >= y0;

    const zero_u = @as(U32x8, @splat(0));
    const g2v = @as(F32x8, @splat(g2));
    const g2_minus_one = @as(F32x8, @splat(g2 - 1.0));
    const two_g2_minus_one = @as(F32x8, @splat(2.0 * g2 - 1.0));

    const x1p = x0p +% @select(u32, x_ge_y, @as(U32x8, @splat(prime_x)), zero_u);
    const y1p = y0p +% @select(u32, x_ge_y, zero_u, @as(U32x8, @splat(prime_y)));
    const x2p = x0p +% @as(U32x8, @splat(prime_x));
    const y2p = y0p +% @as(U32x8, @splat(prime_y));

    const x1 = x0 + @select(f32, x_ge_y, g2_minus_one, g2v);
    const y1 = y0 + @select(f32, x_ge_y, g2v, g2_minus_one);
    const x2 = x0 + two_g2_minus_one;
    const y2 = y0 + two_g2_minus_one;

    const h0 = hash2x8(seed, x0p, y0p) & @as(U32x8, @splat(15));
    const h1 = hash2x8(seed, x1p, y1p) & @as(U32x8, @splat(15));
    const h2 = hash2x8(seed, x2p, y2p) & @as(U32x8, @splat(15));

    const sum =
        simplexCornerSample2d8F16FromF32(h0, x0, y0) +
        simplexCornerSample2d8F16FromF32(h1, x1, y1) +
        simplexCornerSample2d8F16FromF32(h2, x2, y2);
    return @as(F32x8, @splat(70.0)) * sum;
}

inline fn simplexNoise2d8ShiftedSquare(seed: u32, x: F32x8, y: F32x8) F32x8 {
    const s = (x + y) * @as(F32x8, @splat(f2));
    const i = floorToI32x8(x + s);
    const j = floorToI32x8(y + s);

    const x0p = @as(U32x8, @bitCast(i)) *% @as(U32x8, @splat(prime_x));
    const y0p = @as(U32x8, @bitCast(j)) *% @as(U32x8, @splat(prime_y));

    const t = @as(F32x8, @floatFromInt(i +% j)) * @as(F32x8, @splat(g2));
    const cell_x = @as(F32x8, @floatFromInt(i)) - t;
    const cell_y = @as(F32x8, @floatFromInt(j)) - t;

    const x0 = x - cell_x;
    const y0 = y - cell_y;
    const x_ge_y = x0 >= y0;

    const zero_u = @as(U32x8, @splat(0));
    const g2v = @as(F32x8, @splat(g2));
    const g2_minus_one = @as(F32x8, @splat(g2 - 1.0));
    const two_g2_minus_one = @as(F32x8, @splat(2.0 * g2 - 1.0));

    const x1p = x0p +% @select(u32, x_ge_y, @as(U32x8, @splat(prime_x)), zero_u);
    const y1p = y0p +% @select(u32, x_ge_y, zero_u, @as(U32x8, @splat(prime_y)));
    const x2p = x0p +% @as(U32x8, @splat(prime_x));
    const y2p = y0p +% @as(U32x8, @splat(prime_y));

    const x1 = x0 + @select(f32, x_ge_y, g2_minus_one, g2v);
    const y1 = y0 + @select(f32, x_ge_y, g2v, g2_minus_one);
    const x2 = x0 + two_g2_minus_one;
    const y2 = y0 + two_g2_minus_one;

    const h0 = hash2x8(seed, x0p, y0p) & @as(U32x8, @splat(15));
    const h1 = hash2x8(seed, x1p, y1p) & @as(U32x8, @splat(15));
    const h2 = hash2x8(seed, x2p, y2p) & @as(U32x8, @splat(15));

    const sum =
        simplexCornerSample2d8F16FromF32ShiftedSquare(h0, x0, y0) +
        simplexCornerSample2d8F16FromF32ShiftedSquare(h1, x1, y1) +
        simplexCornerSample2d8F16FromF32ShiftedSquare(h2, x2, y2);
    return @as(F32x8, @splat(simplex_output_scale_2d_shifted_square)) * sum;
}

pub fn simplexNoiseUniformGrid2d(
    noalias noise_out: []f32,
    x_offset: f32,
    y_offset: f32,
    x_count: usize,
    y_count: usize,
    x_step_size: f32,
    y_step_size: f32,
    seed: u32,
    frequency: f32,
) void {
    const total = x_count * y_count;
    std.debug.assert(noise_out.len >= total);

    const x_base = x_offset * frequency;
    const y_base = y_offset * frequency;
    const x_step = x_step_size * frequency;
    const y_step = y_step_size * frequency;

    const lane: F32x8 = .{ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 };
    const x_step_v = @as(F32x8, @splat(x_step));
    const x_step8 = @as(F32x8, @splat(8.0 * x_step));

    var index: usize = 0;
    var y = y_base;
    var y_index: usize = 0;
    while (y_index < y_count) : ({
        y_index += 1;
        y += y_step;
    }) {
        const y8 = @as(F32x8, @splat(y));
        var x8 = @as(F32x8, @splat(x_base)) + lane * x_step_v;

        var x_index: usize = 0;
        while (x_index + 8 <= x_count) : ({
            x_index += 8;
            x8 += x_step8;
            index += 8;
        }) {
            store8(noise_out, index, simplexNoise2d8(seed, x8, y8));
        }

        var x = x_base + @as(f32, @floatFromInt(x_index)) * x_step;
        while (x_index < x_count) : ({
            x_index += 1;
            x += x_step;
            index += 1;
        }) {
            noise_out[index] = simplexNoise2d(seed, x, y);
        }
    }
}

pub fn simplexNoiseUniformGrid2dShiftedSquare(
    noalias noise_out: []f32,
    x_offset: f32,
    y_offset: f32,
    x_count: usize,
    y_count: usize,
    x_step_size: f32,
    y_step_size: f32,
    seed: u32,
    frequency: f32,
) void {
    const total = x_count * y_count;
    std.debug.assert(noise_out.len >= total);

    const x_base = x_offset * frequency;
    const y_base = y_offset * frequency;
    const x_step = x_step_size * frequency;
    const y_step = y_step_size * frequency;

    const lane: F32x8 = .{ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 };
    const x_step_v = @as(F32x8, @splat(x_step));
    const x_step8 = @as(F32x8, @splat(8.0 * x_step));

    var index: usize = 0;
    var y = y_base;
    var y_index: usize = 0;
    while (y_index < y_count) : ({
        y_index += 1;
        y += y_step;
    }) {
        const y8 = @as(F32x8, @splat(y));
        var x8 = @as(F32x8, @splat(x_base)) + lane * x_step_v;

        var x_index: usize = 0;
        while (x_index + 8 <= x_count) : ({
            x_index += 8;
            x8 += x_step8;
            index += 8;
        }) {
            store8(noise_out, index, simplexNoise2d8ShiftedSquare(seed, x8, y8));
        }

        var x = x_base + @as(f32, @floatFromInt(x_index)) * x_step;
        while (x_index < x_count) : ({
            x_index += 1;
            x += x_step;
            index += 1;
        }) {
            noise_out[index] = simplexNoise2dShiftedSquare(seed, x, y);
        }
    }
}

pub fn simplexNoisePositionArray2d(
    noalias noise_out: []f32,
    count: usize,
    noalias x_positions: []const f32,
    noalias y_positions: []const f32,
    x_offset: f32,
    y_offset: f32,
    seed: u32,
    frequency: f32,
) void {
    std.debug.assert(noise_out.len >= count);
    std.debug.assert(x_positions.len >= count);
    std.debug.assert(y_positions.len >= count);

    const x_offset_scaled = x_offset * frequency;
    const y_offset_scaled = y_offset * frequency;
    const frequency_v = @as(F32x8, @splat(frequency));
    const x_offset_scaled_v = @as(F32x8, @splat(x_offset_scaled));
    const y_offset_scaled_v = @as(F32x8, @splat(y_offset_scaled));

    var index: usize = 0;
    while (index + 8 <= count) : (index += 8) {
        const x8 = @mulAdd(F32x8, loadF32x8(x_positions.ptr + index), frequency_v, x_offset_scaled_v);
        const y8 = @mulAdd(F32x8, loadF32x8(y_positions.ptr + index), frequency_v, y_offset_scaled_v);
        store8(noise_out, index, simplexNoise2d8(seed, x8, y8));
    }

    while (index < count) : (index += 1) {
        noise_out[index] = simplexNoise2d(
            seed,
            @mulAdd(f32, x_positions[index], frequency, x_offset_scaled),
            @mulAdd(f32, y_positions[index], frequency, y_offset_scaled),
        );
    }
}

pub fn simplexNoisePositionArray2dShiftedSquare(
    noalias noise_out: []f32,
    count: usize,
    noalias x_positions: []const f32,
    noalias y_positions: []const f32,
    x_offset: f32,
    y_offset: f32,
    seed: u32,
    frequency: f32,
) void {
    std.debug.assert(noise_out.len >= count);
    std.debug.assert(x_positions.len >= count);
    std.debug.assert(y_positions.len >= count);

    const x_offset_scaled = x_offset * frequency;
    const y_offset_scaled = y_offset * frequency;
    const frequency_v = @as(F32x8, @splat(frequency));
    const x_offset_scaled_v = @as(F32x8, @splat(x_offset_scaled));
    const y_offset_scaled_v = @as(F32x8, @splat(y_offset_scaled));

    var index: usize = 0;
    while (index + 8 <= count) : (index += 8) {
        const x8 = @mulAdd(F32x8, loadF32x8(x_positions.ptr + index), frequency_v, x_offset_scaled_v);
        const y8 = @mulAdd(F32x8, loadF32x8(y_positions.ptr + index), frequency_v, y_offset_scaled_v);
        store8(noise_out, index, simplexNoise2d8ShiftedSquare(seed, x8, y8));
    }

    while (index < count) : (index += 1) {
        noise_out[index] = simplexNoise2dShiftedSquare(
            seed,
            @mulAdd(f32, x_positions[index], frequency, x_offset_scaled),
            @mulAdd(f32, y_positions[index], frequency, y_offset_scaled),
        );
    }
}

pub fn simplexNoise3d(seed: u32, x_input: f32, y_input: f32, z_input: f32, frequency: f32) f32 {
    const x = x_input * frequency;
    const y = y_input * frequency;
    const z = z_input * frequency;

    const s = (x + y + z) * f3;

    const i: i32 = @floor(x + s);
    const j: i32 = @floor(y + s);
    const k: i32 = @floor(z + s);

    const iu: u32 = @bitCast(i);
    const ju: u32 = @bitCast(j);
    const ku: u32 = @bitCast(k);

    const x0p = iu *% prime_x;
    const y0p = ju *% prime_y;
    const z0p = ku *% prime_z;

    const t = @as(f32, @floatFromInt(i + j + k)) * g3;

    const cell_x = @as(f32, @floatFromInt(i)) - t;
    const cell_y = @as(f32, @floatFromInt(j)) - t;
    const cell_z = @as(f32, @floatFromInt(k)) - t;

    const x0 = x - cell_x;
    const y0 = y - cell_y;
    const z0 = z - cell_z;

    const x_ge_y: u32 = @intFromBool(x0 >= y0);
    const x_ge_z: u32 = @intFromBool(x0 >= z0);
    const y_ge_z: u32 = @intFromBool(y0 >= z0);

    const x_lt_y = 1 - x_ge_y;
    const x_lt_z = 1 - x_ge_z;
    const y_lt_z = 1 - y_ge_z;

    const i1_mask = x_ge_y & x_ge_z;
    const j1_mask = x_lt_y & y_ge_z;
    const k1_mask = x_lt_z & y_lt_z;

    const i2_mask = x_ge_y | x_ge_z;
    const j2_mask = x_lt_y | y_ge_z;
    const k2_mask = x_lt_z | y_lt_z;

    const x1p = x0p +% primeIf(i1_mask, prime_x);
    const y1p = y0p +% primeIf(j1_mask, prime_y);
    const z1p = z0p +% primeIf(k1_mask, prime_z);

    const x2p = x0p +% primeIf(i2_mask, prime_x);
    const y2p = y0p +% primeIf(j2_mask, prime_y);
    const z2p = z0p +% primeIf(k2_mask, prime_z);

    const x3p = x0p +% prime_x;
    const y3p = y0p +% prime_y;
    const z3p = z0p +% prime_z;

    const x1 = x0 - @as(f32, @floatFromInt(i1_mask)) + g3_drift_1;
    const y1 = y0 - @as(f32, @floatFromInt(j1_mask)) + g3_drift_1;
    const z1 = z0 - @as(f32, @floatFromInt(k1_mask)) + g3_drift_1;

    const x2 = x0 - @as(f32, @floatFromInt(i2_mask)) + g3_drift_2;
    const y2 = y0 - @as(f32, @floatFromInt(j2_mask)) + g3_drift_2;
    const z2 = z0 - @as(f32, @floatFromInt(k2_mask)) + g3_drift_2;

    const x3 = x0 - 0.5;
    const y3 = y0 - 0.5;
    const z3 = z0 - 0.5;

    const n =
        simplexCornerSample3dF32(seed, x0p, y0p, z0p, x0, y0, z0) +
        simplexCornerSample3dF32(seed, x1p, y1p, z1p, x1, y1, z1) +
        simplexCornerSample3dF32(seed, x2p, y2p, z2p, x2, y2, z2) +
        simplexCornerSample3dF32(seed, x3p, y3p, z3p, x3, y3, z3);

    return 32.0 * n;
}

pub fn simplexNoise3dShiftedSquare(seed: u32, x_input: f32, y_input: f32, z_input: f32, frequency: f32) f32 {
    const x = x_input * frequency;
    const y = y_input * frequency;
    const z = z_input * frequency;

    const s = (x + y + z) * f3;

    const i: i32 = @floor(x + s);
    const j: i32 = @floor(y + s);
    const k: i32 = @floor(z + s);

    const iu: u32 = @bitCast(i);
    const ju: u32 = @bitCast(j);
    const ku: u32 = @bitCast(k);

    const x0p = iu *% prime_x;
    const y0p = ju *% prime_y;
    const z0p = ku *% prime_z;

    const t = @as(f32, @floatFromInt(i + j + k)) * g3;

    const cell_x = @as(f32, @floatFromInt(i)) - t;
    const cell_y = @as(f32, @floatFromInt(j)) - t;
    const cell_z = @as(f32, @floatFromInt(k)) - t;

    const x0 = x - cell_x;
    const y0 = y - cell_y;
    const z0 = z - cell_z;

    const x_ge_y: u32 = @intFromBool(x0 >= y0);
    const x_ge_z: u32 = @intFromBool(x0 >= z0);
    const y_ge_z: u32 = @intFromBool(y0 >= z0);

    const x_lt_y = 1 - x_ge_y;
    const x_lt_z = 1 - x_ge_z;
    const y_lt_z = 1 - y_ge_z;

    const i1_mask = x_ge_y & x_ge_z;
    const j1_mask = x_lt_y & y_ge_z;
    const k1_mask = x_lt_z & y_lt_z;

    const i2_mask = x_ge_y | x_ge_z;
    const j2_mask = x_lt_y | y_ge_z;
    const k2_mask = x_lt_z | y_lt_z;

    const x1p = x0p +% primeIf(i1_mask, prime_x);
    const y1p = y0p +% primeIf(j1_mask, prime_y);
    const z1p = z0p +% primeIf(k1_mask, prime_z);

    const x2p = x0p +% primeIf(i2_mask, prime_x);
    const y2p = y0p +% primeIf(j2_mask, prime_y);
    const z2p = z0p +% primeIf(k2_mask, prime_z);

    const x3p = x0p +% prime_x;
    const y3p = y0p +% prime_y;
    const z3p = z0p +% prime_z;

    const x1 = x0 - @as(f32, @floatFromInt(i1_mask)) + g3_drift_1;
    const y1 = y0 - @as(f32, @floatFromInt(j1_mask)) + g3_drift_1;
    const z1 = z0 - @as(f32, @floatFromInt(k1_mask)) + g3_drift_1;

    const x2 = x0 - @as(f32, @floatFromInt(i2_mask)) + g3_drift_2;
    const y2 = y0 - @as(f32, @floatFromInt(j2_mask)) + g3_drift_2;
    const z2 = z0 - @as(f32, @floatFromInt(k2_mask)) + g3_drift_2;

    const x3 = x0 - 0.5;
    const y3 = y0 - 0.5;
    const z3 = z0 - 0.5;

    const n =
        simplexCornerSample3dF32ShiftedSquare(seed, x0p, y0p, z0p, x0, y0, z0) +
        simplexCornerSample3dF32ShiftedSquare(seed, x1p, y1p, z1p, x1, y1, z1) +
        simplexCornerSample3dF32ShiftedSquare(seed, x2p, y2p, z2p, x2, y2, z2) +
        simplexCornerSample3dF32ShiftedSquare(seed, x3p, y3p, z3p, x3, y3, z3);

    return simplex_output_scale_3d_shifted_square * n;
}

inline fn simplexNoise3d8(seed: u32, x: F32x8, y: F32x8, z: F32x8) F32x8 {
    const s = (x + y + z) * @as(F32x8, @splat(f3));
    const i = floorToI32x8(x + s);
    const j = floorToI32x8(y + s);
    const k = floorToI32x8(z + s);

    const x0p = @as(U32x8, @bitCast(i)) *% @as(U32x8, @splat(prime_x));
    const y0p = @as(U32x8, @bitCast(j)) *% @as(U32x8, @splat(prime_y));
    const z0p = @as(U32x8, @bitCast(k)) *% @as(U32x8, @splat(prime_z));

    const t = @as(F32x8, @floatFromInt(i +% j +% k)) * @as(F32x8, @splat(g3));
    const cell_x = @as(F32x8, @floatFromInt(i)) - t;
    const cell_y = @as(F32x8, @floatFromInt(j)) - t;
    const cell_z = @as(F32x8, @floatFromInt(k)) - t;

    const x0 = x - cell_x;
    const y0 = y - cell_y;
    const z0 = z - cell_z;

    const x_ge_y = x0 >= y0;
    const x_ge_z = x0 >= z0;
    const y_ge_z = y0 >= z0;

    const i1_mask = x_ge_y & x_ge_z;
    const j1_mask = !x_ge_y & y_ge_z;
    const k1_mask = !x_ge_z & !y_ge_z;
    const i2_mask = x_ge_y | x_ge_z;
    const j2_mask = !x_ge_y | y_ge_z;
    const k2_mask = !x_ge_z | !y_ge_z;

    const zero_u = @as(U32x8, @splat(0));
    const seed_v = @as(U32x8, @splat(seed));

    const x1p = x0p +% @select(u32, i1_mask, @as(U32x8, @splat(prime_x)), zero_u);
    const y1p = y0p +% @select(u32, j1_mask, @as(U32x8, @splat(prime_y)), zero_u);
    const z1p = z0p +% @select(u32, k1_mask, @as(U32x8, @splat(prime_z)), zero_u);

    const x2p = x0p +% @select(u32, i2_mask, @as(U32x8, @splat(prime_x)), zero_u);
    const y2p = y0p +% @select(u32, j2_mask, @as(U32x8, @splat(prime_y)), zero_u);
    const z2p = z0p +% @select(u32, k2_mask, @as(U32x8, @splat(prime_z)), zero_u);

    const x3p = x0p +% @as(U32x8, @splat(prime_x));
    const y3p = y0p +% @as(U32x8, @splat(prime_y));
    const z3p = z0p +% @as(U32x8, @splat(prime_z));

    const zero_h = @as(F16x8, @splat(@as(f16, 0.0)));
    const one_h = @as(F16x8, @splat(@as(f16, 1.0)));
    const g3_h = @as(F16x8, @splat(@as(f16, g3_drift_1)));
    const two_g3_h = @as(F16x8, @splat(@as(f16, g3_drift_2)));
    const neg_half_h = @as(F16x8, @splat(@as(f16, -0.5)));

    const x0h: F16x8 = @floatCast(x0);
    const y0h: F16x8 = @floatCast(y0);
    const z0h: F16x8 = @floatCast(z0);

    const x1h = x0h - @select(f16, i1_mask, one_h, zero_h) + g3_h;
    const y1h = y0h - @select(f16, j1_mask, one_h, zero_h) + g3_h;
    const z1h = z0h - @select(f16, k1_mask, one_h, zero_h) + g3_h;

    const x2h = x0h - @select(f16, i2_mask, one_h, zero_h) + two_g3_h;
    const y2h = y0h - @select(f16, j2_mask, one_h, zero_h) + two_g3_h;
    const z2h = z0h - @select(f16, k2_mask, one_h, zero_h) + two_g3_h;

    const x3h = x0h + neg_half_h;
    const y3h = y0h + neg_half_h;
    const z3h = z0h + neg_half_h;

    const c0: F32x8 = @floatCast(simplexCornerSample3d8F16Preconverted(seed_v, x0p, y0p, z0p, x0h, y0h, z0h));
    const c1: F32x8 = @floatCast(simplexCornerSample3d8F16Preconverted(seed_v, x1p, y1p, z1p, x1h, y1h, z1h));
    const c2: F32x8 = @floatCast(simplexCornerSample3d8F16Preconverted(seed_v, x2p, y2p, z2p, x2h, y2h, z2h));
    const c3: F32x8 = @floatCast(simplexCornerSample3d8F16Preconverted(seed_v, x3p, y3p, z3p, x3h, y3h, z3h));

    return @as(F32x8, @splat(32.0)) * (c0 + c1 + c2 + c3);
}

inline fn simplexNoise3d8ShiftedSquare(seed: u32, x: F32x8, y: F32x8, z: F32x8) F32x8 {
    const s = (x + y + z) * @as(F32x8, @splat(f3));
    const i = floorToI32x8(x + s);
    const j = floorToI32x8(y + s);
    const k = floorToI32x8(z + s);

    const x0p = @as(U32x8, @bitCast(i)) *% @as(U32x8, @splat(prime_x));
    const y0p = @as(U32x8, @bitCast(j)) *% @as(U32x8, @splat(prime_y));
    const z0p = @as(U32x8, @bitCast(k)) *% @as(U32x8, @splat(prime_z));

    const t = @as(F32x8, @floatFromInt(i +% j +% k)) * @as(F32x8, @splat(g3));
    const cell_x = @as(F32x8, @floatFromInt(i)) - t;
    const cell_y = @as(F32x8, @floatFromInt(j)) - t;
    const cell_z = @as(F32x8, @floatFromInt(k)) - t;

    const x0 = x - cell_x;
    const y0 = y - cell_y;
    const z0 = z - cell_z;

    const x_ge_y = x0 >= y0;
    const x_ge_z = x0 >= z0;
    const y_ge_z = y0 >= z0;

    const i1_mask = x_ge_y & x_ge_z;
    const j1_mask = !x_ge_y & y_ge_z;
    const k1_mask = !x_ge_z & !y_ge_z;
    const i2_mask = x_ge_y | x_ge_z;
    const j2_mask = !x_ge_y | y_ge_z;
    const k2_mask = !x_ge_z | !y_ge_z;

    const zero_u = @as(U32x8, @splat(0));
    const seed_v = @as(U32x8, @splat(seed));

    const x1p = x0p +% @select(u32, i1_mask, @as(U32x8, @splat(prime_x)), zero_u);
    const y1p = y0p +% @select(u32, j1_mask, @as(U32x8, @splat(prime_y)), zero_u);
    const z1p = z0p +% @select(u32, k1_mask, @as(U32x8, @splat(prime_z)), zero_u);

    const x2p = x0p +% @select(u32, i2_mask, @as(U32x8, @splat(prime_x)), zero_u);
    const y2p = y0p +% @select(u32, j2_mask, @as(U32x8, @splat(prime_y)), zero_u);
    const z2p = z0p +% @select(u32, k2_mask, @as(U32x8, @splat(prime_z)), zero_u);

    const x3p = x0p +% @as(U32x8, @splat(prime_x));
    const y3p = y0p +% @as(U32x8, @splat(prime_y));
    const z3p = z0p +% @as(U32x8, @splat(prime_z));

    const zero_h = @as(F16x8, @splat(@as(f16, 0.0)));
    const one_h = @as(F16x8, @splat(@as(f16, 1.0)));
    const g3_h = @as(F16x8, @splat(@as(f16, g3_drift_1)));
    const two_g3_h = @as(F16x8, @splat(@as(f16, g3_drift_2)));
    const neg_half_h = @as(F16x8, @splat(@as(f16, -0.5)));

    const x0h: F16x8 = @floatCast(x0);
    const y0h: F16x8 = @floatCast(y0);
    const z0h: F16x8 = @floatCast(z0);

    const x1h = x0h - @select(f16, i1_mask, one_h, zero_h) + g3_h;
    const y1h = y0h - @select(f16, j1_mask, one_h, zero_h) + g3_h;
    const z1h = z0h - @select(f16, k1_mask, one_h, zero_h) + g3_h;

    const x2h = x0h - @select(f16, i2_mask, one_h, zero_h) + two_g3_h;
    const y2h = y0h - @select(f16, j2_mask, one_h, zero_h) + two_g3_h;
    const z2h = z0h - @select(f16, k2_mask, one_h, zero_h) + two_g3_h;

    const x3h = x0h + neg_half_h;
    const y3h = y0h + neg_half_h;
    const z3h = z0h + neg_half_h;

    const c0: F32x8 = @floatCast(simplexCornerSample3d8F16PreconvertedShiftedSquare(seed_v, x0p, y0p, z0p, x0h, y0h, z0h));
    const c1: F32x8 = @floatCast(simplexCornerSample3d8F16PreconvertedShiftedSquare(seed_v, x1p, y1p, z1p, x1h, y1h, z1h));
    const c2: F32x8 = @floatCast(simplexCornerSample3d8F16PreconvertedShiftedSquare(seed_v, x2p, y2p, z2p, x2h, y2h, z2h));
    const c3: F32x8 = @floatCast(simplexCornerSample3d8F16PreconvertedShiftedSquare(seed_v, x3p, y3p, z3p, x3h, y3h, z3h));

    return @as(F32x8, @splat(simplex_output_scale_3d_shifted_square)) * (c0 + c1 + c2 + c3);
}

inline fn simplexNoise3d8Nonnegative(seed: u32, x: F32x8, y: F32x8, z: F32x8) F32x8 {
    const s = (x + y + z) * @as(F32x8, @splat(f3));
    debugAssertNonnegativeSimplexDomain8(x, y, z, s);
    const i = truncToI32x8(x + s);
    const j = truncToI32x8(y + s);
    const k = truncToI32x8(z + s);

    const x0p = @as(U32x8, @bitCast(i)) *% @as(U32x8, @splat(prime_x));
    const y0p = @as(U32x8, @bitCast(j)) *% @as(U32x8, @splat(prime_y));
    const z0p = @as(U32x8, @bitCast(k)) *% @as(U32x8, @splat(prime_z));

    const t = @as(F32x8, @floatFromInt(i +% j +% k)) * @as(F32x8, @splat(g3));
    const cell_x = @as(F32x8, @floatFromInt(i)) - t;
    const cell_y = @as(F32x8, @floatFromInt(j)) - t;
    const cell_z = @as(F32x8, @floatFromInt(k)) - t;

    const x0 = x - cell_x;
    const y0 = y - cell_y;
    const z0 = z - cell_z;

    const x_ge_y = x0 >= y0;
    const x_ge_z = x0 >= z0;
    const y_ge_z = y0 >= z0;

    const i1_mask = x_ge_y & x_ge_z;
    const j1_mask = !x_ge_y & y_ge_z;
    const k1_mask = !x_ge_z & !y_ge_z;
    const i2_mask = x_ge_y | x_ge_z;
    const j2_mask = !x_ge_y | y_ge_z;
    const k2_mask = !x_ge_z | !y_ge_z;

    const zero_u = @as(U32x8, @splat(0));
    const seed_v = @as(U32x8, @splat(seed));

    const x1p = x0p +% @select(u32, i1_mask, @as(U32x8, @splat(prime_x)), zero_u);
    const y1p = y0p +% @select(u32, j1_mask, @as(U32x8, @splat(prime_y)), zero_u);
    const z1p = z0p +% @select(u32, k1_mask, @as(U32x8, @splat(prime_z)), zero_u);

    const x2p = x0p +% @select(u32, i2_mask, @as(U32x8, @splat(prime_x)), zero_u);
    const y2p = y0p +% @select(u32, j2_mask, @as(U32x8, @splat(prime_y)), zero_u);
    const z2p = z0p +% @select(u32, k2_mask, @as(U32x8, @splat(prime_z)), zero_u);

    const x3p = x0p +% @as(U32x8, @splat(prime_x));
    const y3p = y0p +% @as(U32x8, @splat(prime_y));
    const z3p = z0p +% @as(U32x8, @splat(prime_z));

    const zero_h = @as(F16x8, @splat(@as(f16, 0.0)));
    const one_h = @as(F16x8, @splat(@as(f16, 1.0)));
    const g3_h = @as(F16x8, @splat(@as(f16, g3_drift_1)));
    const two_g3_h = @as(F16x8, @splat(@as(f16, g3_drift_2)));
    const neg_half_h = @as(F16x8, @splat(@as(f16, -0.5)));

    const x0h: F16x8 = @floatCast(x0);
    const y0h: F16x8 = @floatCast(y0);
    const z0h: F16x8 = @floatCast(z0);

    const x1h = x0h - @select(f16, i1_mask, one_h, zero_h) + g3_h;
    const y1h = y0h - @select(f16, j1_mask, one_h, zero_h) + g3_h;
    const z1h = z0h - @select(f16, k1_mask, one_h, zero_h) + g3_h;

    const x2h = x0h - @select(f16, i2_mask, one_h, zero_h) + two_g3_h;
    const y2h = y0h - @select(f16, j2_mask, one_h, zero_h) + two_g3_h;
    const z2h = z0h - @select(f16, k2_mask, one_h, zero_h) + two_g3_h;

    const x3h = x0h + neg_half_h;
    const y3h = y0h + neg_half_h;
    const z3h = z0h + neg_half_h;

    const c0: F32x8 = @floatCast(simplexCornerSample3d8F16Preconverted(seed_v, x0p, y0p, z0p, x0h, y0h, z0h));
    const c1: F32x8 = @floatCast(simplexCornerSample3d8F16Preconverted(seed_v, x1p, y1p, z1p, x1h, y1h, z1h));
    const c2: F32x8 = @floatCast(simplexCornerSample3d8F16Preconverted(seed_v, x2p, y2p, z2p, x2h, y2h, z2h));
    const c3: F32x8 = @floatCast(simplexCornerSample3d8F16Preconverted(seed_v, x3p, y3p, z3p, x3h, y3h, z3h));

    return @as(F32x8, @splat(32.0)) * (c0 + c1 + c2 + c3);
}

pub fn simplexNoiseUniformGrid3d(
    noalias noise_out: []f32,
    x_offset: f32,
    y_offset: f32,
    z_offset: f32,
    x_count: usize,
    y_count: usize,
    z_count: usize,
    x_step_size: f32,
    y_step_size: f32,
    z_step_size: f32,
    seed: u32,
    frequency: f32,
) void {
    const total = x_count * y_count * z_count;
    std.debug.assert(noise_out.len >= total);

    const x_base = x_offset * frequency;
    const y_base = y_offset * frequency;
    const z_base = z_offset * frequency;
    const x_step = x_step_size * frequency;
    const y_step = y_step_size * frequency;
    const z_step = z_step_size * frequency;

    const lane: F32x8 = .{ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 };
    const x_step_v = @as(F32x8, @splat(x_step));
    const x_step8 = @as(F32x8, @splat(8.0 * x_step));

    var index: usize = 0;
    var z = z_base;
    var z_index: usize = 0;
    while (z_index < z_count) : ({
        z_index += 1;
        z += z_step;
    }) {
        const z8 = @as(F32x8, @splat(z));
        var y = y_base;
        var y_index: usize = 0;
        while (y_index < y_count) : ({
            y_index += 1;
            y += y_step;
        }) {
            const y8 = @as(F32x8, @splat(y));
            var x8 = @as(F32x8, @splat(x_base)) + lane * x_step_v;

            var x_index: usize = 0;
            while (x_index + 8 <= x_count) : ({
                x_index += 8;
                x8 += x_step8;
                index += 8;
            }) {
                store8(noise_out, index, simplexNoise3d8(seed, x8, y8, z8));
            }

            var x = x_base + @as(f32, @floatFromInt(x_index)) * x_step;
            while (x_index < x_count) : ({
                x_index += 1;
                x += x_step;
                index += 1;
            }) {
                noise_out[index] = simplexNoise3d(seed, x, y, z, 1.0);
            }
        }
    }
}

pub fn simplexNoiseUniformGrid3dShiftedSquare(
    noalias noise_out: []f32,
    x_offset: f32,
    y_offset: f32,
    z_offset: f32,
    x_count: usize,
    y_count: usize,
    z_count: usize,
    x_step_size: f32,
    y_step_size: f32,
    z_step_size: f32,
    seed: u32,
    frequency: f32,
) void {
    const total = x_count * y_count * z_count;
    std.debug.assert(noise_out.len >= total);

    const x_base = x_offset * frequency;
    const y_base = y_offset * frequency;
    const z_base = z_offset * frequency;
    const x_step = x_step_size * frequency;
    const y_step = y_step_size * frequency;
    const z_step = z_step_size * frequency;

    const lane: F32x8 = .{ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 };
    const x_step_v = @as(F32x8, @splat(x_step));
    const x_step8 = @as(F32x8, @splat(8.0 * x_step));

    var index: usize = 0;
    var z = z_base;
    var z_index: usize = 0;
    while (z_index < z_count) : ({
        z_index += 1;
        z += z_step;
    }) {
        const z8 = @as(F32x8, @splat(z));
        var y = y_base;
        var y_index: usize = 0;
        while (y_index < y_count) : ({
            y_index += 1;
            y += y_step;
        }) {
            const y8 = @as(F32x8, @splat(y));
            var x8 = @as(F32x8, @splat(x_base)) + lane * x_step_v;

            var x_index: usize = 0;
            while (x_index + 8 <= x_count) : ({
                x_index += 8;
                x8 += x_step8;
                index += 8;
            }) {
                store8(noise_out, index, simplexNoise3d8ShiftedSquare(seed, x8, y8, z8));
            }

            var x = x_base + @as(f32, @floatFromInt(x_index)) * x_step;
            while (x_index < x_count) : ({
                x_index += 1;
                x += x_step;
                index += 1;
            }) {
                noise_out[index] = simplexNoise3dShiftedSquare(seed, x, y, z, 1.0);
            }
        }
    }
}

pub fn simplexNoiseChunk3d(
    comptime side_count: usize,
    noalias noise_out: []f32,
    origin_x: f32,
    origin_y: f32,
    origin_z: f32,
    block_size: f32,
    seed: u32,
    frequency: f32,
) void {
    comptime {
        if (side_count != 16 and side_count != 32 and side_count != 64) {
            @compileError("simplexNoiseChunk3d only supports 16, 32, or 64");
        }
    }

    const total = side_count * side_count * side_count;
    std.debug.assert(noise_out.len >= total);

    const x_base = origin_x * frequency;
    const y_base = origin_y * frequency;
    const z_base = origin_z * frequency;
    const step = block_size * frequency;

    const lane: F32x8 = .{ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 };
    const step_v = @as(F32x8, @splat(step));
    const step8 = @as(F32x8, @splat(8.0 * step));

    var index: usize = 0;
    var z = z_base;
    var z_index: usize = 0;
    while (z_index < side_count) : ({
        z_index += 1;
        z += step;
    }) {
        const z8 = @as(F32x8, @splat(z));
        var y = y_base;
        var y_index: usize = 0;
        while (y_index < side_count) : ({
            y_index += 1;
            y += step;
        }) {
            const y8 = @as(F32x8, @splat(y));
            var x8 = @as(F32x8, @splat(x_base)) + lane * step_v;

            var x_index: usize = 0;
            while (x_index < side_count) : ({
                x_index += 8;
                x8 += step8;
                index += 8;
            }) {
                store8(noise_out, index, simplexNoise3d8(seed, x8, y8, z8));
            }
        }
    }
}

pub fn simplexNoiseChunk16_3d(
    noalias noise_out: []f32,
    origin_x: f32,
    origin_y: f32,
    origin_z: f32,
    block_size: f32,
    seed: u32,
    frequency: f32,
) void {
    simplexNoiseChunk3d(16, noise_out, origin_x, origin_y, origin_z, block_size, seed, frequency);
}

pub fn simplexNoiseChunk32_3d(
    noalias noise_out: []f32,
    origin_x: f32,
    origin_y: f32,
    origin_z: f32,
    block_size: f32,
    seed: u32,
    frequency: f32,
) void {
    simplexNoiseChunk3d(32, noise_out, origin_x, origin_y, origin_z, block_size, seed, frequency);
}

pub fn simplexNoiseChunk64_3d(
    noalias noise_out: []f32,
    origin_x: f32,
    origin_y: f32,
    origin_z: f32,
    block_size: f32,
    seed: u32,
    frequency: f32,
) void {
    simplexNoiseChunk3d(64, noise_out, origin_x, origin_y, origin_z, block_size, seed, frequency);
}

pub fn simplexNoisePositionArray3d(
    noalias noise_out: []f32,
    count: usize,
    noalias x_positions: []const f32,
    noalias y_positions: []const f32,
    noalias z_positions: []const f32,
    x_offset: f32,
    y_offset: f32,
    z_offset: f32,
    seed: u32,
    frequency: f32,
) void {
    std.debug.assert(noise_out.len >= count);
    std.debug.assert(x_positions.len >= count);
    std.debug.assert(y_positions.len >= count);
    std.debug.assert(z_positions.len >= count);

    const frequency_v = @as(F32x8, @splat(frequency));
    const x_offset_scaled = @as(F32x8, @splat(x_offset * frequency));
    const y_offset_scaled = @as(F32x8, @splat(y_offset * frequency));
    const z_offset_scaled = @as(F32x8, @splat(z_offset * frequency));

    var index: usize = 0;
    while (index + 16 <= count) : (index += 16) {
        const x8 = @mulAdd(F32x8, loadF32x8(x_positions.ptr + index), frequency_v, x_offset_scaled);
        const y8 = @mulAdd(F32x8, loadF32x8(y_positions.ptr + index), frequency_v, y_offset_scaled);
        const z8 = @mulAdd(F32x8, loadF32x8(z_positions.ptr + index), frequency_v, z_offset_scaled);
        const x8_next = @mulAdd(F32x8, loadF32x8(x_positions.ptr + index + 8), frequency_v, x_offset_scaled);
        const y8_next = @mulAdd(F32x8, loadF32x8(y_positions.ptr + index + 8), frequency_v, y_offset_scaled);
        const z8_next = @mulAdd(F32x8, loadF32x8(z_positions.ptr + index + 8), frequency_v, z_offset_scaled);
        store8(noise_out, index, simplexNoise3d8(seed, x8, y8, z8));
        store8(noise_out, index + 8, simplexNoise3d8(seed, x8_next, y8_next, z8_next));
    }

    while (index + 8 <= count) : (index += 8) {
        const x8 = @mulAdd(F32x8, loadF32x8(x_positions.ptr + index), frequency_v, x_offset_scaled);
        const y8 = @mulAdd(F32x8, loadF32x8(y_positions.ptr + index), frequency_v, y_offset_scaled);
        const z8 = @mulAdd(F32x8, loadF32x8(z_positions.ptr + index), frequency_v, z_offset_scaled);
        store8(noise_out, index, simplexNoise3d8(seed, x8, y8, z8));
    }

    while (index < count) : (index += 1) {
        noise_out[index] = simplexNoise3d(
            seed,
            @mulAdd(f32, x_positions[index], frequency, x_offset * frequency),
            @mulAdd(f32, y_positions[index], frequency, y_offset * frequency),
            @mulAdd(f32, z_positions[index], frequency, z_offset * frequency),
            1.0,
        );
    }
}

pub fn simplexNoisePositionArray3dShiftedSquare(
    noalias noise_out: []f32,
    count: usize,
    noalias x_positions: []const f32,
    noalias y_positions: []const f32,
    noalias z_positions: []const f32,
    x_offset: f32,
    y_offset: f32,
    z_offset: f32,
    seed: u32,
    frequency: f32,
) void {
    std.debug.assert(noise_out.len >= count);
    std.debug.assert(x_positions.len >= count);
    std.debug.assert(y_positions.len >= count);
    std.debug.assert(z_positions.len >= count);

    const frequency_v = @as(F32x8, @splat(frequency));
    const x_offset_scaled = @as(F32x8, @splat(x_offset * frequency));
    const y_offset_scaled = @as(F32x8, @splat(y_offset * frequency));
    const z_offset_scaled = @as(F32x8, @splat(z_offset * frequency));

    var index: usize = 0;
    while (index + 16 <= count) : (index += 16) {
        const x8 = @mulAdd(F32x8, loadF32x8(x_positions.ptr + index), frequency_v, x_offset_scaled);
        const y8 = @mulAdd(F32x8, loadF32x8(y_positions.ptr + index), frequency_v, y_offset_scaled);
        const z8 = @mulAdd(F32x8, loadF32x8(z_positions.ptr + index), frequency_v, z_offset_scaled);
        const x8_next = @mulAdd(F32x8, loadF32x8(x_positions.ptr + index + 8), frequency_v, x_offset_scaled);
        const y8_next = @mulAdd(F32x8, loadF32x8(y_positions.ptr + index + 8), frequency_v, y_offset_scaled);
        const z8_next = @mulAdd(F32x8, loadF32x8(z_positions.ptr + index + 8), frequency_v, z_offset_scaled);
        store8(noise_out, index, simplexNoise3d8ShiftedSquare(seed, x8, y8, z8));
        store8(noise_out, index + 8, simplexNoise3d8ShiftedSquare(seed, x8_next, y8_next, z8_next));
    }

    while (index + 8 <= count) : (index += 8) {
        const x8 = @mulAdd(F32x8, loadF32x8(x_positions.ptr + index), frequency_v, x_offset_scaled);
        const y8 = @mulAdd(F32x8, loadF32x8(y_positions.ptr + index), frequency_v, y_offset_scaled);
        const z8 = @mulAdd(F32x8, loadF32x8(z_positions.ptr + index), frequency_v, z_offset_scaled);
        store8(noise_out, index, simplexNoise3d8ShiftedSquare(seed, x8, y8, z8));
    }

    while (index < count) : (index += 1) {
        noise_out[index] = simplexNoise3dShiftedSquare(
            seed,
            @mulAdd(f32, x_positions[index], frequency, x_offset * frequency),
            @mulAdd(f32, y_positions[index], frequency, y_offset * frequency),
            @mulAdd(f32, z_positions[index], frequency, z_offset * frequency),
            1.0,
        );
    }
}

pub fn simplexNoisePositionArray3dPrescaled(
    noalias noise_out: []f32,
    count: usize,
    noalias x_positions: []const f32,
    noalias y_positions: []const f32,
    noalias z_positions: []const f32,
    seed: u32,
) void {
    std.debug.assert(noise_out.len >= count);
    std.debug.assert(x_positions.len >= count);
    std.debug.assert(y_positions.len >= count);
    std.debug.assert(z_positions.len >= count);

    var index: usize = 0;
    while (index + 16 <= count) : (index += 16) {
        const x8 = loadF32x8(x_positions.ptr + index);
        const y8 = loadF32x8(y_positions.ptr + index);
        const z8 = loadF32x8(z_positions.ptr + index);
        const x8_next = loadF32x8(x_positions.ptr + index + 8);
        const y8_next = loadF32x8(y_positions.ptr + index + 8);
        const z8_next = loadF32x8(z_positions.ptr + index + 8);
        store8(noise_out, index, simplexNoise3d8(seed, x8, y8, z8));
        store8(noise_out, index + 8, simplexNoise3d8(seed, x8_next, y8_next, z8_next));
    }

    while (index + 8 <= count) : (index += 8) {
        const x8 = loadF32x8(x_positions.ptr + index);
        const y8 = loadF32x8(y_positions.ptr + index);
        const z8 = loadF32x8(z_positions.ptr + index);
        store8(noise_out, index, simplexNoise3d8(seed, x8, y8, z8));
    }

    while (index < count) : (index += 1) {
        noise_out[index] = simplexNoise3d(seed, x_positions[index], y_positions[index], z_positions[index], 1.0);
    }
}

pub fn simplexNoisePositionArray3dPreScaledPositive(
    noalias noise_out: []f32,
    count: usize,
    noalias x_positions: []const f32,
    noalias y_positions: []const f32,
    noalias z_positions: []const f32,
    seed: u32,
) void {
    std.debug.assert(noise_out.len >= count);
    std.debug.assert(x_positions.len >= count);
    std.debug.assert(y_positions.len >= count);
    std.debug.assert(z_positions.len >= count);

    var index: usize = 0;
    while (index + 16 <= count) : (index += 16) {
        const x8 = loadF32x8(x_positions.ptr + index);
        const y8 = loadF32x8(y_positions.ptr + index);
        const z8 = loadF32x8(z_positions.ptr + index);
        const x8_next = loadF32x8(x_positions.ptr + index + 8);
        const y8_next = loadF32x8(y_positions.ptr + index + 8);
        const z8_next = loadF32x8(z_positions.ptr + index + 8);
        store8(noise_out, index, simplexNoise3d8Nonnegative(seed, x8, y8, z8));
        store8(noise_out, index + 8, simplexNoise3d8Nonnegative(seed, x8_next, y8_next, z8_next));
    }

    while (index + 8 <= count) : (index += 8) {
        const x8 = loadF32x8(x_positions.ptr + index);
        const y8 = loadF32x8(y_positions.ptr + index);
        const z8 = loadF32x8(z_positions.ptr + index);
        store8(noise_out, index, simplexNoise3d8Nonnegative(seed, x8, y8, z8));
    }

    while (index < count) : (index += 1) {
        debugAssertNonnegativeSimplexDomainScalar(x_positions[index], y_positions[index], z_positions[index]);
        noise_out[index] = simplexNoise3d(seed, x_positions[index], y_positions[index], z_positions[index], 1.0);
    }
}

pub fn simplexNoisePositionArray3dPreScaledPositiveMultipleOf16(
    noalias noise_out: []f32,
    count: usize,
    noalias x_positions: []const f32,
    noalias y_positions: []const f32,
    noalias z_positions: []const f32,
    seed: u32,
) void {
    std.debug.assert(count % 16 == 0);
    std.debug.assert(noise_out.len >= count);
    std.debug.assert(x_positions.len >= count);
    std.debug.assert(y_positions.len >= count);
    std.debug.assert(z_positions.len >= count);

    var index: usize = 0;
    while (index < count) : (index += 16) {
        const x8 = loadF32x8(x_positions.ptr + index);
        const y8 = loadF32x8(y_positions.ptr + index);
        const z8 = loadF32x8(z_positions.ptr + index);
        const x8_next = loadF32x8(x_positions.ptr + index + 8);
        const y8_next = loadF32x8(y_positions.ptr + index + 8);
        const z8_next = loadF32x8(z_positions.ptr + index + 8);
        store8(noise_out, index, simplexNoise3d8Nonnegative(seed, x8, y8, z8));
        store8(noise_out, index + 8, simplexNoise3d8Nonnegative(seed, x8_next, y8_next, z8_next));
    }
}

pub fn simplexNoisePositionArray3dNonnegative(
    noalias noise_out: []f32,
    count: usize,
    noalias x_positions: []const f32,
    noalias y_positions: []const f32,
    noalias z_positions: []const f32,
    x_offset: f32,
    y_offset: f32,
    z_offset: f32,
    seed: u32,
    frequency: f32,
) void {
    std.debug.assert(noise_out.len >= count);
    std.debug.assert(x_positions.len >= count);
    std.debug.assert(y_positions.len >= count);
    std.debug.assert(z_positions.len >= count);

    const frequency_v = @as(F32x8, @splat(frequency));
    const x_offset_scaled = @as(F32x8, @splat(x_offset * frequency));
    const y_offset_scaled = @as(F32x8, @splat(y_offset * frequency));
    const z_offset_scaled = @as(F32x8, @splat(z_offset * frequency));

    var index: usize = 0;
    while (index + 16 <= count) : (index += 16) {
        const x8 = @mulAdd(F32x8, loadF32x8(x_positions.ptr + index), frequency_v, x_offset_scaled);
        const y8 = @mulAdd(F32x8, loadF32x8(y_positions.ptr + index), frequency_v, y_offset_scaled);
        const z8 = @mulAdd(F32x8, loadF32x8(z_positions.ptr + index), frequency_v, z_offset_scaled);
        const x8_next = @mulAdd(F32x8, loadF32x8(x_positions.ptr + index + 8), frequency_v, x_offset_scaled);
        const y8_next = @mulAdd(F32x8, loadF32x8(y_positions.ptr + index + 8), frequency_v, y_offset_scaled);
        const z8_next = @mulAdd(F32x8, loadF32x8(z_positions.ptr + index + 8), frequency_v, z_offset_scaled);
        store8(noise_out, index, simplexNoise3d8Nonnegative(seed, x8, y8, z8));
        store8(noise_out, index + 8, simplexNoise3d8Nonnegative(seed, x8_next, y8_next, z8_next));
    }

    while (index + 8 <= count) : (index += 8) {
        const x8 = @mulAdd(F32x8, loadF32x8(x_positions.ptr + index), frequency_v, x_offset_scaled);
        const y8 = @mulAdd(F32x8, loadF32x8(y_positions.ptr + index), frequency_v, y_offset_scaled);
        const z8 = @mulAdd(F32x8, loadF32x8(z_positions.ptr + index), frequency_v, z_offset_scaled);
        store8(noise_out, index, simplexNoise3d8Nonnegative(seed, x8, y8, z8));
    }

    while (index < count) : (index += 1) {
        const x = @mulAdd(f32, x_positions[index], frequency, x_offset * frequency);
        const y = @mulAdd(f32, y_positions[index], frequency, y_offset * frequency);
        const z = @mulAdd(f32, z_positions[index], frequency, z_offset * frequency);
        debugAssertNonnegativeSimplexDomainScalar(x, y, z);
        noise_out[index] = simplexNoise3d(
            seed,
            x,
            y,
            z,
            1.0,
        );
    }
}

pub fn simplexNoisePositionArray3dNonnegativeFrequency(
    comptime frequency: f32,
    noalias noise_out: []f32,
    count: usize,
    noalias x_positions: []const f32,
    noalias y_positions: []const f32,
    noalias z_positions: []const f32,
    x_offset: f32,
    y_offset: f32,
    z_offset: f32,
    seed: u32,
) void {
    std.debug.assert(noise_out.len >= count);
    std.debug.assert(x_positions.len >= count);
    std.debug.assert(y_positions.len >= count);
    std.debug.assert(z_positions.len >= count);

    const frequency_v = @as(F32x8, @splat(frequency));
    const x_offset_scaled = @as(F32x8, @splat(x_offset * frequency));
    const y_offset_scaled = @as(F32x8, @splat(y_offset * frequency));
    const z_offset_scaled = @as(F32x8, @splat(z_offset * frequency));

    var index: usize = 0;
    while (index + 16 <= count) : (index += 16) {
        const x8 = @mulAdd(F32x8, loadF32x8(x_positions.ptr + index), frequency_v, x_offset_scaled);
        const y8 = @mulAdd(F32x8, loadF32x8(y_positions.ptr + index), frequency_v, y_offset_scaled);
        const z8 = @mulAdd(F32x8, loadF32x8(z_positions.ptr + index), frequency_v, z_offset_scaled);
        const x8_next = @mulAdd(F32x8, loadF32x8(x_positions.ptr + index + 8), frequency_v, x_offset_scaled);
        const y8_next = @mulAdd(F32x8, loadF32x8(y_positions.ptr + index + 8), frequency_v, y_offset_scaled);
        const z8_next = @mulAdd(F32x8, loadF32x8(z_positions.ptr + index + 8), frequency_v, z_offset_scaled);
        store8(noise_out, index, simplexNoise3d8Nonnegative(seed, x8, y8, z8));
        store8(noise_out, index + 8, simplexNoise3d8Nonnegative(seed, x8_next, y8_next, z8_next));
    }

    while (index + 8 <= count) : (index += 8) {
        const x8 = @mulAdd(F32x8, loadF32x8(x_positions.ptr + index), frequency_v, x_offset_scaled);
        const y8 = @mulAdd(F32x8, loadF32x8(y_positions.ptr + index), frequency_v, y_offset_scaled);
        const z8 = @mulAdd(F32x8, loadF32x8(z_positions.ptr + index), frequency_v, z_offset_scaled);
        store8(noise_out, index, simplexNoise3d8Nonnegative(seed, x8, y8, z8));
    }

    while (index < count) : (index += 1) {
        const x = @mulAdd(f32, x_positions[index], frequency, x_offset * frequency);
        const y = @mulAdd(f32, y_positions[index], frequency, y_offset * frequency);
        const z = @mulAdd(f32, z_positions[index], frequency, z_offset * frequency);
        debugAssertNonnegativeSimplexDomainScalar(x, y, z);
        noise_out[index] = simplexNoise3d(
            seed,
            x,
            y,
            z,
            1.0,
        );
    }
}

fn expectSimplexChunkSamplesMatchScalar(
    comptime side_count: usize,
    values: []const f32,
    origin_x: f32,
    origin_y: f32,
    origin_z: f32,
    block_size: f32,
    seed: u32,
    frequency: f32,
) !void {
    const total = side_count * side_count * side_count;
    std.debug.assert(values.len >= total);

    const tolerance: f32 = 0.006;
    const stride = @max(@as(usize, 1), total / 127);
    var index: usize = 0;
    while (index < total) : (index += stride) {
        const x_index = index % side_count;
        const yz = index / side_count;
        const y_index = yz % side_count;
        const z_index = yz / side_count;
        const x = @mulAdd(f32, @as(f32, @floatFromInt(x_index)), block_size, origin_x);
        const y = @mulAdd(f32, @as(f32, @floatFromInt(y_index)), block_size, origin_y);
        const z = @mulAdd(f32, @as(f32, @floatFromInt(z_index)), block_size, origin_z);
        try std.testing.expectApproxEqAbs(simplexNoise3d(seed, x, y, z, frequency), values[index], tolerance);
    }

    const last = total - 1;
    const x_index = last % side_count;
    const yz = last / side_count;
    const y_index = yz % side_count;
    const z_index = yz / side_count;
    const x = @mulAdd(f32, @as(f32, @floatFromInt(x_index)), block_size, origin_x);
    const y = @mulAdd(f32, @as(f32, @floatFromInt(y_index)), block_size, origin_y);
    const z = @mulAdd(f32, @as(f32, @floatFromInt(z_index)), block_size, origin_z);
    try std.testing.expectApproxEqAbs(simplexNoise3d(seed, x, y, z, frequency), values[last], tolerance);
}

test "simplex 2d SIMD8 batch APIs match scalar within tolerance" {
    const seed: u32 = 1337;
    const frequency: f32 = 0.05;
    const tolerance: f32 = 0.006;

    var grid: [45]f32 = undefined;
    simplexNoiseUniformGrid2d(grid[0..], 0.125, -3.5, 9, 5, 1.0, 2.0, seed, frequency);
    for (grid, 0..) |value, index| {
        const x_index = index % 9;
        const y_index = index / 9;
        const x = @mulAdd(f32, @as(f32, @floatFromInt(x_index)), 1.0 * frequency, 0.125 * frequency);
        const y = @mulAdd(f32, @as(f32, @floatFromInt(y_index)), 2.0 * frequency, -3.5 * frequency);
        try std.testing.expectApproxEqAbs(simplexNoise2d(seed, x, y), value, tolerance);
    }

    const x_positions = [_]f32{ 0.0, 1.25, 2.5, 3.75, -4.5, 9.0, 11.125, -13.5, 21.25, 34.0, -55.5 };
    const y_positions = [_]f32{ 7.0, -2.25, 3.5, 8.75, 0.5, -11.0, 12.25, 6.5, -9.75, 14.0, 18.5 };
    var values: [x_positions.len]f32 = undefined;
    simplexNoisePositionArray2d(values[0..], values.len, x_positions[0..], y_positions[0..], 0.125, -3.5, seed, frequency);
    for (values, 0..) |value, index| {
        const x = @mulAdd(f32, x_positions[index], frequency, 0.125 * frequency);
        const y = @mulAdd(f32, y_positions[index], frequency, -3.5 * frequency);
        try std.testing.expectApproxEqAbs(simplexNoise2d(seed, x, y), value, tolerance);
    }
}

test "simplex 3d SIMD8 batch APIs match scalar within tolerance" {
    const seed: u32 = 1337;
    const frequency: f32 = 0.05;
    const tolerance: f32 = 0.006;

    var grid: [90]f32 = undefined;
    simplexNoiseUniformGrid3d(grid[0..], 0.125, -3.5, 7.25, 9, 5, 2, 1.0, 2.0, 0.5, seed, frequency);
    for (grid, 0..) |value, index| {
        const x_index = index % 9;
        const yz = index / 9;
        const y_index = yz % 5;
        const z_index = yz / 5;
        const x = @mulAdd(f32, @as(f32, @floatFromInt(x_index)), 1.0, 0.125);
        const y = @mulAdd(f32, @as(f32, @floatFromInt(y_index)), 2.0, -3.5);
        const z = @mulAdd(f32, @as(f32, @floatFromInt(z_index)), 0.5, 7.25);
        try std.testing.expectApproxEqAbs(simplexNoise3d(seed, x, y, z, frequency), value, tolerance);
    }

    var chunk16: [simplex_noise_chunk_16_3d_count]f32 = undefined;
    var chunk16_generic: [simplex_noise_chunk_16_3d_count]f32 = undefined;
    simplexNoiseChunk16_3d(chunk16[0..], 0.125, -3.5, 7.25, 1.0, seed, frequency);
    simplexNoiseChunk3d(16, chunk16_generic[0..], 0.125, -3.5, 7.25, 1.0, seed, frequency);
    for (chunk16, chunk16_generic) |chunk_value, generic_value| {
        try std.testing.expectEqual(chunk_value, generic_value);
    }
    try expectSimplexChunkSamplesMatchScalar(16, chunk16[0..], 0.125, -3.5, 7.25, 1.0, seed, frequency);

    var chunk32: [simplex_noise_chunk_32_3d_count]f32 = undefined;
    simplexNoiseChunk32_3d(chunk32[0..], 0.125, -3.5, 7.25, 1.0, seed, frequency);
    try expectSimplexChunkSamplesMatchScalar(32, chunk32[0..], 0.125, -3.5, 7.25, 1.0, seed, frequency);

    var chunk64: [simplex_noise_chunk_64_3d_count]f32 = undefined;
    simplexNoiseChunk64_3d(chunk64[0..], 0.125, -3.5, 7.25, 1.0, seed, frequency);
    try expectSimplexChunkSamplesMatchScalar(64, chunk64[0..], 0.125, -3.5, 7.25, 1.0, seed, frequency);

    const x_positions = [_]f32{ 0.0, 1.25, 2.5, 3.75, -4.5, 9.0, 11.125, -13.5, 21.25, 34.0, -55.5 };
    const y_positions = [_]f32{ 7.0, -2.25, 3.5, 8.75, 0.5, -11.0, 12.25, 6.5, -9.75, 14.0, 18.5 };
    const z_positions = [_]f32{ -1.0, 0.25, 4.5, -7.75, 10.5, 2.0, 3.25, -4.0, 8.125, -15.0, 22.5 };
    var values: [x_positions.len]f32 = undefined;
    simplexNoisePositionArray3d(values[0..], values.len, x_positions[0..], y_positions[0..], z_positions[0..], 0.125, -3.5, 7.25, seed, frequency);
    for (values, 0..) |value, index| {
        try std.testing.expectApproxEqAbs(
            simplexNoise3d(
                seed,
                x_positions[index] + 0.125,
                y_positions[index] - 3.5,
                z_positions[index] + 7.25,
                frequency,
            ),
            value,
            tolerance,
        );
    }

    var x_prescaled: [x_positions.len]f32 = undefined;
    var y_prescaled: [y_positions.len]f32 = undefined;
    var z_prescaled: [z_positions.len]f32 = undefined;
    for (x_positions, 0..) |x_position, index| {
        x_prescaled[index] = @mulAdd(f32, x_position, frequency, 0.125 * frequency);
        y_prescaled[index] = @mulAdd(f32, y_positions[index], frequency, -3.5 * frequency);
        z_prescaled[index] = @mulAdd(f32, z_positions[index], frequency, 7.25 * frequency);
    }

    var prescaled_values: [x_positions.len]f32 = undefined;
    simplexNoisePositionArray3dPrescaled(prescaled_values[0..], prescaled_values.len, x_prescaled[0..], y_prescaled[0..], z_prescaled[0..], seed);
    for (prescaled_values, values) |prescaled_value, value| {
        try std.testing.expectApproxEqAbs(value, prescaled_value, tolerance);
    }

    var pre_scaled_values: [x_positions.len]f32 = undefined;
    simplexNoisePositionArray3dPrescaled(pre_scaled_values[0..], pre_scaled_values.len, x_prescaled[0..], y_prescaled[0..], z_prescaled[0..], seed);
    for (pre_scaled_values, values) |pre_scaled_value, value| {
        try std.testing.expectApproxEqAbs(value, pre_scaled_value, tolerance);
    }

    const positive_x = [_]f32{ 0.0, 1.25, 2.5, 3.75, 4.5, 9.0, 11.125, 13.5, 21.25, 34.0, 55.5 };
    const positive_y = [_]f32{ 7.0, 2.25, 3.5, 8.75, 0.5, 11.0, 12.25, 6.5, 9.75, 14.0, 18.5 };
    const positive_z = [_]f32{ 1.0, 0.25, 4.5, 7.75, 10.5, 2.0, 3.25, 4.0, 8.125, 15.0, 22.5 };
    var positive_reference: [positive_x.len]f32 = undefined;
    var positive_fast: [positive_x.len]f32 = undefined;
    var positive_known_frequency: [positive_x.len]f32 = undefined;
    simplexNoisePositionArray3d(positive_reference[0..], positive_reference.len, positive_x[0..], positive_y[0..], positive_z[0..], 0.0, 0.0, 0.0, seed, frequency);
    simplexNoisePositionArray3dNonnegative(positive_fast[0..], positive_fast.len, positive_x[0..], positive_y[0..], positive_z[0..], 0.0, 0.0, 0.0, seed, frequency);
    simplexNoisePositionArray3dNonnegativeFrequency(frequency, positive_known_frequency[0..], positive_known_frequency.len, positive_x[0..], positive_y[0..], positive_z[0..], 0.0, 0.0, 0.0, seed);
    for (positive_fast, positive_reference) |fast_value, reference_value| {
        try std.testing.expectApproxEqAbs(reference_value, fast_value, tolerance);
    }
    for (positive_known_frequency, positive_fast) |known_frequency_value, runtime_frequency_value| {
        try std.testing.expectApproxEqAbs(runtime_frequency_value, known_frequency_value, tolerance);
    }

    var positive_x_prescaled: [positive_x.len]f32 = undefined;
    var positive_y_prescaled: [positive_y.len]f32 = undefined;
    var positive_z_prescaled: [positive_z.len]f32 = undefined;
    for (positive_x, 0..) |x_position, index| {
        positive_x_prescaled[index] = x_position * frequency;
        positive_y_prescaled[index] = positive_y[index] * frequency;
        positive_z_prescaled[index] = positive_z[index] * frequency;
    }

    var positive_pre_scaled: [positive_x.len]f32 = undefined;
    simplexNoisePositionArray3dPreScaledPositive(positive_pre_scaled[0..], positive_pre_scaled.len, positive_x_prescaled[0..], positive_y_prescaled[0..], positive_z_prescaled[0..], seed);
    for (positive_pre_scaled, positive_reference) |fast_value, reference_value| {
        try std.testing.expectApproxEqAbs(reference_value, fast_value, tolerance);
    }

    var positive_multiple16_reference: [16]f32 = undefined;
    var positive_multiple16_fast: [16]f32 = undefined;
    var positive_multiple16_x: [16]f32 = undefined;
    var positive_multiple16_y: [16]f32 = undefined;
    var positive_multiple16_z: [16]f32 = undefined;
    for (0..16) |index| {
        positive_multiple16_x[index] = @as(f32, @floatFromInt(index)) * frequency;
        positive_multiple16_y[index] = @as(f32, @floatFromInt(index / 4 + 2)) * frequency;
        positive_multiple16_z[index] = @as(f32, @floatFromInt(index / 8 + 3)) * frequency;
    }
    simplexNoisePositionArray3dPrescaled(positive_multiple16_reference[0..], positive_multiple16_reference.len, positive_multiple16_x[0..], positive_multiple16_y[0..], positive_multiple16_z[0..], seed);
    simplexNoisePositionArray3dPreScaledPositiveMultipleOf16(positive_multiple16_fast[0..], positive_multiple16_fast.len, positive_multiple16_x[0..], positive_multiple16_y[0..], positive_multiple16_z[0..], seed);
    for (positive_multiple16_fast, positive_multiple16_reference) |fast_value, reference_value| {
        try std.testing.expectApproxEqAbs(reference_value, fast_value, tolerance);
    }

    var positive_multiple32_reference: [32]f32 = undefined;
    var positive_multiple32_fast: [32]f32 = undefined;
    var positive_multiple32_x: [32]f32 = undefined;
    var positive_multiple32_y: [32]f32 = undefined;
    var positive_multiple32_z: [32]f32 = undefined;
    for (0..32) |index| {
        positive_multiple32_x[index] = @as(f32, @floatFromInt(index % 8)) * frequency;
        positive_multiple32_y[index] = @as(f32, @floatFromInt(index / 4 + 1)) * frequency;
        positive_multiple32_z[index] = @as(f32, @floatFromInt(index / 16 + 2)) * frequency;
    }
    simplexNoisePositionArray3dPrescaled(positive_multiple32_reference[0..], positive_multiple32_reference.len, positive_multiple32_x[0..], positive_multiple32_y[0..], positive_multiple32_z[0..], seed);
    simplexNoisePositionArray3dPreScaledPositiveMultipleOf16(positive_multiple32_fast[0..], positive_multiple32_fast.len, positive_multiple32_x[0..], positive_multiple32_y[0..], positive_multiple32_z[0..], seed);
    for (positive_multiple32_fast, positive_multiple32_reference) |fast_value, reference_value| {
        try std.testing.expectApproxEqAbs(reference_value, fast_value, tolerance);
    }
}

test "shifted-square simplex 2d batch APIs match shifted scalar within tolerance" {
    const seed: u32 = 1337;
    const frequency: f32 = 0.05;
    const tolerance: f32 = 0.006;

    var grid: [45]f32 = undefined;
    simplexNoiseUniformGrid2dShiftedSquare(grid[0..], 0.125, -3.5, 9, 5, 1.0, 2.0, seed, frequency);
    for (grid, 0..) |value, index| {
        const x_index = index % 9;
        const y_index = index / 9;
        const x = @mulAdd(f32, @as(f32, @floatFromInt(x_index)), 1.0 * frequency, 0.125 * frequency);
        const y = @mulAdd(f32, @as(f32, @floatFromInt(y_index)), 2.0 * frequency, -3.5 * frequency);
        try std.testing.expectApproxEqAbs(simplexNoise2dShiftedSquare(seed, x, y), value, tolerance);
    }

    const x_positions = [_]f32{ 0.0, 1.25, 2.5, 3.75, -4.5, 9.0, 11.125, -13.5, 21.25, 34.0, -55.5 };
    const y_positions = [_]f32{ 7.0, -2.25, 3.5, 8.75, 0.5, -11.0, 12.25, 6.5, -9.75, 14.0, 18.5 };
    var values: [x_positions.len]f32 = undefined;
    simplexNoisePositionArray2dShiftedSquare(values[0..], values.len, x_positions[0..], y_positions[0..], 0.125, -3.5, seed, frequency);
    for (values, 0..) |value, index| {
        const x = @mulAdd(f32, x_positions[index], frequency, 0.125 * frequency);
        const y = @mulAdd(f32, y_positions[index], frequency, -3.5 * frequency);
        try std.testing.expectApproxEqAbs(simplexNoise2dShiftedSquare(seed, x, y), value, tolerance);
    }
}

test "shifted-square simplex 3d batch APIs match shifted scalar within tolerance" {
    const seed: u32 = 1337;
    const frequency: f32 = 0.05;
    const tolerance: f32 = 0.006;

    var grid: [90]f32 = undefined;
    simplexNoiseUniformGrid3dShiftedSquare(grid[0..], 0.125, -3.5, 7.25, 9, 5, 2, 1.0, 2.0, 0.5, seed, frequency);
    for (grid, 0..) |value, index| {
        const x_index = index % 9;
        const yz = index / 9;
        const y_index = yz % 5;
        const z_index = yz / 5;
        const x = @mulAdd(f32, @as(f32, @floatFromInt(x_index)), 1.0, 0.125);
        const y = @mulAdd(f32, @as(f32, @floatFromInt(y_index)), 2.0, -3.5);
        const z = @mulAdd(f32, @as(f32, @floatFromInt(z_index)), 0.5, 7.25);
        try std.testing.expectApproxEqAbs(simplexNoise3dShiftedSquare(seed, x, y, z, frequency), value, tolerance);
    }

    const x_positions = [_]f32{ 0.0, 1.25, 2.5, 3.75, -4.5, 9.0, 11.125, -13.5, 21.25, 34.0, -55.5 };
    const y_positions = [_]f32{ 7.0, -2.25, 3.5, 8.75, 0.5, -11.0, 12.25, 6.5, -9.75, 14.0, 18.5 };
    const z_positions = [_]f32{ -1.0, 0.25, 4.5, -7.75, 10.5, 2.0, 3.25, -4.0, 8.125, -15.0, 22.5 };
    var values: [x_positions.len]f32 = undefined;
    simplexNoisePositionArray3dShiftedSquare(values[0..], values.len, x_positions[0..], y_positions[0..], z_positions[0..], 0.125, -3.5, 7.25, seed, frequency);
    for (values, 0..) |value, index| {
        try std.testing.expectApproxEqAbs(
            simplexNoise3dShiftedSquare(
                seed,
                x_positions[index] + 0.125,
                y_positions[index] - 3.5,
                z_positions[index] + 7.25,
                frequency,
            ),
            value,
            tolerance,
        );
    }
}
