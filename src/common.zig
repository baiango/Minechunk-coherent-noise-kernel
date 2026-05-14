comptime {
    @setFloatMode(.optimized);
}

pub const AIR: u32 = 0;
pub const BEDROCK: u32 = 1;
pub const STONE: u32 = 2;
pub const DIRT: u32 = 3;
pub const GRASS: u32 = 4;
pub const SAND: u32 = 5;
pub const SNOW: u32 = 6;

pub const F16x8 = @Vector(8, f16);
pub const U16x8 = @Vector(8, u16);
pub const F32x2 = @Vector(2, f32);
pub const F32x4 = @Vector(4, f32);
pub const I32x4 = @Vector(4, i32);
pub const U32x4 = @Vector(4, u32);
pub const Boolx4 = @Vector(4, bool);
pub const F32x8 = @Vector(8, f32);
pub const I32x8 = @Vector(8, i32);
pub const U32x8 = @Vector(8, u32);

pub fn fade(t: f32) f32 {
    const inner = @mulAdd(f32, t, 6.0, -15.0);
    const mid = @mulAdd(f32, t, inner, 10.0);
    return t * t * t * mid;
}

pub fn fade4(t: F32x4) F32x4 {
    const inner = @mulAdd(F32x4, t, @as(F32x4, @splat(@as(f32, 6.0))), @as(F32x4, @splat(@as(f32, -15.0))));
    const mid = @mulAdd(F32x4, t, inner, @as(F32x4, @splat(@as(f32, 10.0))));
    return t * t * t * mid;
}

pub fn lerp(a: f32, b: f32, t: f32) f32 {
    return @mulAdd(f32, b - a, t, a);
}

pub fn lerp4(a: F32x4, b: F32x4, t: F32x4) F32x4 {
    return @mulAdd(F32x4, b - a, t, a);
}

pub fn loadU32x4(pointer: [*]const u32) U32x4 {
    const vector_pointer: *align(4) const U32x4 = @ptrCast(pointer);
    return vector_pointer.*;
}

pub fn storeU32x4(pointer: [*]u32, value: U32x4) void {
    const vector_pointer: *align(4) U32x4 = @ptrCast(pointer);
    vector_pointer.* = value;
}

pub fn clamp01(value: f32) f32 {
    return @min(@max(value, 0.0), 1.0);
}

pub fn absF32(value: f32) f32 {
    return @abs(value);
}

pub fn maxI32(a: i32, b: i32) i32 {
    return if (a > b) a else b;
}

pub fn maxI64(a: i64, b: i64) i64 {
    return if (a > b) a else b;
}

pub fn minI64(a: i64, b: i64) i64 {
    return if (a < b) a else b;
}

pub fn maxUsize(a: usize, b: usize) usize {
    return if (a > b) a else b;
}

pub fn minUsize(a: usize, b: usize) usize {
    return if (a < b) a else b;
}
