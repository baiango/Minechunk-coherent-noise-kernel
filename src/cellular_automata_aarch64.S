#if defined(__APPLE__)
#define SYMBOL(name) _##name
#else
#define SYMBOL(name) name
#endif

.text

.arch_extension sme

.macro row_sum3_sme ptr, dst
    sub x15, \ptr, #1
    ld1b { z0.b }, p0/z, [x15]
    ld1b { z1.b }, p0/z, [\ptr]
    add x15, \ptr, #1
    ld1b { z2.b }, p0/z, [x15]
    add \dst, z0.b, z1.b
    add \dst, \dst, z2.b
.endm

.macro band_sum3_sme lower, middle, upper, dst
    row_sum3_sme \lower, z16.b
    row_sum3_sme \middle, z17.b
    row_sum3_sme \upper, z24.b
    add \dst, z16.b, z17.b
    add \dst, \dst, z24.b
.endm

.macro band_core_only_sme lower, middle, upper, core_dst
    row_sum3_sme \lower, z16.b
    row_sum3_sme \upper, z17.b
    add z16.b, z16.b, z17.b
    sub x15, \middle, #1
    ld1b { z0.b }, p0/z, [x15]
    add x15, \middle, #1
    ld1b { z2.b }, p0/z, [x15]
    add z17.b, z0.b, z2.b
    add \core_dst, z16.b, z17.b
.endm

.macro band_core_center_sme lower, middle, upper, core_dst
    row_sum3_sme \lower, z16.b
    row_sum3_sme \upper, z17.b
    add z16.b, z16.b, z17.b
    sub x15, \middle, #1
    ld1b { z0.b }, p0/z, [x15]
    add x15, \middle, #1
    ld1b { z2.b }, p0/z, [x15]
    add z17.b, z0.b, z2.b
    ld1b { z25.b }, p0/z, [\middle]
    add \core_dst, z16.b, z17.b
.endm

.macro emit_store_sme
    cmphi p1.b, p0/z, z23.b, z31.b
    mov z23.b, #0
    mov z23.b, p1/m, #1
    st1b { z23.b }, p0, [x0]
.endm

.macro emit_store_sme_from sum
    cmphi p1.b, p0/z, \sum, z31.b
    mov z30.b, #0
    mov z30.b, p1/m, #1
    st1b { z30.b }, p0, [x0]
.endm

.macro column_body_sme
    cmp x6, #1
    b.eq 3f

    sub x8, x1, x4
    sub x9, x2, x4
    sub x10, x3, x4
    band_sum3_sme x8, x9, x10, z18.b

    band_core_center_sme x1, x2, x3, z20.b
    add z19.b, z20.b, z25.b
    add z20.b, z18.b, z20.b

    add x8, x1, x4
    add x9, x2, x4
    add x10, x3, x4
    band_core_center_sme x8, x9, x10, z22.b

    add z23.b, z20.b, z22.b
    add z23.b, z23.b, z25.b
    add x14, x0, x5
    emit_store_sme
    subs x6, x6, #1
    b.eq 5f

    add z20.b, z19.b, z22.b
    add z19.b, z22.b, z25.b
    mov x0, x14
    add x1, x1, x4, lsl #1
    add x2, x2, x4, lsl #1
    add x3, x3, x4, lsl #1

    cmp x6, #1
    b.eq 2f
1:
    band_core_center_sme x1, x2, x3, z22.b
    add z23.b, z20.b, z22.b
    add z23.b, z23.b, z25.b
    emit_store_sme
    add z20.b, z19.b, z22.b
    add z19.b, z22.b, z25.b
    add x0, x0, x5
    add x1, x1, x4
    add x2, x2, x4
    add x3, x3, x4
    subs x6, x6, #1
    cmp x6, #1
    b.ne 1b
2:
    band_sum3_sme x1, x2, x3, z21.b
    add z23.b, z20.b, z21.b
    emit_store_sme
    b 5f

3:
    sub x8, x1, x4
    sub x9, x2, x4
    sub x10, x3, x4
    band_sum3_sme x8, x9, x10, z18.b

    band_core_only_sme x1, x2, x3, z20.b
    add z20.b, z18.b, z20.b

    add x8, x1, x4
    add x9, x2, x4
    add x10, x3, x4
    band_sum3_sme x8, x9, x10, z21.b

    add z23.b, z20.b, z21.b
    emit_store_sme
5:
.endm

.macro column_body_ge3_nextfull_sme
    sub x8, x1, x4
    sub x9, x2, x4
    sub x10, x3, x4
    band_sum3_sme x8, x9, x10, z18.b

    band_core_center_sme x1, x2, x3, z20.b
    add z19.b, z20.b, z25.b
    add z20.b, z18.b, z20.b

    add x8, x1, x4
    add x9, x2, x4
    add x10, x3, x4
    band_core_center_sme x8, x9, x10, z22.b

    add z21.b, z22.b, z25.b
    add z23.b, z20.b, z21.b
    emit_store_sme_from z23.b
    add x0, x0, x5

    add z20.b, z19.b, z22.b
    mov z19.d, z21.d
    add x1, x1, x4, lsl #1
    add x2, x2, x4, lsl #1
    add x3, x3, x4, lsl #1
    sub x6, x6, #2

.p2align 5,,15
1:
    band_core_center_sme x1, x2, x3, z22.b
    add z21.b, z22.b, z25.b
    add z23.b, z20.b, z21.b
    emit_store_sme_from z23.b
    add x0, x0, x5

    add z20.b, z19.b, z22.b
    mov z19.d, z21.d
    add x1, x1, x4
    add x2, x2, x4
    add x3, x3, x4
    subs x6, x6, #1
    b.ne 1b

    band_sum3_sme x1, x2, x3, z21.b
    add z23.b, z20.b, z21.b
    emit_store_sme_from z23.b
.endm

.arch_extension sme2

.macro row_sum3_sme2 ptr, dst
    sub x15, \ptr, #1
    ld1b { z0.b }, p0/z, [x15]
    ld1b { z1.b }, p0/z, [\ptr]
    add x15, \ptr, #1
    ld1b { z2.b }, p0/z, [x15]
    add \dst, z0.b, z1.b
    add \dst, \dst, z2.b
.endm

.macro band_sum3_sme2 lower, middle, upper, dst
    row_sum3_sme2 \lower, z16.b
    row_sum3_sme2 \middle, z17.b
    row_sum3_sme2 \upper, z24.b
    add \dst, z16.b, z17.b
    add \dst, \dst, z24.b
.endm

.macro band_core_only_sme2 lower, middle, upper, core_dst
    row_sum3_sme2 \lower, z16.b
    row_sum3_sme2 \upper, z17.b
    add z16.b, z16.b, z17.b
    sub x15, \middle, #1
    ld1b { z0.b }, p0/z, [x15]
    add x15, \middle, #1
    ld1b { z2.b }, p0/z, [x15]
    add z17.b, z0.b, z2.b
    add \core_dst, z16.b, z17.b
.endm

.macro band_core_center_sme2 lower, middle, upper, core_dst
    row_sum3_sme2 \lower, z16.b
    row_sum3_sme2 \upper, z17.b
    add z16.b, z16.b, z17.b
    sub x15, \middle, #1
    ld1b { z0.b }, p0/z, [x15]
    add x15, \middle, #1
    ld1b { z2.b }, p0/z, [x15]
    add z17.b, z0.b, z2.b
    add \core_dst, z16.b, z17.b
    ld1b { z25.b }, p0/z, [\middle]
.endm

.macro column_body_ge3_nextfull_sme2
    sub x8, x1, x4
    sub x9, x2, x4
    sub x10, x3, x4
    band_sum3_sme2 x8, x9, x10, z18.b

    band_core_center_sme2 x1, x2, x3, z20.b
    add z19.b, z20.b, z25.b
    add z20.b, z18.b, z20.b

    add x8, x1, x4
    add x9, x2, x4
    add x10, x3, x4
    band_core_center_sme2 x8, x9, x10, z22.b

    add z21.b, z22.b, z25.b
    add z23.b, z20.b, z21.b
    emit_store_sme_from z23.b
    add x0, x0, x5

    add z20.b, z19.b, z22.b
    mov z19.d, z21.d
    add x1, x1, x4, lsl #1
    add x2, x2, x4, lsl #1
    add x3, x3, x4, lsl #1
    sub x6, x6, #2

.p2align 5,,15
1:
    band_core_center_sme2 x1, x2, x3, z22.b
    add z21.b, z22.b, z25.b
    add z23.b, z20.b, z21.b
    emit_store_sme_from z23.b
    add x0, x0, x5

    add z20.b, z19.b, z22.b
    mov z19.d, z21.d
    add x1, x1, x4
    add x2, x2, x4
    add x3, x3, x4
    subs x6, x6, #1
    b.ne 1b

    band_sum3_sme2 x1, x2, x3, z21.b
    add z23.b, z20.b, z21.b
    emit_store_sme_from z23.b
.endm

.macro trusted_sme_kernel name
.p2align 5
.globl SYMBOL(\name)
SYMBOL(\name):
    stp x29, x30, [sp, #-96]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    stp x27, x28, [sp, #80]

    mov x24, x0
    mov x25, x1
    mov x19, x2
    mov x20, x3
    mov w21, w4

    mul x22, x19, x19
    mul x23, x20, x20

    smstart sm
    cntb x28
    cbz w21, .Ltrusted_fill_ones\@

    cmp w21, #14
    b.eq .Ltrusted_threshold14\@
    sub w21, w21, #1
    dup z31.b, w21
    b .Ltrusted_begin\@
.Ltrusted_threshold14\@:
    mov z31.b, #13

.Ltrusted_begin\@:
    mov x26, x20
    add x25, x1, x19
    add x25, x25, #1
    add x27, x25, x22
    add x21, x27, x22

.p2align 5,,15
.Ltrusted_z_loop\@:
    mov x11, x24
    mov x12, x25
    mov x13, x27
    mov x14, x21
    mov x17, x20
    ptrue p0.b

.Ltrusted_x_full_loop\@:
    mov x0, x11
    mov x1, x12
    mov x2, x13
    mov x3, x14
    mov x4, x19
    mov x5, x20
    mov x6, x20
    column_body_ge3_nextfull_sme

    add x11, x11, x28
    add x12, x12, x28
    add x13, x13, x28
    add x14, x14, x28
    subs x17, x17, x28
    b.ne .Ltrusted_x_full_loop\@

.Ltrusted_next_z\@:
    add x24, x24, x23
    add x25, x25, x22
    add x27, x27, x22
    add x21, x21, x22
    subs x26, x26, #1
    b.ne .Ltrusted_z_loop\@

    smstop sm
    ldp x27, x28, [sp, #80]
    ldp x25, x26, [sp, #64]
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #96
    ret

.Ltrusted_fill_ones\@:
    mov z23.b, #1
    mul x17, x20, x20
    mul x17, x17, x20
    mov x11, x24
    mov x12, #0
.Ltrusted_fill_loop\@:
    whilelo p0.b, x12, x17
    st1b { z23.b }, p0, [x11]
    add x11, x11, x28
    add x12, x12, x28
    cmp x12, x17
    b.lo .Ltrusted_fill_loop\@
    smstop sm
    ldp x27, x28, [sp, #80]
    ldp x25, x26, [sp, #64]
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #96
    ret
.endm

trusted_sme_kernel cellularAutomataSmooth3dBytesShrinkToBytesTrustedAarch64

.macro trusted_sme2_kernel name
.p2align 5
.globl SYMBOL(\name)
SYMBOL(\name):
    stp x29, x30, [sp, #-96]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    stp x27, x28, [sp, #80]

    mov x24, x0
    mov x25, x1
    mov x19, x2
    mov x20, x3
    mov w21, w4

    mul x22, x19, x19
    mul x23, x20, x20

    smstart sm
    cntb x28
    udiv x16, x20, x28
    cbz w21, .Ltrusted_fill_ones\@

    cmp w21, #14
    b.eq .Ltrusted_threshold14\@
    sub w21, w21, #1
    dup z31.b, w21
    b .Ltrusted_begin\@
.Ltrusted_threshold14\@:
    mov z31.b, #13

.Ltrusted_begin\@:
    mov x26, x20
    add x25, x1, x19
    add x25, x25, #1
    add x27, x25, x22
    add x21, x27, x22

.p2align 5,,15
.Ltrusted_z_loop\@:
    mov x11, x24
    mov x12, x25
    mov x13, x27
    mov x14, x21
    mov x17, x16
    ptrue p0.b

.Ltrusted_x_full_loop\@:
    mov x0, x11
    mov x1, x12
    mov x2, x13
    mov x3, x14
    mov x4, x19
    mov x5, x20
    mov x6, x20
    column_body_ge3_nextfull_sme2

    addvl x11, x11, #1
    addvl x12, x12, #1
    addvl x13, x13, #1
    addvl x14, x14, #1
    subs x17, x17, #1
    b.ne .Ltrusted_x_full_loop\@

.Ltrusted_next_z\@:
    add x24, x24, x23
    add x25, x25, x22
    add x27, x27, x22
    add x21, x21, x22
    subs x26, x26, #1
    b.ne .Ltrusted_z_loop\@

    smstop sm
    ldp x27, x28, [sp, #80]
    ldp x25, x26, [sp, #64]
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #96
    ret

.Ltrusted_fill_ones\@:
    mov z23.b, #1
    mul x17, x20, x20
    mul x17, x17, x20
    mov x11, x24
    mov x12, #0
.Ltrusted_fill_loop\@:
    whilelo p0.b, x12, x17
    st1b { z23.b }, p0, [x11]
    add x11, x11, x28
    add x12, x12, x28
    cmp x12, x17
    b.lo .Ltrusted_fill_loop\@
    smstop sm
    ldp x27, x28, [sp, #80]
    ldp x25, x26, [sp, #64]
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #96
    ret
.endm

trusted_sme2_kernel cellularAutomataSmooth3dBytesShrinkToBytesTrustedAarch64Sme2

.p2align 5
.globl SYMBOL(smoothInteriorByteColumnToBytesAarch64Sme)
SYMBOL(smoothInteriorByteColumnToBytesAarch64Sme):
    cbz x6, 5f

    smstart sm
    ptrue p0.b, vl16
    cbz w7, 7f

    cmp w7, #14
    b.eq 8f
    sub w7, w7, #1
    dup z31.b, w7
    b 9f
8:
    mov z31.b, #13
9:
    cmp x6, #1
    b.eq 3f

    sub x8, x1, x4
    sub x9, x2, x4
    sub x10, x3, x4
    band_sum3_sme x8, x9, x10, z18.b

    band_core_center_sme x1, x2, x3, z20.b
    add z19.b, z20.b, z25.b
    add z20.b, z18.b, z20.b

    add x8, x1, x4
    add x9, x2, x4
    add x10, x3, x4
    band_core_center_sme x8, x9, x10, z22.b

    add z23.b, z20.b, z22.b
    add z23.b, z23.b, z25.b
    add x14, x0, x5
    emit_store_sme
    subs x6, x6, #1
    b.eq 4f

    add z20.b, z19.b, z22.b
    add z19.b, z22.b, z25.b
    mov x0, x14
    add x1, x1, x4, lsl #1
    add x2, x2, x4, lsl #1
    add x3, x3, x4, lsl #1

    cmp x6, #1
    b.eq 2f
.p2align 5,,15
1:
    band_core_center_sme x1, x2, x3, z22.b
    add z23.b, z20.b, z22.b
    add z23.b, z23.b, z25.b
    emit_store_sme
    add z20.b, z19.b, z22.b
    add z19.b, z22.b, z25.b
    add x0, x0, x5
    add x1, x1, x4
    add x2, x2, x4
    add x3, x3, x4
    subs x6, x6, #1
    cmp x6, #1
    b.ne 1b
2:
    band_sum3_sme x1, x2, x3, z21.b
    add z23.b, z20.b, z21.b
    emit_store_sme
4:
    smstop sm
5:
    ret

3:
    sub x8, x1, x4
    sub x9, x2, x4
    sub x10, x3, x4
    band_sum3_sme x8, x9, x10, z18.b

    band_core_only_sme x1, x2, x3, z20.b
    add z20.b, z18.b, z20.b

    add x8, x1, x4
    add x9, x2, x4
    add x10, x3, x4
    band_sum3_sme x8, x9, x10, z21.b

    add z23.b, z20.b, z21.b
    emit_store_sme
    smstop sm
    ret

.p2align 4
7:
    mov z23.b, #1
1:
    st1b { z23.b }, p0, [x0]
    add x0, x0, x5
    subs x6, x6, #1
    b.ne 1b
    smstop sm
    ret
