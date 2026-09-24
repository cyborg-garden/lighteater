// Stage 5, ordered variants: the tile-threshold-quantise core of
// dtouch.dither._ordered_indices. The threshold texture is the 4x4 Bayer
// matrix or the 64x64 blue-noise asset. The sRGB transfer functions are
// NOT computed here: the host uploads dtouch.dither's own float32 LUTs
// (linearize-x-levels by u8 code value, and the level-encode table), so
// every arithmetic step is the same IEEE f32 op on the same values as the
// numpy path and the pass is bit-exact against it.
uniform sampler2D u_src;      // float working image
uniform sampler2D u_mat;      // threshold matrix, R32F
uniform sampler2D u_lin_lut;  // 256x1 R32F: srgb_to_linear(k/255) * levels
uniform sampler2D u_enc_lut;  // 256x1 R32F: level index -> output value
uniform ivec2 u_mat_size;
uniform float u_levels;
uniform int u_invert;
uniform int u_gamma;

layout(location = 0) out vec4 f_color;

float work_of(float v) {
    if (u_gamma == 1) {
        // _to_u8 then the linearization LUT (x levels folded in)
        int k = int(floor(clamp(v, 0.0, 1.0) * 255.0 + 0.5));
        return texelFetch(u_lin_lut, ivec2(k, 0), 0).r;
    }
    return clamp(v, 0.0, 1.0) * u_levels;
}

float dither_of(float v, float t) {
    float work = work_of(v);
    if (u_invert == 1) work = u_levels - work;
    float idx = floor(work + t);         // the uint8 cast IS the floor
    if (u_invert == 1) idx = u_levels - idx;
    return texelFetch(u_enc_lut, ivec2(int(idx), 0), 0).r;
}

void main() {
    ivec2 xy = ivec2(gl_FragCoord.xy);
    vec3 v = texelFetch(u_src, xy, 0).rgb;
    // tile the matrix by integer arithmetic: xy and u_mat_size are never
    // negative, so this equals xy % u_mat_size, and ES 3.00 leaves % on
    // ints undefined for negative operands (see README)
    float t = texelFetch(u_mat, xy - u_mat_size * (xy / u_mat_size), 0).r;
    f_color = vec4(dither_of(v.r, t), dither_of(v.g, t), dither_of(v.b, t), 1.0);
}
