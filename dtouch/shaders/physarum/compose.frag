// Physarum fractal-veins compose — the fractal amount's screen-size drawing
// pass (no stock counterpart; runs only when the amount is above 0).
//
// Draws the three scales at SCREEN size (or the largest size the pixel
// budget allows) from the grid-size fields tonemap_fractal.frag wrote. The
// fields are sampled with hardware bilinear filtering and THEN thresholded,
// the way distance-field text is drawn: the outline is a smooth curve
// between texels, cut with a one-pixel anti-aliased step (fwidth), so a
// thread stays a hairline and a trunk keeps a hard flank however far the
// grid is stretched. Back to front: threads, then veins over them, then
// trunks over both; each level occludes the finer ones where it is solid.
// Depth is value (u_lvlVal) and relief (the field's alpha), never blur.
//
// Life, all slow: two attention zones (u_bloom, grid px) drift over the
// frame and bloom in turn; inside one the finer levels' outlines open
// (a lower threshold, u_bloomThr), on top of the physics there, so more and
// finer lace shows where the zone is. A faint travelling shimmer runs
// through the finer levels (u_shimmer: amplitude, wavelength in grid px,
// time), a few percent of their value, never a flicker.
//
// u_mix < 1 blends toward the stock picture itself (u_stock: tonemap.frag's
// output at grid size, relief included), so small amounts morph out of
// today's picture. (The browser demo rebuilt the stock side here from the
// trail as the FLAT sum; with the relief on, which every look has, the
// first nudge of the amount dropped the lighting and the picture jumped
// ~26% brighter. Desktop 2026-09-25; tests/test_physarum_fractal.py.)

uniform sampler2D u_fields;
uniform sampler2D u_line;    // tonemap_fractal.frag's second target: centreline offset (texels), direction
uniform vec2 u_hair;         // thread half-width in screen px: (faint, strong)
uniform sampler2D u_stock;   // the stock picture (tonemap.frag), grid size; read when u_mix < 1
uniform vec2 u_outSize;
uniform ivec2 u_grid;
uniform float u_mix;
uniform vec3 u_lvlVal;     // value (distance) per level, 1 = front
uniform vec3 u_thr;        // field value where each level's outline sits
uniform vec3 u_body;       // interior floor per level: how flat the inside is
uniform float u_glow;      // faint underglow of what lies outside an outline, 0 = none
uniform vec4 u_bloom[2];   // attention zones: centre, radius (grid px), strength 0..1
uniform float u_bloomThr;  // share the finer outlines' threshold drops at full bloom
uniform vec3 u_shimmer;    // amplitude, wavelength (grid px), time (s)

layout(location = 0) out vec4 f_color;

float aastep(float t, float f) {
    float w = max(0.7 * fwidth(f), 1e-4);
    return smoothstep(t - w, t + w, f);
}

// bilinear, edge texels held: what the host's own upscale of the stock
// picture does (cv2.INTER_LINEAR on the desktop)
float stockLin(vec2 gp) {
    vec2 q = gp - 0.5;
    vec2 fr = fract(q);
    ivec2 hi = u_grid - 1;
    ivec2 c0 = clamp(ivec2(floor(q)), ivec2(0), hi);
    ivec2 c1 = clamp(c0 + 1, ivec2(0), hi);
    float a = texelFetch(u_stock, c0, 0).r;
    float b = texelFetch(u_stock, ivec2(c1.x, c0.y), 0).r;
    float c = texelFetch(u_stock, ivec2(c0.x, c1.y), 0).r;
    float d = texelFetch(u_stock, c1, 0).r;
    return mix(mix(a, b, fr.x), mix(c, d, fr.x), fr.y);
}

void main() {
    vec2 uv = gl_FragCoord.xy / u_outSize;
    vec2 gp = uv * vec2(u_grid);
    vec4 F = texture(u_fields, uv);

    float bl = 0.0;
    vec2 g = vec2(u_grid);
    for (int i = 0; i < 2; i++) {
        vec2 d = abs(gp - u_bloom[i].xy);
        d = min(d, g - d);
        float r = max(u_bloom[i].z, 1.0);
        bl = max(bl, u_bloom[i].w * exp(-dot(d, d) / (r * r)));
    }
    vec3 thr = u_thr * vec3(1.0, 1.0 - 0.5 * u_bloomThr * bl, 1.0 - u_bloomThr * bl);

    float a0 = aastep(thr.x, F.r);
    float a1 = aastep(thr.y, F.g);
    // Threads: a line of fixed SCREEN width around the centreline. The
    // offset is interpolated by hand, each neighbour's direction flipped to
    // agree with the nearest one first (the sign of an eigenvector is
    // arbitrary; averaged unaligned, the offset would cross zero off the
    // crest and print ghost lines).
    vec2 q = gp - 0.5;
    vec2 fq = fract(q);
    ivec2 hi = u_grid - 1;
    ivec2 b0 = clamp(ivec2(floor(q)), ivec2(0), hi), b1 = clamp(b0 + 1, ivec2(0), hi);
    vec4 l00 = texelFetch(u_line, b0, 0), l10 = texelFetch(u_line, ivec2(b1.x, b0.y), 0);
    vec4 l01 = texelFetch(u_line, ivec2(b0.x, b1.y), 0), l11 = texelFetch(u_line, b1, 0);
    vec2 ref = (fq.x < 0.5) ? ((fq.y < 0.5) ? l00.yz : l01.yz) : ((fq.y < 0.5) ? l10.yz : l11.yz);
    float o00 = l00.x * sign(dot(l00.yz, ref) + 1e-6), o10 = l10.x * sign(dot(l10.yz, ref) + 1e-6);
    float o01 = l01.x * sign(dot(l01.yz, ref) + 1e-6), o11 = l11.x * sign(dot(l11.yz, ref) + 1e-6);
    float off = mix(mix(o00, o10, fq.x), mix(o01, o11, fq.x), fq.y);
    float pxPerTexel = u_outSize.x / float(u_grid.x);
    // a little wider where a texel covers several screen pixels (2x DPR), so
    // the hairline still reads, but by the square root, so it stays one
    float hw = mix(u_hair.x, u_hair.y, clamp(F.b, 0.0, 1.0)) * sqrt(max(pxPerTexel, 1.0));
    float a2 = clamp(hw + 0.5 - abs(off) * pxPerTexel, 0.0, 1.0) * aastep(thr.z, F.b);
    vec3 fv = vec3(F.r, F.g, F.b);
    vec3 v3 = u_lvlVal * (u_body + (1.0 - u_body) * fv);

    // the shimmer: two slow crossed waves, their product a drifting lattice
    // of soft swells a few percent deep, on the finer levels only
    float k = 6.2831853 / max(u_shimmer.y, 1.0);
    float sh = sin(dot(gp, vec2(0.83, 0.56)) * k - 0.61 * u_shimmer.z)
             * sin(dot(gp, vec2(-0.47, 0.88)) * k * 0.77 + 0.43 * u_shimmer.z);
    v3.yz *= 1.0 + u_shimmer.x * vec2(0.6, 1.0) * sh;

    float v = a2 * v3.z;
    v = mix(v, v3.y, a1);
    v = mix(v, v3.x, a0);
    // what lies outside every outline may keep a faint underglow
    v = max(v, u_glow * max(u_lvlVal.x * F.r, max(u_lvlVal.y * F.g, u_lvlVal.z * F.b)));
    float outL = clamp(v * F.a, 0.0, 1.0);

    if (u_mix < 1.0) {
        outL = mix(stockLin(gp), outL, u_mix);
    }
    f_color = vec4(vec3(outL), 1.0);
}
