// Physarum picture — the trail tonemapped to [0,1] luminance, optionally
// lit as a relief.
//
//   x   = trail / norm + grain * laid / gnorm
//   lum = 1 - exp(-exposure * x)
//
// norm is the trail's 95th percentile and gnorm = 4 * mean(laid) (both from
// stats.frag); `grain` mixes this frame's raw deposits over the smooth trail.
// Drawn with fullscreen.vert; the host colorizes lum through a palette LUT.
//
// Relief (u_depth > 0): the trail is read as a height field
// h = 1 - exp(-RELIEF_K * trail / norm), softer than the exposure curve so a
// thick trunk stays rounded instead of a flat clipped band. Its gradient
// (1 px and 3 px central differences) gives a normal lit by one key light,
// u_light; two samples toward the light cast a short shadow from taller
// veins onto whatever lies under them; the 3 px ring mean darkens cavities
// beside trunks; faint ground is pushed back. (A specular sheen on the
// ridges was cut on 2026-09-24: measured on grown fields of all five looks,
// six seeds, four orbit angles, it moved lum by at most 0.014 and at most
// 0.13% of pixels by more than 1/255, so it was per-texel arithmetic for
// nothing a viewer sees.) The result stays a luminance, so every palette, the desktop R8
// path and any downstream dither see an ordinary lum.
//
// u_light is a unit vector in GRID-TEXEL space (+x = column, +y = row index,
// +z = out of the picture). It is a uniform rather than a constant for two
// reasons: the hosts orbit it (a light that never moves reads as an emboss
// filter; a moving one is the strongest shape cue a flat picture has), and
// "upper left on screen" is a different grid vector per host, because each
// host decides which row lands at the top. The HOST normalises it and keeps
// u_light.z >= 0.05: the key term divides by L.z, and a guard here would be
// per-texel arithmetic for a condition the host rules out once per frame.
//
// u_depth = 0 (also the default for a host that never sets it) takes the
// original two-tap path and returns today's picture bit for bit
// (tests/test_physarum_gl.py holds this against a frozen copy of the flat
// shader). A host that sets u_depth > 0 must also set u_light: its default
// of (0, 0, 0) divides by zero.
// Cost with depth on: 12 texel fetches per grid texel (2 today).
// dtouch/physarum.py relief() (called from PhysarumField.luminance) is the
// numpy twin of the relief branch (the CPU fallback); change both together.

uniform sampler2D u_trail;
uniform sampler2D u_laid;
uniform float u_inv_norm;
uniform float u_grain;
uniform float u_inv_gnorm;
uniform float u_exposure;
uniform float u_depth;     // 0 = flat (today), 1 = full relief
uniform vec3 u_light;      // key light: unit, grid-texel space, z >= 0.05

layout(location = 0) out vec4 f_color;

const float RELIEF_K = 0.25;      // height compression (trail/norm -> h)
const float RELIEF_BUMP = 5.0;    // normal steepness
const float RELIEF_BROAD = 0.5;   // share of the 3 px gradient in the normal
const float RELIEF_AMB = 0.30;    // key-light floor on the far side of a vein
const float RELIEF_SHADOW = 2.5;  // cast shadow strength
const float RELIEF_CAVITY = 1.0;  // cavity darkening strength
const float RELIEF_GROUND = 0.8;  // faint ground brightness (1 = untouched)

float reliefH(ivec2 c, ivec2 hi) {
    vec3 t = texelFetch(u_trail, clamp(c, ivec2(0), hi), 0).rgb;
    return 1.0 - exp(-RELIEF_K * (t.r + t.g + t.b) * u_inv_norm);
}

void main() {
    ivec2 c = ivec2(gl_FragCoord.xy);
    vec3 tr = texelFetch(u_trail, c, 0).rgb;
    vec3 la = texelFetch(u_laid, c, 0).rgb;
    float x = (tr.r + tr.g + tr.b) * u_inv_norm
            + u_grain * (la.r + la.g + la.b) * u_inv_gnorm;
    float lum = 1.0 - exp(-u_exposure * x);
    if (u_depth <= 0.0) {
        f_color = vec4(vec3(lum), 1.0);
        return;
    }

    ivec2 hi = textureSize(u_trail, 0) - 1;
    float h0 = 1.0 - exp(-RELIEF_K * (tr.r + tr.g + tr.b) * u_inv_norm);
    float e1 = reliefH(c + ivec2(1, 0), hi);
    float w1 = reliefH(c - ivec2(1, 0), hi);
    float n1 = reliefH(c + ivec2(0, 1), hi);
    float s1 = reliefH(c - ivec2(0, 1), hi);
    float e3 = reliefH(c + ivec2(3, 0), hi);
    float w3 = reliefH(c - ivec2(3, 0), hi);
    float n3 = reliefH(c + ivec2(0, 3), hi);
    float s3 = reliefH(c - ivec2(0, 3), hi);
    vec2 g = mix(vec2(e1 - w1, n1 - s1) * 0.5, vec2(e3 - w3, n3 - s3) / 6.0, RELIEF_BROAD);
    vec3 n = normalize(vec3(-g * RELIEF_BUMP, 1.0));
    vec3 L = u_light;
    // key light, normalised so flat ground keeps its brightness
    float shade = RELIEF_AMB + (1.0 - RELIEF_AMB) * max(dot(n, L), 0.0) / L.z;
    // cast shadow: does a taller vein stand between here and the light?
    vec2 ld = normalize(L.xy);
    float o2 = reliefH(c + ivec2(round(ld * 2.0)), hi) - h0;
    float o5 = reliefH(c + ivec2(round(ld * 5.0)), hi) - h0;
    float shadow = 1.0 - min(0.75, RELIEF_SHADOW * max(0.0, max(o2, o5 * 0.7)));
    // cavity: darker where the surroundings stand above this texel
    float ring = 0.25 * (e3 + w3 + n3 + s3);
    float cavity = 1.0 - min(0.8, RELIEF_CAVITY * max(0.0, ring - h0));
    float ground = mix(RELIEF_GROUND, 1.0, smoothstep(0.05, 0.45, h0));
    float lit = clamp(lum * shade * shadow * cavity * ground, 0.0, 1.0);
    f_color = vec4(vec3(mix(lum, lit, u_depth)), 1.0);
}
