// Physarum fractal-veins fields — the tonemap pass when the fractal amount is
// above 0 (the host runs tonemap.frag at amount 0 and this file otherwise).
//
// A separate file rather than a branch in tonemap.frag: it has a second
// render target (f_line), writes four FIELDS rather than one luminance, and
// reads a 5x5 neighbourhood; folded into tonemap.frag it would put an MRT
// output and ~150 never-run lines in front of the flat two-tap path, and the
// host has to switch render targets for it anyway, so a uniform gate would
// save nothing.
//
// The stock tonemap sums the three channels into one luminance. Here they
// are three scales of one organism (r trunks, g veins, b threads), and this
// pass no longer draws the picture: it runs at GRID size and writes one
// FIELD per scale, shaped so that a threshold on it is the level's outline.
// compose.frag then draws those outlines at SCREEN size, so every
// edge is cut after the upscale and stays one screen pixel wide at any DPR
// (a luminance tonemapped at grid size and then stretched 2x is soft by
// construction; that was most of the fuzz).
//   r: trunks, each level against its OWN bright end (u_inv_nl), toned,
//      less a share of its 6-texel ring (u_crest.z), so a flat plateau of
//      trail (haze, a crowd) falls under the outline and a trunk does not;
//   g: veins, the same against a 4-texel ring (u_crest.y): a crowd of
//      veins is a plateau too, and without this it flooded into grey slabs;
//   b: threads as LINES, not as light: a ridge detector (the Hessian's
//      strongest negative curvature, minus a share of the weaker one) on the
//      3x3-smoothed thread channel. A line of any width answers on its crest
//      only, and a round blob answers weakly, so the finest level cannot
//      render as haze or as glowing dots, only as hairlines;
//      A second target (f_line) carries where that crest IS: the signed
//      distance, in texels, from this texel to the thread's centreline
//      along the direction of strongest curvature (one Newton step on the
//      smoothed profile, -grad.e / lambda), and that direction e. compose
//      draws the thread as a line of fixed screen width around the zero of
//      that distance, so a thread is a true hairline at 1x and at 2x DPR,
//      whatever the grid-to-screen ratio;
//   a: the stock relief lighting (shade, shadow, cavity, ground) as one
//      multiplier, weighted per level by u_lvlRelief, blended by u_depth.
// Grain (the per-frame sparkle) stays out of the threads: on a hairline it
// reads as dust.

uniform sampler2D u_trail;
uniform sampler2D u_laid;
uniform float u_inv_norm;
uniform float u_grain;
uniform float u_inv_gnorm;
uniform float u_exposure;
uniform float u_depth;
uniform vec3 u_light;
uniform vec3 u_inv_nl;     // 1 / bright end per level
uniform vec3 u_lvlRelief;  // height-field weight per level
uniform vec3 u_lvlExp;     // exposure multiplier per level
uniform float u_grainK;    // share of the stock grain kept on the trunks
uniform vec3 u_crest;      // (thread ridge gain, share of its ring a vein loses, share a trunk loses)
uniform float u_lineBeta;  // how much a round blob is discounted against a line
uniform float u_lineGate;  // thread value (own bright end units) under which a ridge is noise

layout(location = 0) out vec4 f_color;
layout(location = 1) out vec4 f_line;   // (signed distance to the thread centreline in texels, e.x, e.y, 0)

const float RELIEF_K = 0.25;
const float RELIEF_BUMP = 5.0;
const float RELIEF_BROAD = 0.5;
const float RELIEF_AMB = 0.30;
const float RELIEF_SHADOW = 2.5;
const float RELIEF_CAVITY = 1.0;
const float RELIEF_GROUND = 0.8;

ivec2 HI;

vec3 T(ivec2 c) { return texelFetch(u_trail, clamp(c, ivec2(0), HI), 0).rgb * u_inv_nl; }

// the height field, on a 5-tap cross mean: lit at 5x bump, the texel
// stipple of a crowded trunk read as frost
float reliefH(ivec2 c) {
    vec3 t = 0.5 * T(c) + 0.125 * (T(c + ivec2(1, 0)) + T(c - ivec2(1, 0)) + T(c + ivec2(0, 1)) + T(c - ivec2(0, 1)));
    return 1.0 - exp(-RELIEF_K * dot(t, u_lvlRelief));
}

void main() {
    ivec2 c = ivec2(gl_FragCoord.xy);
    HI = textureSize(u_trail, 0) - 1;
    // one 5x5 neighbourhood serves every level
    vec3 nb[25];
    for (int j = 0; j < 5; j++)
        for (int i = 0; i < 5; i++)
            nb[j * 5 + i] = T(c + ivec2(i - 2, j - 2));
    vec3 x = nb[12];
    vec3 la = texelFetch(u_laid, c, 0).rgb;
    float xr = x.r + u_grainK * u_grain * la.r * u_inv_gnorm * (u_inv_nl.r / u_inv_norm);

    // Trunks and veins are cut on a 3x3 binomial-smoothed centre: the
    // sparse skirt of a fast point is stippled texel to texel, and a
    // threshold on the raw value frays the outline into speckle.
    vec3 xs = 0.25 * x + 0.125 * (nb[13] + nb[11] + nb[17] + nb[7])
            + 0.0625 * (nb[18] + nb[6] + nb[8] + nb[16]);
    // the trunks on the wider 5x5 binomial: a crowd's skirt is the
    // stippliest trail in the picture, and its outline frayed on 3x3
    float xw = 0.0;
    const float BW[5] = float[5](1.0, 4.0, 6.0, 4.0, 1.0);
    for (int j = 0; j < 5; j++)
        for (int i = 0; i < 5; i++)
            xw += BW[i] * BW[j] * nb[j * 5 + i].r;
    xs.r = xw * (1.0 / 256.0);
    // trunks: the level's tone, less a share (u_crest.z) of the 6-texel
    // ring. On a flat plateau of trail (haze, a crowd) that leaves only
    // (1 - share) of it, so the plateau falls under the outline; on a trunk
    // the ring is dark and the trunk keeps nearly all of itself.
    float ringT = 0.125 * (T(c + ivec2(6, 0)).r + T(c - ivec2(6, 0)).r + T(c + ivec2(0, 6)).r + T(c - ivec2(0, 6)).r
                         + T(c + ivec2(4, 4)).r + T(c - ivec2(4, 4)).r + T(c + ivec2(4, -4)).r + T(c - ivec2(4, -4)).r);
    xr += xs.r - x.r;
    // (divided by 1 - share/2: unchanged where the ring averages half the
    // centre; a lone trunk on dark ground, ring ~0, comes out 1/(1 - share/2)
    // brighter, 1.6x at the shipped 0.75, and a flat plateau
    // (1 - share)/(1 - share/2), 0.4x)
    xr = max(xr - u_crest.z * ringT, 0.0) / (1.0 - 0.5 * u_crest.z);
    float f0 = 1.0 - exp(-u_exposure * u_lvlExp.x * xr);

    // veins: the same, against the 4-texel ring (u_crest.y)
    float ring = 0.125 * (T(c + ivec2(4, 0)).g + T(c - ivec2(4, 0)).g + T(c + ivec2(0, 4)).g + T(c - ivec2(0, 4)).g
                        + T(c + ivec2(3, 3)).g + T(c - ivec2(3, 3)).g + T(c + ivec2(3, -3)).g + T(c - ivec2(3, -3)).g);
    float xg = max(xs.g - u_crest.y * ring, 0.0) / (1.0 - 0.5 * u_crest.y);
    float f1 = 1.0 - exp(-u_exposure * u_lvlExp.y * xg);

    // threads: ridge strength on the 3x3 binomial-smoothed channel
    float v[25];
    for (int k = 0; k < 25; k++) v[k] = nb[k].b;
    float sm[9];
    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < 3; i++) {
            int o = (j + 1) * 5 + (i + 1);
            sm[j * 3 + i] = (4.0 * v[o]
                + 2.0 * (v[o - 1] + v[o + 1] + v[o - 5] + v[o + 5])
                + v[o - 6] + v[o - 4] + v[o + 4] + v[o + 6]) * (1.0 / 16.0);
        }
    }
    float dxx = sm[5] - 2.0 * sm[4] + sm[3];
    float dyy = sm[7] - 2.0 * sm[4] + sm[1];
    float dxy = 0.25 * (sm[8] - sm[6] - sm[2] + sm[0]);
    float m = 0.5 * (dxx + dyy);
    float q = sqrt(0.25 * (dxx - dyy) * (dxx - dyy) + dxy * dxy);
    float lStrong = m - q, lWeak = m + q;
    float line = max(-lStrong, 0.0) - u_lineBeta * max(-lWeak, 0.0);
    // a ridge in near-empty ground is noise, not a thread
    line *= smoothstep(u_lineGate * 0.5, u_lineGate, sm[4]);
    float f2 = 1.0 - exp(-u_exposure * u_lvlExp.z * u_crest.x * max(line, 0.0));

    // the centreline: eigenvector of the strongest (most negative) curvature,
    // and the Newton offset to the crest along it; e points to +x by
    // convention (compose aligns neighbours before interpolating)
    vec2 v1 = vec2(lStrong - dyy, dxy), v2 = vec2(dxy, lStrong - dxx);
    vec2 e = dot(v1, v1) > dot(v2, v2) ? v1 : v2;
    e = dot(e, e) > 1e-12 ? normalize(e) : vec2(1.0, 0.0);
    if (e.x < 0.0 || (e.x == 0.0 && e.y < 0.0)) e = -e;
    vec2 grad = 0.5 * vec2(sm[5] - sm[3], sm[7] - sm[1]);
    float off = lStrong < -1e-6 ? clamp(dot(grad, e) / lStrong, -3.0, 3.0) : 3.0;
    f_line = vec4(off, e, 0.0);

    float lit = 1.0;
    if (u_depth > 0.0) {
        float h0 = 1.0 - exp(-RELIEF_K * dot(vec3(xr, x.g, x.b), u_lvlRelief));
        float e1 = reliefH(c + ivec2(1, 0));
        float w1 = reliefH(c - ivec2(1, 0));
        float n1 = reliefH(c + ivec2(0, 1));
        float s1 = reliefH(c - ivec2(0, 1));
        float e3 = reliefH(c + ivec2(3, 0));
        float w3 = reliefH(c - ivec2(3, 0));
        float n3 = reliefH(c + ivec2(0, 3));
        float s3 = reliefH(c - ivec2(0, 3));
        vec2 g = mix(vec2(e1 - w1, n1 - s1) * 0.5, vec2(e3 - w3, n3 - s3) / 6.0, RELIEF_BROAD);
        vec3 n = normalize(vec3(-g * RELIEF_BUMP, 1.0));
        vec3 L = u_light;
        float shade = RELIEF_AMB + (1.0 - RELIEF_AMB) * max(dot(n, L), 0.0) / L.z;
        vec2 ld = normalize(L.xy);
        float o2 = reliefH(c + ivec2(round(ld * 2.0))) - h0;
        float o5 = reliefH(c + ivec2(round(ld * 5.0))) - h0;
        float shadow = 1.0 - min(0.75, RELIEF_SHADOW * max(0.0, max(o2, o5 * 0.7)));
        float rng = 0.25 * (e3 + w3 + n3 + s3);
        float cavity = 1.0 - min(0.8, RELIEF_CAVITY * max(0.0, rng - h0));
        float ground = mix(RELIEF_GROUND, 1.0, smoothstep(0.05, 0.45, h0));
        lit = mix(1.0, clamp(shade * shadow * cavity * ground, 0.0, 1.5), u_depth);
    }
    f_color = vec4(f0, f1, f2, lit);
}
