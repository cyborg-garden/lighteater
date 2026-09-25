// Physarum agent update — one Jones step per texel of the agent texture.
//
// Agent texture (RGBA32F): x, y, heading, spare. Drawn over the agent
// texture with fullscreen.vert; writes the next agent texture (ping-pong).
// Per agent: read the matte under it (the pen) and blend the field/body
// behavior points, sense trail+food at three points ahead, turn toward the
// strongest, step, wrap, and — with probability u_reseed — respawn onto the
// lit subject (matte * luma) by rejection sampling.
//
// The trail carries THREE species channels (rgb); u_matte and u_gray are
// still .r only.

uniform sampler2D u_agents;
uniform sampler2D u_trail;
uniform sampler2D u_matte;
uniform sampler2D u_gray;
uniform ivec2 u_grid;      // trail size (gw, gh)
uniform int u_aw;          // agent texture width
uniform vec2 u_sense;      // (field point, body point) — sensor distance, px
uniform vec2 u_spread;     // sensor half-angle, rad
uniform vec2 u_turn;       // rotation when a side sensor wins, rad
uniform vec2 u_step;       // stride per frame, px
uniform float u_gain;      // tempo: scales sense + step
uniform float u_food;      // how strongly luma is added to what sensors read
uniform float u_reseed;    // per-agent respawn probability this frame
uniform float u_wmax;      // upper bound of matte*clamp(gray,.05,1); <= 0: respawn uniformly
uniform uint u_salt;       // per-frame random salt
uniform float u_satcap;    // sensed-trail soft cap (absolute units); <= 0 = off
uniform float u_jitter;    // per-step heading wobble, rad; 0 = off
uniform float u_hetero;    // 0..1 blend toward the 3-sub-population sense split
uniform float u_cross;     // species cross-term: how hard the populations push
uniform float u_nspec;     // active species count (1..3); 1 = legacy single
uniform float u_mosaic;    // 0..1 how differently the zones behave; 0 = uniform
uniform float u_zones;     // zone lattice density, cells across the grid
uniform float u_sense_max; // absolute ceiling on sensor distance, px
uniform float u_ballistic; // 0..1 steering suppression (the wave's shockwave)
uniform float u_time;      // seconds, drives the zones' slow drift
uniform vec4 u_zreg[6];    // per-zone-regime multipliers: sense, turn, spread, step
uniform float u_zcross[6]; // per-zone-regime multiplier on u_cross

layout(location = 0) out vec4 f_agent;

uint hash(uint x) {
    x ^= x >> 16u; x *= 0x7feb352du;
    x ^= x >> 15u; x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}
float rnd(inout uint s) { s = hash(s); return float(s) * (1.0 / 4294967296.0); }

ivec2 cell(vec2 p) { return ivec2(mod(floor(p), vec2(u_grid))); }

vec3 hash3(vec2 c) {
    uint h = hash(uint(int(c.x)) * 0x27d4eb2du ^ hash(uint(int(c.y)) * 0x9e3779b9u));
    uint a = hash(h), b = hash(a);
    return vec3(float(h), float(a), float(b)) * (1.0 / 4294967296.0);
}

// ---- the mosaic -----------------------------------------------------------
// A single set of parameters over the whole canvas can only ever grow one
// texture; a mold that is the same everywhere looks the same everywhere, no
// matter how the sliders move. So the grid is partitioned into drifting
// Voronoi zones and each zone gets its own multipliers, drawn from its own
// hash. Boundaries are hard: an agent crossing one changes behaviour on that
// step, which is what puts fronts, halos and abutting morphologies into the
// same frame instead of one averaged mesh.
//
// Returns the zone's three hash values in .xyz. The lattice wraps with the
// torus the agents already live on.
vec3 zone_at(vec2 p, out vec3 second, out float edge) {
    vec2 g = vec2(u_grid);
    // Domain warp before the lattice lookup. A Voronoi diagram is made of
    // straight lines, and a straight line across a slime mold is instantly
    // legible as machinery — `web`, whose sensors reach 170 px, sampled clean
    // across the seams and printed the polygons into the picture. Two cheap
    // octaves bend the boundaries into something the organism could have
    // grown. The warp is in lattice units so it scales with zone size.
    vec2 n = p / g * u_zones;
    n += 0.22 * vec2(sin(n.y * 2.7 + u_time * 0.05) + 0.5 * sin(n.y * 6.1 - u_time * 0.031),
                     cos(n.x * 2.3 - u_time * 0.043) + 0.5 * cos(n.x * 5.3 + u_time * 0.037));
    vec2 i0 = floor(n);
    float best = 1e9, next = 1e9;
    vec3 h = vec3(0.5), h2 = vec3(0.5);
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            vec2 gi = i0 + vec2(float(dx), float(dy));
            vec3 r = hash3(mod(gi, vec2(u_zones)));
            vec2 site = gi + 0.5 + 0.42 * vec2(
                sin(u_time * (0.09 + 0.08 * r.x) + 6.2831853 * r.y),
                cos(u_time * (0.07 + 0.07 * r.y) + 6.2831853 * r.z));
            vec2 d = n - site;
            float dd = dot(d, d);
            if (dd < best) { next = best; h2 = h; best = dd; h = r; }
            else if (dd < next) { next = dd; h2 = r; }
        }
    }
    second = h2;
    // 0 deep inside a zone, 0.5 on the seam: how far to cross-fade the two
    // regimes. Narrow, so the seam stays a seam and not a gradient.
    float b = sqrt(max(best, 0.0)), nx = sqrt(max(next, 0.0));
    edge = 0.5 * (1.0 - smoothstep(0.0, 0.16, nx - b));
    return h;
}

// A zone picks ONE of six regimes, not a smooth blend of all of them. That is
// the point: continuous multipliers average back into a single texture, while
// a decisive pick puts fine lace hard against fat trunks against a radial
// bloom, with a real seam between them. The table is ours (see
// dtouch.physarum.SPATIAL_REGIMES), derived from this model's own points.
int zone_regime(float v) { return int(clamp(floor(v * 6.0), 0.0, 5.0)); }

// Bilinear trail read. texelFetch quantizes the field to whole cells, which
// throws away exactly the sub-cell gradient the short-sense points steer on —
// two sensors less than a texel apart returned the SAME value, so fl == fr and
// the agent turned one way forever. Four fetches and a lerp, wrapped.
vec3 trail_at(vec2 p) {
    vec2 g = vec2(u_grid);
    vec2 q = p - 0.5;
    vec2 f = fract(q);
    ivec2 c0 = ivec2(mod(floor(q), g));
    ivec2 c1 = ivec2(mod(floor(q) + 1.0, g));
    vec3 t00 = texelFetch(u_trail, ivec2(c0.x, c0.y), 0).rgb;
    vec3 t10 = texelFetch(u_trail, ivec2(c1.x, c0.y), 0).rgb;
    vec3 t01 = texelFetch(u_trail, ivec2(c0.x, c1.y), 0).rgb;
    vec3 t11 = texelFetch(u_trail, ivec2(c1.x, c1.y), 0).rgb;
    return mix(mix(t00, t10, f.x), mix(t01, t11, f.x), f.y);
}

// What species `sp` reads at p. Its own channel attracts; the other two pull
// with signed weights, built here one row at a time rather than passed as a
// matrix, because `repel` varies per agent with the zone it is standing in.
//
// The negative terms are the whole point. A species that avoids another's
// trail leaves a thin dark exclusion membrane between their territories, and
// single-channel attract-only physarum cannot produce that at any parameter
// setting — which is why every look used to read as the same mesh recoloured.
// The arrangement is rock-paper-scissors (repelled by the NEXT species,
// mildly drawn to the previous); the asymmetry is what makes the membranes
// travel and chase instead of freezing into a static partition.
//
// `repel`, not `cross`: cross() is a GLSL built-in, and shadowing it is legal
// but not something to bet a driver on.
float food(vec2 p, int sp, float repel) {
    // The row BLENDS from all-ones toward rock-paper-scissors. At repel 0
    // every species reads the total trail, which is exactly the old
    // single-channel model: three populations depositing into three channels
    // and sensing their sum is arithmetically identical to one population
    // depositing into one. That matters — the bottom of `weave` is supposed
    // to BE the legacy bold-canal engine, and an identity row instead gave
    // three mutually invisible organisms at a third of the density each,
    // which is a different picture wearing the same label.
    float wn = 1.0 - 2.0 * repel;          // the next species: +1 -> -1
    float wp = 1.0 - 0.65 * repel;         // the previous one: +1 -> +0.35
    vec3 row = (sp == 0) ? vec3(1.0, wn, wp)
             : (sp == 1) ? vec3(wp, 1.0, wn)
                         : vec3(wn, wp, 1.0);
    if (u_nspec <= 1.0) row = vec3(1.0, 0.0, 0.0);
    float t = dot(row, trail_at(p));
    // Sensor saturation, signed. The cap softly compresses what a sensor can
    // report, so a fat canal reads much like a merely strong thin vein and
    // the mold stops pouring everything into its own highways. It sits ABOVE
    // the trail's bright end: the old cap sat at ~0.2x it, which flattened
    // the entire sensed field into a few units and left every agent inside
    // the network steering on noise. `t` is signed now that repulsion exists,
    // and exp(-t/C) diverges on a negative t, so cap the magnitude and keep
    // the sign.
    if (u_satcap > 0.0) {
        float m = abs(t);
        t = sign(t) * u_satcap * (1.0 - exp(-m / u_satcap));
    }
    // u_food arrives already scaled into trail units by the host (x the
    // trail's own bright end), exactly as u_satcap is. Unscaled, the video's
    // 0..1 luma is a ~1% perturbation of a field whose mean is ~33, and the
    // camera stops steering the mold the moment any trail exists anywhere.
    return t + u_food * texelFetch(u_gray, cell(p), 0).r;
}

void main() {
    ivec2 ac = ivec2(gl_FragCoord.xy);
    uint idx = uint(ac.y * u_aw + ac.x);
    vec4 a = texelFetch(u_agents, ac, 0);
    vec2 p = a.xy;
    float h = a.z;
    // .w carries the agent's species. It is assigned once, at spawn, and
    // survives reseeds and impulses: a lineage, not a per-frame lookup of
    // where the agent happens to be standing.
    int sp = int(clamp(a.w, 0.0, u_nspec - 1.0));
    uint s = hash(idx * 0x9e3779b9u + u_salt);

    // the pen: blend field -> body by the matte under the agent
    float t = texelFetch(u_matte, cell(p), 0).r;
    float sense = mix(u_sense.x, u_sense.y, t) * u_gain;
    if (u_hetero > 0.0) {
        // 3 sub-populations by agent index (float mod — ES 3.00 has no
        // integer % for this): short / mid / long sense ranges, blended in
        // by u_hetero. Multi-scale sensing grows multi-scale structure.
        float g3 = mod(float(idx), 3.0);
        float m = (g3 < 0.5) ? 0.45 : (g3 < 1.5) ? 1.0 : 1.9;
        sense *= 1.0 + (m - 1.0) * u_hetero;
    }
    float spread = mix(u_spread.x, u_spread.y, t);
    float turn = mix(u_turn.x, u_turn.y, t);
    float stp = mix(u_step.x, u_step.y, t) * u_gain;

    // the zone the agent is standing in decides which regime it runs
    float repel = u_cross;
    if (u_mosaic > 0.0) {
        vec3 z2; float edge;
        vec3 z = zone_at(p, z2, edge);
        int za = zone_regime(z.x), zb = zone_regime(z2.x);
        // hard inside the zone, a narrow cross-fade on the seam itself, so
        // the transition is a front rather than a visible polygon edge
        vec4 r = mix(u_zreg[za], u_zreg[zb], edge);
        float rc = mix(u_zcross[za], u_zcross[zb], edge);
        sense  *= mix(1.0, r.x, u_mosaic);
        turn   *= mix(1.0, r.y, u_mosaic);
        spread *= mix(1.0, r.z, u_mosaic);
        stp    *= mix(1.0, r.w, u_mosaic);
        repel  *= mix(1.0, rc, u_mosaic);
        repel = min(repel, 1.5);   // zones scale it; do not let it run away
    }
    // Sensing further than a tenth of the frame is sensing globally: the
    // three sensors stop reporting on a neighbourhood and the mold dissolves
    // into a cloud with no veins in it. `web` already sits at the ceiling by
    // design, and the mosaic's long-range regimes would otherwise multiply it
    // past the point where any local structure can survive.
    sense = min(sense, u_sense_max);

    // The wave points every heading outward, and in a field this lively the
    // very next frame steers most of them back — the gesture used to spend
    // itself in about one frame. For a beat afterwards the organism goes
    // BALLISTIC instead: steering and wobble are suppressed, so the outward
    // front actually travels and reads as a shockwave rather than a flicker.
    // It decays, so the mold does not simply fly apart.
    turn *= 1.0 - u_ballistic;

    // Jones steering: hold when ahead wins; coin flip when ahead loses to
    // both sides; otherwise turn toward the stronger side.
    float fc = food(p + vec2(cos(h), sin(h)) * sense, sp, repel);
    float fl = food(p + vec2(cos(h - spread), sin(h - spread)) * sense, sp, repel);
    float fr = food(p + vec2(cos(h + spread), sin(h + spread)) * sense, sp, repel);
    float dir;
    if (fc > fl && fc > fr) dir = 0.0;
    else if (fc < fl && fc < fr) dir = (rnd(s) < 0.5) ? -1.0 : 1.0;
    else dir = (fl > fr) ? -1.0 : 1.0;
    h += dir * turn;
    // per-step heading wobble: highways stop being perfectly straight
    // attractors and the mold keeps probing sideways
    if (u_jitter > 0.0) h += (rnd(s) * 2.0 - 1.0) * u_jitter * (1.0 - u_ballistic);

    p += vec2(cos(h), sin(h)) * stp;
    p = mod(p, vec2(u_grid));

    // recycle a trickle of agents onto the lit subject (matte * luma)
    if (u_reseed > 0.0 && rnd(s) < u_reseed) {
        vec2 g = vec2(u_grid);
        if (u_wmax <= 0.0) {
            p = vec2(rnd(s), rnd(s)) * g;
            h = rnd(s) * 6.2831853;
        } else {
            for (int i = 0; i < 16; i++) {
                vec2 c = vec2(rnd(s), rnd(s)) * g;
                ivec2 ci = cell(c);
                float w = texelFetch(u_matte, ci, 0).r
                        * clamp(texelFetch(u_gray, ci, 0).r, 0.05, 1.0);
                if (rnd(s) * u_wmax < w) {
                    p = c;
                    h = rnd(s) * 6.2831853;
                    break;
                }
            }
        }
    }
    f_agent = vec4(p, h, a.w);
}
