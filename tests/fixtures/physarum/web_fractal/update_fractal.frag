// DEMO, WEB ONLY: moves to the desktop shared unit (dtouch/shaders/physarum)
// if kept. A modified copy of vendor/physarum/update.frag for the fractal
// veins demo; the host loads it only when the fractal amount is above 0, so
// the stock path (amount 0) still runs the vendored file byte for byte.
//
// What changes against the vendored shader: the three trail channels stop
// being three rival species and become three SCALES of one organism.
//   level 0 (r) trunks, level 1 (g) veins, level 2 (b) capillaries/threads.
// An agent's level IS its stock species (.w): species 0 grows trunks, 1
// veins, 2 threads. Each level runs the same Jones step with its sensor
// distance and stride scaled by u_lvlScale.x^level (sensor angle by .y), and
// it reads the trail through the stock rock-paper-scissors row with the
// coarser terms swapped, by u_fractal, for a flank coupling:
//   - a finer level is drawn to the FLANKS of the coarser trail (a bump that
//     peaks at a fraction of the coarse bright end and falls off on the
//     centreline), so capillaries sprout from and hug trunks and lace the
//     gaps between them;
//   - a coarser level is mildly repelled by dense finer lace, so trunks stay
//     clean;
//   - a fine agent standing in empty dark field (no coarse trail, no matte)
//     dies back and reseeds on the subject or beside a trunk; in the room
//     (matte 0) it pays a share of that rate even beside a trunk
//     (u_roomCull), and over the subject the finer levels shrink further
//     (u_bodyFine): the lace is densest and finest on the person or bright
//     thing, and the room keeps mostly trunks;
//   - light stays food, at a smaller share down the ladder (u_lvlFood):
//     a thread that climbs the glow is haze, one that follows trails is lace.
// At u_fractal 0 every one of those terms is the stock shader's own, so the
// organism morphs out of today's picture rather than switching to another.
//
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
uniform float u_fractal;   // 0..1 how far the scales are pulled apart
uniform vec3 u_lvlScale;   // per-level multiplier on (sense, spread, step) at amount 1: x = sense/step ratio, y = spread ratio
uniform float u_inv_norm;  // 1 / the whole trail's bright end (stock norm)
uniform float u_flank;     // how hard a finer level is pulled to the coarser trail's flanks
uniform float u_flankAt;   // where the flank bump peaks, trail units
uniform float u_shun;      // how hard a coarser level is pushed off denser finer lace
uniform float u_dieback;   // per-frame probability a stranded fine agent reseeds
uniform vec3 u_lvlFood;    // per-level share of the light-as-food pull
uniform float u_roomCull;  // share of u_dieback a fine agent in the room (matte 0) pays even beside a trunk
uniform vec2 u_fineMin;    // floor on a finer level's (sensor distance, stride), grid px
uniform vec2 u_fineMax;    // ceiling on level 1, level 2 sensor distance, grid px (blended in by u_fractal)
uniform vec2 u_fineAngle;  // finer levels: ceiling on sensor half-angle (rad), floor on turn as a share of it
uniform vec2 u_trunkAngle; // the same pair for the trunks, looser
uniform float u_calm;      // stride multiplier on every level: the fractal organism moves slower than the stock one
uniform vec3 u_strideK;    // per-level ceiling on stride as a share of sensor distance (blended in by u_fractal)
uniform vec4 u_bloom[2];   // attention zones: centre (grid px), radius (grid px), strength 0..1
uniform float u_bloomFine; // extra shrink of the finer levels at full bloom
uniform float u_bloomPull; // per-frame chance a fine agent outside a bloom relocates into it
uniform float u_bodyFine;  // extra shrink of the finer levels over the subject (matte 1)
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

// How strongly the travelling attention zones touch p (0..1). The zones
// wrap with the torus, like the agents.
float bloomAt(vec2 p) {
    vec2 g = vec2(u_grid);
    float b = 0.0;
    for (int i = 0; i < 2; i++) {
        vec2 d = abs(p - u_bloom[i].xy);
        d = min(d, g - d);
        float r = max(u_bloom[i].z, 1.0);
        b = max(b, u_bloom[i].w * exp(-dot(d, d) / (r * r)));
    }
    return b;
}

// Flank bump in trail units: 0 on empty ground, peaks at x = a with value
// a (so it weighs like a trail of that strength), and falls off again on the
// centreline of a strong trail.
float flankBump(float x, float a) { return x * exp(1.0 - x / a); }

// What level `sp` reads at p. The stock row (see vendor/physarum/update.frag
// food()) with the terms that point at a COARSER level swapped, by
// u_fractal, for a pull toward that level's flanks, and a push off denser
// finer lace for the coarser levels.
float foodFractal(vec2 p, int sp, float repel) {
    float wn = 1.0 - 2.0 * repel;
    float wp = 1.0 - 0.65 * repel;
    vec3 t = trail_at(p);
    float f = u_fractal;
    float s;
    if (sp == 0) {
        s = t.r + (wn - u_shun) * t.g + (wp - u_shun) * t.b;
    } else if (sp == 1) {
        s = mix(wp * t.r, u_flank * flankBump(t.r, u_flankAt), f) + t.g + (wn - 0.5 * u_shun) * t.b;
    } else {
        s = mix(wn * t.r, 0.6 * u_flank * flankBump(t.r, u_flankAt), f)
          + mix(wp * t.g, u_flank * flankBump(t.g, u_flankAt), f) + t.b;
    }
    if (u_satcap > 0.0) {
        float m = abs(s);
        s = sign(s) * u_satcap * (1.0 - exp(-m / u_satcap));
    }
    // Light stays food, but less so down the ladder: a thread that simply
    // climbs the light gradient is haze; one that follows its own trail and
    // the coarser flanks is lace.
    return s + u_food * u_lvlFood[sp] * texelFetch(u_gray, cell(p), 0).r;
}

// Where a stranded fine agent may land: on the lit subject (the stock
// respawn weight) or on the flank of a coarser trail, whichever is better.
float landWeight(vec2 c, int lv) {
    ivec2 ci = cell(c);
    float w = texelFetch(u_matte, ci, 0).r * clamp(texelFetch(u_gray, ci, 0).r, 0.05, 1.0);
    vec3 t = texelFetch(u_trail, ci, 0).rgb;
    float coarse = (lv == 1) ? t.r : max(t.g, 0.7 * t.r);
    float flank = flankBump(coarse, u_flankAt) / u_flankAt;
    // a bloom zone is a strong landing site on a coarser flank: new threads
    // sprout from the trunks there and grow out into the gaps
    return max(w, (0.8 + 2.0 * bloomAt(c)) * flank);
}

void main() {
    ivec2 ac = ivec2(gl_FragCoord.xy);
    uint idx = uint(ac.y * u_aw + ac.x);
    vec4 a = texelFetch(u_agents, ac, 0);
    vec2 p = a.xy;
    float h = a.z;
    int lv = int(clamp(a.w, 0.0, u_nspec - 1.0));
    float L = float(lv);
    uint s = hash(idx * 0x9e3779b9u + u_salt);

    // the pen: blend field -> body by the matte under the agent
    float t = texelFetch(u_matte, cell(p), 0).r;
    float sense = mix(u_sense.x, u_sense.y, t) * u_gain;
    if (u_hetero > 0.0) {
        // the stock random sense split, damped: the levels are the
        // multi-scale structure now, and a random 0.45x/1.9x on top of a
        // deliberate 1/phi ladder smears the ladder back into one scale
        float g3 = mod(float(idx), 3.0);
        float m = (g3 < 0.5) ? 0.45 : (g3 < 1.5) ? 1.0 : 1.9;
        sense *= 1.0 + (m - 1.0) * u_hetero * (1.0 - 0.8 * u_fractal);
    }
    float spread = mix(u_spread.x, u_spread.y, t);
    float turn = mix(u_turn.x, u_turn.y, t);
    float stp = mix(u_step.x, u_step.y, t) * u_gain;

    float repel = u_cross;
    if (u_mosaic > 0.0) {
        vec3 z2; float edge;
        vec3 z = zone_at(p, z2, edge);
        int za = zone_regime(z.x), zb = zone_regime(z2.x);
        vec4 r = mix(u_zreg[za], u_zreg[zb], edge);
        float rc = mix(u_zcross[za], u_zcross[zb], edge);
        sense  *= mix(1.0, r.x, u_mosaic);
        turn   *= mix(1.0, r.y, u_mosaic);
        spread *= mix(1.0, r.z, u_mosaic);
        stp    *= mix(1.0, r.w, u_mosaic);
        repel  *= mix(1.0, rc, u_mosaic);
        repel = min(repel, 1.5);
    }
    // The absolute sensor ceiling caps the TRUNK (the base) before the
    // ladder, not each level after it: capped after, a long-sighted base
    // (web senses 100+ px on a big grid) clamped trunks and veins to the same
    // distance and the ladder collapsed to two scales.
    sense = min(sense, u_sense_max);

    // the self-similar ladder: each level is the one above it shrunk
    float ks = pow(mix(1.0, u_lvlScale.x, u_fractal), L);
    // over the subject the finer levels shrink further still, so the lace
    // is most intricate on the person or bright thing and the room keeps
    // its trunks (level 0 is never touched: the body/field pen already
    // decides the trunks' physics); inside an attention bloom they shrink
    // again, so that patch laces up finer than the rest for a while
    float bl = (lv > 0 && u_fractal > 0.0) ? bloomAt(p) : 0.0;
    ks *= mix(1.0, u_bodyFine, t * 0.5 * L);
    ks *= mix(1.0, u_bloomFine, bl);
    float ka = pow(mix(1.0, u_lvlScale.y, u_fractal), L);
    sense *= ks;
    stp *= ks;
    spread *= ka;
    if (lv > 0) {
        // Absolute ceilings: a finer level is a hairline in grid pixels, not
        // just a fraction of whatever the base point is. A Jones trail is
        // about sense * sin(spread) wide, so web's 26 px "threads" were
        // ropes. Then a floor: a finer level of an already short-sighted
        // point (haze senses a few px) would sense under a texel and barely
        // move; its agents ball up into round glowing dots. Floor it.
        sense = min(sense, mix(sense, (lv == 1) ? u_fineMax.x : u_fineMax.y, u_fractal));
        sense = max(sense, u_fineMin.x);
        // and a thread must be able to follow what it senses: haze looks
        // 1.5 rad aside (x 1.15^L) and turns 0.08, so its fine agents
        // circled in place and printed spiky stars. The classic filament
        // geometry is a narrow fan and a turn at least as wide.
        spread = min(spread, mix(spread, u_fineAngle.x, u_fractal));
        turn = max(turn, u_fineAngle.y * u_fractal * spread);
    } else {
        // the trunks, more loosely: a haze trunk that cannot steer smears
        // into a frosted slab whose inside is all stipple
        spread = min(spread, mix(spread, u_trunkAngle.x, u_fractal));
        turn = max(turn, u_trunkAngle.y * u_fractal * spread);
    }
    // A stride longer than a fraction of the sensor distance stipples the
    // trail into separate dots (lightning's fingers step 3.4 on a 4.5 sense):
    // the grain reads as fuzz. Capped, the motion is also calmer.
    // Calmer as a whole: the trunks are the picture's stable frame and the
    // lace is where the change happens, and neither should rush.
    stp *= mix(1.0, u_calm, u_fractal);
    stp = min(stp, mix(stp, u_strideK[lv] * sense, u_fractal));
    if (lv > 0) stp = max(stp, u_fineMin.y);
    turn *= 1.0 - u_ballistic;

    float fc = foodFractal(p + vec2(cos(h), sin(h)) * sense, lv, repel);
    float fl = foodFractal(p + vec2(cos(h - spread), sin(h - spread)) * sense, lv, repel);
    float fr = foodFractal(p + vec2(cos(h + spread), sin(h + spread)) * sense, lv, repel);
    float dir;
    if (fc > fl && fc > fr) dir = 0.0;
    else if (fc < fl && fc < fr) dir = (rnd(s) < 0.5) ? -1.0 : 1.0;
    else dir = (fl > fr) ? -1.0 : 1.0;
    h += dir * turn;
    if (u_jitter > 0.0) h += (rnd(s) * 2.0 - 1.0) * u_jitter * (1.0 - u_ballistic);

    p += vec2(cos(h), sin(h)) * stp;
    p = mod(p, vec2(u_grid));

    // die-back: a fine agent out in empty dark room (no coarser trail to
    // hang from, no subject under it) is recycled, and one in the room
    // beside a trunk is recycled at u_roomCull of that rate, so the room
    // keeps its trunks and the lace gathers where there is something to lace
    bool reseed = u_reseed > 0.0 && rnd(s) < u_reseed;
    bool pulled = false;
    if (lv > 0 && u_dieback > 0.0) {
        ivec2 pc = cell(p);
        vec3 tt = texelFetch(u_trail, pc, 0).rgb * u_inv_norm;
        float coarse = (lv == 1) ? tt.r : max(tt.g, tt.r);
        float under = texelFetch(u_matte, pc, 0).r;
        float room = 1.0 - smoothstep(0.15, 0.45, under);
        float stranded = max(1.0 - smoothstep(0.005, 0.04, coarse), u_roomCull) * room;
        // inside a bloom the lace is let be, so it can grow out and fill in
        float here = bloomAt(p);
        if (rnd(s) < u_dieback * stranded * (1.0 - here)) reseed = true;
        // and a trickle of fine agents from elsewhere relocates to it: the
        // patch gains lace, the rest of the frame thins a little, and when
        // the zone moves on the lace there relaxes back
        if (rnd(s) < u_bloomPull * (1.0 - here)) { reseed = true; pulled = true; }
    }
    if (reseed) {
        vec2 g = vec2(u_grid);
        if (lv == 0 && u_wmax <= 0.0) {
            p = vec2(rnd(s), rnd(s)) * g;
            h = rnd(s) * 6.2831853;
        } else {
            float bmax = max(u_bloom[0].w, u_bloom[1].w);
            float bound = (lv == 0) ? u_wmax : max(u_wmax, 0.8 + 2.0 * bmax);
            // a pulled agent looks for its landing inside the stronger zone
            vec4 z = (u_bloom[0].w >= u_bloom[1].w) ? u_bloom[0] : u_bloom[1];
            for (int i = 0; i < 16; i++) {
                vec2 c = vec2(rnd(s), rnd(s)) * g;
                if (pulled) {
                    // Box-Muller around the zone centre, one radius wide
                    float r = z.z * sqrt(-2.0 * log(max(rnd(s), 1e-6)));
                    float a = 6.2831853 * rnd(s);
                    c = mod(z.xy + r * vec2(cos(a), sin(a)), g);
                }
                float w = (lv == 0)
                    ? texelFetch(u_matte, cell(c), 0).r * clamp(texelFetch(u_gray, cell(c), 0).r, 0.05, 1.0)
                    : landWeight(c, lv);
                if (rnd(s) * bound < w) {
                    p = c;
                    h = rnd(s) * 6.2831853;
                    break;
                }
            }
        }
    }
    // .w (the stock species) passes through untouched
    f_agent = vec4(p, h, a.w);
}
