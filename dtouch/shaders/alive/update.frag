// alive+
// browser-shared: dtouch/shaders/alive (README there).
// The vendored update.frag plus the "alive" additions, each between an
// `alive+` / `alive-` marker pair, so that dropping every such block gives
// back the vendored file byte for byte (tests/alive.test.mjs holds that).
// Loaded only on H's fractal step. With u_alive = 1:
//   - hunting: fresh motion (scene .r, a fast-decaying food map) reads as
//     food, mostly for the finer levels, and its wide skirt (scene .a)
//     leans headings toward it, so threads reach for a moving hand;
//     reseeding fine agents land on the network's flanks at the edge of the
//     motion, and a trickle is pulled there, heading in;
//   - bending in: headings turn onto the lines of equal depth (scene .b, a
//     matte dome or a luminance pseudo-depth), trunks hardest, and strides
//     slow on steep depth, so veins wrap around shapes.
// alive-
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
//
// Fractal veins (u_fractal > 0; 0 is the stock step above, bit for bit —
// every fractal term sits behind a `u_fractal > 0.0` branch, and
// tests/test_physarum_fractal.py holds this against a frozen copy of the
// pre-fractal shader). The three channels stop being three rival species
// and become three SCALES of one organism: level 0 (r) trunks, level 1 (g)
// veins, level 2 (b) threads. An agent's level IS its species (.w). Each
// level runs the same Jones step with its sensor distance and stride scaled
// by u_lvlScale.x^level (sensor angle by .y), and reads the trail through
// the stock rock-paper-scissors row with the coarser terms swapped, by
// u_fractal, for a flank coupling:
//   - a finer level is drawn to the FLANKS of the coarser trail (a bump that
//     peaks at a fraction of the coarse bright end and falls off on the
//     centreline), so threads sprout from and hug trunks and lace the gaps;
//   - a coarser level is mildly repelled by dense finer lace (u_shun);
//   - a fine agent standing in empty dark field (no coarse trail, no matte)
//     dies back and reseeds on the subject or beside a trunk; in the room
//     (matte 0) it pays u_roomCull of that rate even beside a trunk, and
//     over the subject the finer levels shrink further (u_bodyFine): the
//     lace is densest and finest on the person, the room keeps its trunks;
//   - light stays food, at a smaller share down the ladder (u_lvlFood);
//   - two travelling attention zones (u_bloom) shrink the finer levels
//     further, spare them from die-back and pull a trickle of fine agents in.
// Tuning lives in dtouch.physarum.FRACTAL (exported to looks.json).

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
// fractal veins — read only when u_fractal > 0 (a host that never sets them
// runs the stock step)
uniform float u_fractal;   // 0..1 how far the scales are pulled apart; 0 = stock
uniform vec3 u_lvlScale;   // x = per-level sense/step ratio, y = per-level spread ratio (at amount 1)
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
uniform float u_calm;      // stride multiplier on every level (blended in by u_fractal)
uniform vec3 u_strideK;    // per-level ceiling on stride as a share of sensor distance (blended in by u_fractal)
uniform vec4 u_bloom[2];   // attention zones: centre (grid px), radius (grid px), strength 0..1
uniform float u_bloomFine; // extra shrink of the finer levels at full bloom
uniform float u_bloomPull; // per-frame chance a fine agent outside a bloom relocates into it
uniform float u_bodyFine;  // extra shrink of the finer levels over the subject (matte 1)
// alive+
uniform float u_alive;       // 0 = the vendored step exactly; > 0 = hunting + contour steering (x amount)
uniform sampler2D u_scene;   // matte res, linear: r fresh motion, g matte distance, b depth, a fresh skirt
uniform vec2 u_sceneTexel;   // one scene texel in uv
uniform vec3 u_freshFood;    // per level, trail units: fresh motion read as food
uniform vec3 u_freshTurn;    // per level: largest turn per frame toward the rising skirt, rad
uniform float u_freshGain;   // skirt slope (per scene texel) at which that turn is full
uniform float u_freshLand;   // landing weight of fresh motion for a reseeding fine agent
uniform float u_freshPull;   // per-frame chance a fine agent outside the motion relocates onto it
uniform vec3 u_contour;      // per level: share of the angle to the depth contour turned per frame
uniform float u_contourGain; // depth slope (per scene texel) at which contour steering is full
uniform float u_slope;       // stride on steep depth: 1 / sqrt(1 + (slope * steep)^2)
float aliveFresh(vec2 p) { return texture(u_scene, p / vec2(u_grid)).r; }
// signed angle from unit vector f to direction t
float aliveAngle(vec2 f, vec2 t) { return atan(f.x * t.y - f.y * t.x, dot(f, t)); }
// alive-

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

// ---- fractal veins (u_fractal > 0 only) -----------------------------------

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

// What level `sp` reads at p: food()'s row with the terms that point at a
// COARSER level swapped, by u_fractal, for a pull toward that level's
// flanks, and a push off denser finer lace for the coarser levels.
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
    // alive+
    // fresh motion is fast food, strongest for the finest level
    if (u_alive > 0.0) s += u_freshFood[sp] * aliveFresh(p);
    // alive-
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
    // alive+
    // near fresh motion, a coarser flank is a strong landing site: new
    // threads sprout from the network at the edge of the motion and reach in
    if (u_alive > 0.0) w = max(w, u_freshLand * texture(u_scene, c / vec2(u_grid)).a * flank);
    // alive-
    return max(w, (0.8 + 2.0 * bloomAt(c)) * flank);
}

// The self-similar ladder: each level is the one above it shrunk, with
// absolute ceilings and floors so a finer level is a hairline in grid pixels
// on any base point, and a stride capped to its sensing so no trail stipples.
// Runs after the stock sensor ceiling (which so caps the TRUNK, the base,
// and not each level: capped after, a long-sighted base clamped trunks and
// veins to the same distance and the ladder collapsed to two scales).
void ladder(int lv, float t, vec2 p, inout float sense, inout float spread,
            inout float turn, inout float stp) {
    float L = float(lv);
    float ks = pow(mix(1.0, u_lvlScale.x, u_fractal), L);
    // over the subject the finer levels shrink further still (level 0 is
    // never touched: the body/field pen already decides the trunks' physics);
    // inside an attention bloom they shrink again
    float bl = (lv > 0 && u_fractal > 0.0) ? bloomAt(p) : 0.0;
    ks *= mix(1.0, u_bodyFine, t * 0.5 * L);
    ks *= mix(1.0, u_bloomFine, bl);
    float ka = pow(mix(1.0, u_lvlScale.y, u_fractal), L);
    sense *= ks;
    stp *= ks;
    spread *= ka;
    if (lv > 0) {
        // a Jones trail is about sense * sin(spread) wide: ceilings make a
        // finer level a hairline; the floor keeps a short-sighted point's
        // finer level from sensing under a texel and balling into dots
        sense = min(sense, mix(sense, (lv == 1) ? u_fineMax.x : u_fineMax.y, u_fractal));
        sense = max(sense, u_fineMin.x);
        // a thread must be able to follow what it senses: a narrow fan and a
        // turn at least as wide, or it circles in place and prints stars
        spread = min(spread, mix(spread, u_fineAngle.x, u_fractal));
        turn = max(turn, u_fineAngle.y * u_fractal * spread);
    } else {
        // the trunks, more loosely: a trunk that cannot steer smears into a
        // frosted slab
        spread = min(spread, mix(spread, u_trunkAngle.x, u_fractal));
        turn = max(turn, u_trunkAngle.y * u_fractal * spread);
    }
    // calmer as a whole, and a stride past a fraction of the sensor distance
    // stipples the trail into separate dots
    stp *= mix(1.0, u_calm, u_fractal);
    stp = min(stp, mix(stp, u_strideK[lv] * sense, u_fractal));
    if (lv > 0) stp = max(stp, u_fineMin.y);
}

// The stock reseed plus die-back and the bloom pull; returns (p, heading).
// A fine agent out in empty dark room (no coarser trail to hang from, no
// subject under it) is recycled, and one in the room beside a trunk at
// u_roomCull of that rate, so the room keeps its trunks and the lace gathers
// where there is something to lace. Inside a bloom the lace is let be, and a
// trickle of fine agents from elsewhere relocates into it.
vec3 fractalReseed(int lv, vec2 p, float h, inout uint s) {
    bool reseed = u_reseed > 0.0 && rnd(s) < u_reseed;
    bool pulled = false;
    if (lv > 0 && u_dieback > 0.0) {
        ivec2 pc = cell(p);
        vec3 tt = texelFetch(u_trail, pc, 0).rgb * u_inv_norm;
        float coarse = (lv == 1) ? tt.r : max(tt.g, tt.r);
        float under = texelFetch(u_matte, pc, 0).r;
        float room = 1.0 - smoothstep(0.15, 0.45, under);
        float stranded = max(1.0 - smoothstep(0.005, 0.04, coarse), u_roomCull) * room;
        float here = bloomAt(p);
        if (rnd(s) < u_dieback * stranded * (1.0 - here)) reseed = true;
        if (rnd(s) < u_bloomPull * (1.0 - here)) { reseed = true; pulled = true; }
    }
    // alive+
    // a trickle of fine agents relocates to the network's edge by fresh
    // motion (a coarser flank inside the skirt), heading into the motion,
    // so new threads reach out from the veins toward a moving hand
    if (u_alive > 0.0 && lv > 0 && u_freshPull > 0.0 && rnd(s) < u_freshPull * (1.0 - aliveFresh(p))) {
        vec2 g = vec2(u_grid);
        for (int i = 0; i < 12; i++) {
            vec2 c = vec2(rnd(s), rnd(s)) * g;
            vec2 uc = c / g;
            float sk = texture(u_scene, uc).a;
            if (sk < 0.05) continue;
            vec3 tt = texelFetch(u_trail, cell(c), 0).rgb;
            float coarse = (lv == 1) ? tt.r : max(tt.g, 0.7 * tt.r);
            float fl = flankBump(coarse, u_flankAt) / u_flankAt;
            if (rnd(s) < sk * min(fl, 1.0)) {
                vec2 gs = vec2(texture(u_scene, uc + vec2(u_sceneTexel.x, 0.0)).a - texture(u_scene, uc - vec2(u_sceneTexel.x, 0.0)).a,
                               texture(u_scene, uc + vec2(0.0, u_sceneTexel.y)).a - texture(u_scene, uc - vec2(0.0, u_sceneTexel.y)).a);
                float hd = dot(gs, gs) > 1e-10 ? atan(gs.y, gs.x) : rnd(s) * 6.2831853;
                return vec3(c, hd + (rnd(s) - 0.5) * 0.8);
            }
        }
    }
    // alive-
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
    return vec3(p, h);
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
        if (u_fractal > 0.0) {
            // damped: the levels are the multi-scale structure now, and a
            // random 0.45x/1.9x on top of a deliberate ladder smears the
            // ladder back into one scale
            sense *= 1.0 + (m - 1.0) * u_hetero * (1.0 - 0.8 * u_fractal);
        } else {
            sense *= 1.0 + (m - 1.0) * u_hetero;
        }
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
    if (u_fractal > 0.0) ladder(sp, t, p, sense, spread, turn, stp);
    // alive+
    if (u_alive > 0.0) {
        vec2 uvS = p / vec2(u_grid);
        vec2 dx = vec2(u_sceneTexel.x, 0.0), dy = vec2(0.0, u_sceneTexel.y);
        vec2 fwd = vec2(cos(h), sin(h));
        // bending in: turn onto the line of equal depth, on the side the
        // agent already faces, harder where the depth is steep; slower there
        vec2 gH = 0.5 * vec2(texture(u_scene, uvS + dx).b - texture(u_scene, uvS - dx).b,
                             texture(u_scene, uvS + dy).b - texture(u_scene, uvS - dy).b);
        float steep = length(gH) * u_contourGain;
        if (steep > 1e-3) {
            vec2 tg = vec2(-gH.y, gH.x);
            if (dot(tg, fwd) < 0.0) tg = -tg;
            h += aliveAngle(fwd, tg) * u_contour[sp] * min(steep, 1.0);
            stp /= sqrt(1.0 + u_slope * u_slope * steep * steep);
        }
        // hunting: lean toward the rising skirt of fresh motion
        vec2 gF = 0.25 * vec2(texture(u_scene, uvS + 2.0 * dx).a - texture(u_scene, uvS - 2.0 * dx).a,
                              texture(u_scene, uvS + 2.0 * dy).a - texture(u_scene, uvS - 2.0 * dy).a);
        float lean = length(gF) * u_freshGain;
        if (lean > 1e-3) {
            float a = clamp(aliveAngle(fwd, gF), -u_freshTurn[sp], u_freshTurn[sp]);
            h += a * min(lean, 1.0);
        }
    }
    // alive-

    // The wave points every heading outward, and in a field this lively the
    // very next frame steers most of them back — the gesture used to spend
    // itself in about one frame. For a beat afterwards the organism goes
    // BALLISTIC instead: steering and wobble are suppressed, so the outward
    // front actually travels and reads as a shockwave rather than a flicker.
    // It decays, so the mold does not simply fly apart.
    turn *= 1.0 - u_ballistic;

    // Jones steering: hold when ahead wins; coin flip when ahead loses to
    // both sides; otherwise turn toward the stronger side.
    float fc, fl, fr;
    if (u_fractal > 0.0) {
        fc = foodFractal(p + vec2(cos(h), sin(h)) * sense, sp, repel);
        fl = foodFractal(p + vec2(cos(h - spread), sin(h - spread)) * sense, sp, repel);
        fr = foodFractal(p + vec2(cos(h + spread), sin(h + spread)) * sense, sp, repel);
    } else {
        fc = food(p + vec2(cos(h), sin(h)) * sense, sp, repel);
        fl = food(p + vec2(cos(h - spread), sin(h - spread)) * sense, sp, repel);
        fr = food(p + vec2(cos(h + spread), sin(h + spread)) * sense, sp, repel);
    }
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

    if (u_fractal > 0.0) {
        // .w (the level) passes through untouched
        f_agent = vec4(fractalReseed(sp, p, h, s), a.w);
        return;
    }

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
