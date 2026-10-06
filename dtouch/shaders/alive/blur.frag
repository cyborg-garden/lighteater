// alive+
// browser-shared: dtouch/shaders/alive (README there).
// The vendored blur.frag plus "alive" blocks (alive+ / alive- markers; drop
// them and this is the vendored file byte for byte). With u_alive > 0 the
// fractal branch PRUNES: a level's loss per frame rises where it is no
// longer being laid (u_life .gba, a short memory of the deposits, against
// the trail it holds), so threads nobody walks withdraw and the network
// visibly reorganises around the ones in use (Tero-style flux conductance).
// alive-
// Physarum diffuse + decay — one axis of a separable box blur over the trail.
//
// Run twice per frame with fullscreen.vert:
//   H: u_src = trail,  u_add = laid deposits (u_use_add = 1), u_dir = (1,0),
//      u_scale = 1 / (2r+1), u_decay = 1                     -> tmp
//   V: u_src = tmp,    u_use_add = 0,            u_dir = (0,1),
//      u_scale = 1 / (2r+1), u_decay = decay                 -> next trail
// The V pass may also carry a per-pixel signed "keep" map (u_use_keep = 1):
// where keep is +1 the effective decay is raised to 0.995, so trails linger
// (react paints it from motion history — swept paths hold their veins);
// where keep is -1 the decay is LOWERED by up to 0.12 (floored at 0.70), so
// the trail re-fluidizes (evolve paints it over stale regions — locked
// structure dissolves and regrows instead of ossifying). Semantics mirror
// dtouch.physarum's KEEP_HOLD / MELT_DROP / MELT_FLOOR.
// Edges wrap (the agents already do). All three species channels are
// diffused and decayed together, with the same kernel and the same keep map.
//
// Fractal veins (u_fractal > 0; 0 is the stock arithmetic bit for bit): the
// three channels are three SCALES (r trunks, g veins, b threads), and a
// thread cannot stay a 1 px thread under the box blur that rounds a trunk.
// u_diffC blends each channel between its own undiffused value (0) and the
// full box (1); u_sharpC scales each channel's lateral inhibition; u_decayC
// scales each channel's LOSS per frame (d_c = 1 - (1 - d) * u_decayC): a
// thread laid by a few slow agents has to remember itself longer than a
// trunk a crowd repaints every frame. On the fine channels the inhibition
// may only dig, never raise (see below).

uniform sampler2D u_src;
uniform sampler2D u_add;
uniform sampler2D u_keep;
uniform int u_use_add;
uniform int u_use_keep;
uniform ivec2 u_dir;
uniform int u_radius;      // box half-width in px; 0 = no diffusion
uniform ivec2 u_grid;
uniform float u_scale;
uniform float u_decay;
uniform float u_sharpen;   // lateral inhibition, 0 = plain box blur
uniform int u_wide;        // wide-box half-width for the inhibition surround
uniform float u_fractal;   // 0 = stock; > 0 reads the three below
uniform vec3 u_diffC;      // per-channel diffusion share (1 = full box, 0 = none)
uniform vec3 u_sharpC;     // per-channel multiplier on the lateral inhibition
uniform vec3 u_decayC;     // per-channel multiplier on the per-frame loss, each in (0, 1]
// alive+
uniform float u_alive;     // 0 = vendored; > 0 = pruning (x amount)
uniform sampler2D u_life;  // grid: gba = recent deposits per level (EMA of laid)
uniform vec3 u_prune;      // per level: extra loss share where a trail is unused
uniform vec2 u_pruneAt;    // used-ness ramp on deposits / trail (per frame)
// alive-

layout(location = 0) out vec4 f_color;

void main() {
    ivec2 c = ivec2(gl_FragCoord.xy);
    vec3 acc = vec3(0.0);
    vec3 wide = vec3(0.0);
    vec3 mid = vec3(0.0);
    int rw = (u_sharpen > 0.0) ? u_wide : u_radius;
    for (int i = -rw; i <= rw; i++) {
        ivec2 q = c + u_dir * i;
        // wrap via floor, not integer %: GLSL ES 3.00 leaves % undefined
        // when an operand is negative, and q is negative at the low edge
        q -= u_grid * ivec2(floor(vec2(q) / vec2(u_grid)));
        vec3 v = texelFetch(u_src, q, 0).rgb;
        if (u_use_add == 1) v += texelFetch(u_add, q, 0).rgb;
        wide += v;
        if (i == 0) mid = v;
        if (i >= -u_radius && i <= u_radius) acc += v;
    }
    float d = u_decay;
    if (u_use_keep == 1) {
        float k = clamp(texelFetch(u_keep, c, 0).r, -1.0, 1.0);
        d = u_decay + (0.995 - u_decay) * max(k, 0.0) - 0.12 * max(-k, 0.0);
        d = clamp(d, 0.70, 0.995);
    }
    if (u_fractal > 0.0) {
        vec3 o = mix(mid, acc * u_scale, u_diffC);
        if (u_sharpen > 0.0) {
            vec3 surround = wide * (1.0 / float(2 * rw + 1));
            vec3 inh = max(o - u_sharpen * u_sharpC * (surround - o), vec3(0.0));
            // On the fine levels the centre is barely diffused, so a 1 px
            // thread stands far above its surround and the stock operator
            // would AMPLIFY it every frame (runaway to the float ceiling).
            // There the inhibition may only dig; the trunks keep the stock form.
            o = vec3(inh.r, min(inh.gb, o.gb));
        }
        // loss scaled per channel; with d < 1 and u_decayC > 0 nothing grows
        // alive+
        if (u_alive > 0.0) {
            vec3 fl = texelFetch(u_life, c, 0).gba;
            vec3 used = smoothstep(vec3(u_pruneAt.x), vec3(u_pruneAt.y), fl / max(o, vec3(1e-4)));
            vec3 cut = u_decayC * (1.0 + u_alive * u_prune * (1.0 - used));
            f_color = vec4(o * max(1.0 - (1.0 - d) * cut, vec3(0.0)), 1.0);
            return;
        }
        // alive-
        f_color = vec4(o * (1.0 - (1.0 - d) * u_decayC), 1.0);
        return;
    }
    vec3 out3 = acc * u_scale;
    // Lateral inhibition. A plain box blur is the most structure-destroying
    // kernel there is at a given radius: it only ever smears. Subtracting a
    // slice of the WIDER surround turns diffusion into a centre-surround
    // operator, so a strong vein suppresses its own neighbourhood — which
    // sharpens the vein and digs the dark halo around it. Those halos, and
    // the hard seams they make between neighbouring structures, are most of
    // what reads as "carved" rather than "smoked".
    if (u_sharpen > 0.0) {
        vec3 surround = wide * (1.0 / float(2 * rw + 1));
        out3 = max(out3 - u_sharpen * (surround - out3), vec3(0.0));
    }
    f_color = vec4(out3 * d, 1.0);
}
