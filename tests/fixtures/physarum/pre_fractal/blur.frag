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

layout(location = 0) out vec4 f_color;

void main() {
    ivec2 c = ivec2(gl_FragCoord.xy);
    vec3 acc = vec3(0.0);
    vec3 wide = vec3(0.0);
    int rw = (u_sharpen > 0.0) ? u_wide : u_radius;
    for (int i = -rw; i <= rw; i++) {
        ivec2 q = c + u_dir * i;
        // wrap via floor, not integer %: GLSL ES 3.00 leaves % undefined
        // when an operand is negative, and q is negative at the low edge
        q -= u_grid * ivec2(floor(vec2(q) / vec2(u_grid)));
        vec3 v = texelFetch(u_src, q, 0).rgb;
        if (u_use_add == 1) v += texelFetch(u_add, q, 0).rgb;
        wide += v;
        if (i >= -u_radius && i <= u_radius) acc += v;
    }
    float d = u_decay;
    if (u_use_keep == 1) {
        float k = clamp(texelFetch(u_keep, c, 0).r, -1.0, 1.0);
        d = u_decay + (0.995 - u_decay) * max(k, 0.0) - 0.12 * max(-k, 0.0);
        d = clamp(d, 0.70, 0.995);
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
