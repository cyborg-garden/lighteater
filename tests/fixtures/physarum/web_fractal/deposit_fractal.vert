// DEMO, WEB ONLY: moves to the desktop shared unit (dtouch/shaders/physarum)
// if kept. A modified copy of vendor/physarum/deposit.vert for the fractal
// veins demo, loaded only when the fractal amount is above 0.
//
// Each agent still lays into its species channel, which in this mode is its
// level (r trunks, g veins, b threads). Finer levels lay less per agent (u_lvlDep) so they read as thin
// lines rather than more trunks, and in the room (away from the matte) they
// lay less again (u_fineField): the subject carries the densest lace, the
// room carries mostly trunks. Inside a travelling attention zone (u_bloom)
// the finer levels lay more (u_bloomDep), so that patch's lace brightens as
// it forms and fades back when the zone moves on.

uniform sampler2D u_agents;
uniform sampler2D u_matte;
uniform ivec2 u_grid;
uniform int u_aw;
uniform vec2 u_deposit;    // (field point, body point)
uniform float u_nspec;
uniform vec3 u_lvlDep;     // per-level deposit multiplier
uniform vec2 u_fineField;  // deposit multiplier in the room for level 1, level 2
uniform vec4 u_bloom[2];   // attention zones: centre (grid px), radius (grid px), strength 0..1
uniform float u_bloomDep;  // extra deposit share of the finer levels at full bloom

out float v_dep;
out vec3 v_mask;

void main() {
    ivec2 ac = ivec2(gl_VertexID % u_aw, gl_VertexID / u_aw);
    vec4 a = texelFetch(u_agents, ac, 0);
    ivec2 c = ivec2(mod(floor(a.xy), vec2(u_grid)));
    float t = texelFetch(u_matte, c, 0).r;
    int lv = int(clamp(a.w, 0.0, u_nspec - 1.0));
    float room = lv == 0 ? 1.0 : mix(lv == 1 ? u_fineField.x : u_fineField.y, 1.0, t);
    float bl = 0.0;
    if (lv > 0) {
        vec2 g = vec2(u_grid);
        for (int i = 0; i < 2; i++) {
            vec2 d = abs(a.xy - u_bloom[i].xy);
            d = min(d, g - d);
            float r = max(u_bloom[i].z, 1.0);
            bl = max(bl, u_bloom[i].w * exp(-dot(d, d) / (r * r)));
        }
    }
    // the bloom also lifts the room's cut, so a zone in the room laces too
    room = mix(room, 1.0, bl);
    v_dep = mix(u_deposit.x, u_deposit.y, t) * u_lvlDep[lv] * room * (1.0 + u_bloomDep * bl);
    v_mask = vec3(lv == 0 ? 1.0 : 0.0, lv == 1 ? 1.0 : 0.0, lv == 2 ? 1.0 : 0.0);
    vec2 ndc = (vec2(c) + 0.5) / vec2(u_grid) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    gl_PointSize = 1.0;
}
