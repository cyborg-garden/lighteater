// Physarum deposit — attribute-less GL_POINTS draw of N vertices.
//
// Vertex i fetches agent i from the agent texture by gl_VertexID, reads the
// matte under it to blend the two behavior points' deposit, and lands a 1 px
// point on the agent's cell. Blend ONE, ONE into the "laid" texture: that is
// the CPU field's bincount on the GPU.
//
// Only the .r channel of u_matte is read.
//
// Fractal veins (u_fractal > 0; 0 is the stock deposit bit for bit): the
// species channel is the agent's LEVEL (r trunks, g veins, b threads). Finer
// levels lay less per agent (u_lvlDep) so they read as thin lines rather
// than more trunks, and in the room (away from the matte) less again
// (u_fineField): the subject carries the densest lace. Inside a travelling
// attention zone (u_bloom) the finer levels lay more (u_bloomDep) and the
// room's cut lifts, so that patch's lace brightens as it forms.

uniform sampler2D u_agents;
uniform sampler2D u_matte;
uniform ivec2 u_grid;
uniform int u_aw;
uniform vec2 u_deposit;    // (field point, body point)
uniform float u_nspec;     // active species count (1..3)
uniform float u_fractal;   // 0 = stock; > 0 reads the four below
uniform vec3 u_lvlDep;     // per-level deposit multiplier
uniform vec2 u_fineField;  // deposit multiplier in the room for level 1, level 2
uniform vec4 u_bloom[2];   // attention zones: centre (grid px), radius (grid px), strength 0..1
uniform float u_bloomDep;  // extra deposit share of the finer levels at full bloom

out float v_dep;
out vec3 v_mask;      // one-hot species channel

void main() {
    ivec2 ac = ivec2(gl_VertexID % u_aw, gl_VertexID / u_aw);
    vec4 a = texelFetch(u_agents, ac, 0);
    ivec2 c = ivec2(mod(floor(a.xy), vec2(u_grid)));
    // each species lays into its own channel, so the others can sense it
    // separately and be repelled by it
    int sp = int(clamp(a.w, 0.0, u_nspec - 1.0));
    if (u_fractal > 0.0) {
        float t = texelFetch(u_matte, c, 0).r;
        float room = sp == 0 ? 1.0 : mix(sp == 1 ? u_fineField.x : u_fineField.y, 1.0, t);
        float bl = 0.0;
        if (sp > 0) {
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
        v_dep = mix(u_deposit.x, u_deposit.y, t) * u_lvlDep[sp] * room * (1.0 + u_bloomDep * bl);
    } else {
        v_dep = mix(u_deposit.x, u_deposit.y, texelFetch(u_matte, c, 0).r);
    }
    v_mask = vec3(sp == 0 ? 1.0 : 0.0, sp == 1 ? 1.0 : 0.0, sp == 2 ? 1.0 : 0.0);
    vec2 ndc = (vec2(c) + 0.5) / vec2(u_grid) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    gl_PointSize = 1.0;
}
