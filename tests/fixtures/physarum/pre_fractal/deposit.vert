// Physarum deposit — attribute-less GL_POINTS draw of N vertices.
//
// Vertex i fetches agent i from the agent texture by gl_VertexID, reads the
// matte under it to blend the two behavior points' deposit, and lands a 1 px
// point on the agent's cell. Blend ONE, ONE into the "laid" texture: that is
// the CPU field's bincount on the GPU.
//
// Only the .r channel of u_matte is read.

uniform sampler2D u_agents;
uniform sampler2D u_matte;
uniform ivec2 u_grid;
uniform int u_aw;
uniform vec2 u_deposit;    // (field point, body point)
uniform float u_nspec;     // active species count (1..3)

out float v_dep;
out vec3 v_mask;      // one-hot species channel

void main() {
    ivec2 ac = ivec2(gl_VertexID % u_aw, gl_VertexID / u_aw);
    vec4 a = texelFetch(u_agents, ac, 0);
    ivec2 c = ivec2(mod(floor(a.xy), vec2(u_grid)));
    v_dep = mix(u_deposit.x, u_deposit.y, texelFetch(u_matte, c, 0).r);
    // each species lays into its own channel, so the others can sense it
    // separately and be repelled by it
    int sp = int(clamp(a.w, 0.0, u_nspec - 1.0));
    v_mask = vec3(sp == 0 ? 1.0 : 0.0, sp == 1 ? 1.0 : 0.0, sp == 2 ? 1.0 : 0.0);
    vec2 ndc = (vec2(c) + 0.5) / vec2(u_grid) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    gl_PointSize = 1.0;
}
