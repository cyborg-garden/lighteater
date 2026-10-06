// browser-shared: dtouch/shaders/alive (README there).
// Alive shuttle pulse, the distance field it travels on: quarter grid
// resolution, ping-pong, a couple of relaxation steps a frame. On the veins
// (the trail at this cell, any level) the value is the distance in cells,
// ALONG the veins, to food: the nearest few percent of the scene (its depth
// above u_src, which the host keeps at a high quantile of the depth), the
// lit subject or the hand in front of it; crossing empty
// ground costs u_offCost a cell, so the field is smooth between the veins
// and its level lines follow the network. The compose pass draws a slow
// wave sin(phase - k D) over it, so brightness and width travel out along
// the network from where the food is. The distance creeps up a little each
// frame (u_creep) so it can follow a network that moves; it only needs to
// be roughly right.
uniform sampler2D u_prev;     // r = distance, cells
uniform sampler2D u_trail;    // grid (nearest: a 32-bit float trail need not filter)
uniform sampler2D u_scene;    // matte res, linear: b = depth
uniform vec3 u_inv_nl;
uniform vec2 u_texel;         // one cell in uv
uniform float u_src;         // depth above which a vein cell is food
uniform float u_creep;
uniform float u_far;
uniform float u_offCost;      // cost of a cell of empty ground, x a cell of vein
uniform float u_reset;
in vec2 v_uv;
layout(location = 0) out vec4 o;
void main() {
  vec3 tn = texture(u_trail, v_uv).rgb * u_inv_nl;
  float vein = step(0.2, 0.6 * tn.r + tn.g + tn.b);
  float near = texture(u_scene, v_uv).b;
  float d = u_reset > 0.5 ? u_far : texture(u_prev, v_uv).r;
  // a step along a vein costs 1 cell, a step across empty ground u_offCost:
  // the field is defined everywhere (so it samples smoothly between veins)
  // but its level lines follow the network
  float cost = mix(u_offCost, 1.0, vein);
  float nb = u_far;
  for (int j = -1; j <= 1; j++)
    for (int i = -1; i <= 1; i++) {
      if (i == 0 && j == 0) continue;
      float w = (i != 0 && j != 0) ? 1.4142 : 1.0;
      nb = min(nb, texture(u_prev, v_uv + vec2(float(i), float(j)) * u_texel).r + w * cost);
    }
  float dn = (near > u_src && vein > 0.5) ? 0.0 : min(d + u_creep, nb);
  o = vec4(min(dn, u_far), 0.0, 0.0, 1.0);
}
