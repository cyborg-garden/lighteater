// browser-shared: dtouch/shaders/alive (README there).
// Alive exploring tips: an attribute-less draw over a fixed subsample of
// the agents (every u_stride-th, so the same agents are followed from frame
// to frame and a head does not sparkle). A fine agent is EXPLORING when its
// own level's trail lies behind it and not ahead: the tip of a thread that
// is still growing. It is drawn as a short fading tail (GL_LINES: vertex 0
// the head, vertex 1 u_tail grid px back along the heading) and, in a
// second draw with u_points = 1, a small round head. Everything else is
// sent off screen.
uniform sampler2D u_agents;
uniform sampler2D u_trail;
uniform ivec2 u_grid;
uniform int u_aw;
uniform int u_n;          // live agents
uniform int u_stride;
uniform vec3 u_inv_nl;
uniform float u_look;     // how far ahead and behind to look, grid px
uniform float u_tail;     // tail length, grid px
uniform float u_points;   // 1 = heads (GL_POINTS), 0 = tails (GL_LINES)
uniform float u_headPx;   // head size, output px
out float v_a;

float own(vec2 p, int lv) {
  ivec2 c = clamp(ivec2(floor(p)), ivec2(0), u_grid - 1);
  vec3 t = texelFetch(u_trail, c, 0).rgb * u_inv_nl;
  return lv == 1 ? t.g : t.b;
}

void main() {
  bool pts = u_points > 0.5;
  int i = pts ? gl_VertexID : gl_VertexID / 2;
  int end = pts ? 0 : gl_VertexID - 2 * i;
  int idx = i * u_stride;
  v_a = 0.0;
  gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
  gl_PointSize = 1.0;
  if (idx >= u_n) return;
  int row = idx / u_aw;                  // idx >= 0, so no integer % needed
  vec4 a = texelFetch(u_agents, ivec2(idx - row * u_aw, row), 0);
  int lv = int(a.w + 0.5);
  if (lv < 1) return;
  vec2 dir = vec2(cos(a.z), sin(a.z));
  float behind = own(a.xy - dir * u_look, lv);
  float ahead = own(a.xy + dir * u_look, lv);
  float s = smoothstep(0.25, 0.7, behind) * (1.0 - smoothstep(0.06, 0.25, ahead));
  if (s < 0.02) return;
  vec2 pos = a.xy - dir * u_tail * float(end);
  gl_Position = vec4(pos / vec2(u_grid) * 2.0 - 1.0, 0.0, 1.0);
  gl_PointSize = u_headPx;
  v_a = s * (end == 0 ? 1.0 : 0.0);
}
