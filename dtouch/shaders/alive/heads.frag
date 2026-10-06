// browser-shared: dtouch/shaders/alive (README there).
// Alive exploring tips: light added (blend ONE, ONE) into the alive
// picture's r (the whole picture) and b (the accents) channels only.
uniform float u_points;
uniform float u_k;        // brightness
in float v_a;
layout(location = 0) out vec4 f_color;
void main() {
  float a = v_a * u_k;
  if (u_points > 0.5) a *= 1.0 - smoothstep(0.25, 0.5, length(gl_PointCoord - 0.5));
  f_color = vec4(a, 0.0, a, 0.0);
}
