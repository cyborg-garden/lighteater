// browser-shared: dtouch/shaders/alive (README there).
// The alive passes' full-screen triangle (attribute-less, by gl_VertexID).
// One oversized triangle covers the viewport; v_uv is 0..1 over the visible
// area and lands exactly on texel centres for a same-size target.
out vec2 v_uv;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  v_uv = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
