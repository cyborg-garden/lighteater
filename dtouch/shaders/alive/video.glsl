// browser-shared: dtouch/shaders/alive (README there).
// Prepended to carve.frag and ink.frag by the host.
// Camera sampling shared by the matte passes: cover-fit the video onto the
// sim aspect and mirror it so the picture behaves like a mirror.
uniform sampler2D u_video;
uniform vec2 u_vidScale;
uniform vec2 u_vidOff;
uniform float u_mirror;
vec2 vuv(vec2 uv) {
  uv.x = mix(uv.x, 1.0 - uv.x, u_mirror);
  return uv * u_vidScale + u_vidOff;
}
float luma(vec2 uv) {
  vec3 c = texture(u_video, vuv(uv)).rgb;
  return dot(c, vec3(0.299, 0.587, 0.114));
}
