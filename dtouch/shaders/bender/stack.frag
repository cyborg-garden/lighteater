// Circuit Bender's long exposure: frame stacking as a running average.
// Each frame is folded into the stack with weight u_alpha (the host picks
// it from the frame time and the exposure, 3, 5 or 10 seconds), so the
// picture is the mean of roughly the last exposure's worth of bent frames:
// what holds still stays sharp, what moves smears. On a machine without
// float render targets the stack is RGBA8 and the average is rounded
// stochastically (u_dither = 1 / 255), so its expected value is still the
// true mean and small weights never get stuck.
uniform sampler2D u_cur;     // this frame, after post.frag
uniform sampler2D u_prev;    // the stack so far
uniform float u_alpha;       // 1 on the first frame of an exposure
uniform float u_dither;      // 0 on a float target
uniform float u_seed;
layout(location = 0) out vec4 o;

float hash(vec3 p) {
  p = fract(p * vec3(0.1031, 0.1030, 0.0973));
  p += dot(p, p.yxz + 33.33);
  return fract((p.x + p.y) * p.z);
}

void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec3 c = texelFetch(u_cur, p, 0).rgb;
  vec3 s = texelFetch(u_prev, p, 0).rgb;
  vec3 m = mix(s, c, u_alpha);
  m += (hash(vec3(gl_FragCoord.xy, u_seed)) - 0.5) * u_dither;
  o = vec4(m, 1.0);
}
