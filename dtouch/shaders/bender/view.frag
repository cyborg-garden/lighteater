// Circuit Bender's last pass: the working picture onto the canvas. The
// working size is about one pixel per CSS pixel (dtouch/bender.py
// working_size), so the scale onto the canvas is rarely a whole number: a
// sharp bilinear filter keeps every working pixel a crisp square and blends
// only the one canvas pixel where two meet, so 8x8 JPEG blocks stay square
// and even instead of the uneven columns plain NEAREST gives at 1.25x or
// 1.5x. Row 0 of the texture is the image top: flip once here.
uniform sampler2D u_src;
uniform ivec2 u_size;        // working size
uniform ivec2 u_canvas;      // canvas size
uniform float u_blackout;    // 1 = black
layout(location = 0) out vec4 o;

vec3 at(ivec2 p) {
  return texelFetch(u_src, clamp(p, ivec2(0), u_size - 1), 0).rgb;
}

void main() {
  vec2 scale = vec2(u_canvas) / vec2(u_size);   // canvas pixels per working pixel
  vec2 dst = vec2(gl_FragCoord.x, float(u_canvas.y) - gl_FragCoord.y);
  vec2 tp = dst / scale - 0.5;                   // texel space, centres on integers
  vec2 i = floor(tp);
  vec2 f = tp - i;
  // the blend band is one canvas pixel wide, wherever the edge falls
  f = clamp((f - 0.5) * max(scale, vec2(1.0)) + 0.5, 0.0, 1.0);
  ivec2 b = ivec2(i);
  vec3 c = mix(mix(at(b), at(b + ivec2(1, 0)), f.x), mix(at(b + ivec2(0, 1)), at(b + ivec2(1, 1)), f.x), f.y);
  o = vec4(c * (1.0 - u_blackout), 1.0);
}
