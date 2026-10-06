// Circuit Bender's post pass, at the working size: COPY (block runs and
// echoed bands) and SPLIT (chromatic aberration). Both are LightEater's own
// readings of effect names in the CyberShot Cam guide's web gallery
// ("copy fx", "chromatic aberration"); no code from that project is used.
// The desktop runs the same rule in numpy (dtouch/bender_post.py post).
//
// Texel space throughout: row 0 is the image top, as uploaded; view.frag
// flips once on the way to the canvas.
uniform sampler2D u_src;     // the decoded, bent frame (RGBA8)
uniform ivec2 u_size;        // working size
uniform float u_split;       // red right, blue left, in working pixels
uniform float u_copy;        // 0 = off, up to 1 = dense
uniform float u_seed;        // changes slowly (the cut clock), so runs hold
uniform int u_block;         // run grid: one 16x16 MCU of a 4:2:0 JPEG
layout(location = 0) out vec4 o;

float hash(vec3 p) {
  p = fract(p * vec3(0.1031, 0.1030, 0.0973));
  p += dot(p, p.yxz + 33.33);
  return fract((p.x + p.y) * p.z);
}

// Where this pixel copies from. A block can start a run: the blocks to its
// right, up to its run length, repeat it (the stuck-decoder look of a JPEG
// that lost its place). A block row can also echo the rows a few blocks
// above it.
// The point is clamped into the frame first: SPLIT asks for points past the
// edge, and integer / and % are only defined here on non-negative operands.
ivec2 copySrc(ivec2 p) {
  p = clamp(p, ivec2(0), u_size - 1);
  if (u_copy <= 0.0) return p;
  ivec2 b = p / u_block;
  float start = 0.16 * u_copy;
  if (hash(vec3(b.x, b.y, u_seed)) >= start) {
    for (int k = 1; k <= 10; k++) {
      int s = b.x - k;
      if (s < 0) break;
      if (hash(vec3(s, b.y, u_seed)) < start) {
        int len = 2 + int(hash(vec3(s, b.y, u_seed + 7.13)) * 9.0);
        if (k <= len) return ivec2(s * u_block + (p.x - u_block * b.x), p.y);
        break;
      }
    }
  }
  if (hash(vec3(-3.0, b.y, u_seed + 1.7)) < 0.1 * u_copy) {
    int up = u_block * (1 + int(hash(vec3(-5.0, b.y, u_seed)) * 3.0));
    return ivec2(p.x, max(0, p.y - up));
  }
  return p;
}

vec4 at(ivec2 p) {
  return texelFetch(u_src, clamp(p, ivec2(0), u_size - 1), 0);
}

void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  int s = int(floor(u_split + 0.5));
  float r = at(copySrc(p + ivec2(s, 0))).r;
  vec4 g = at(copySrc(p));
  float bl = at(copySrc(p - ivec2(s, 0))).b;
  o = vec4(r, g.g, bl, 1.0);
}
