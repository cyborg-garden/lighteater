// browser-shared: dtouch/shaders/alive (README there).
// Alive scene pass, matte resolution, ping-pong. What the mold senses of the
// scene beyond light, for H's fractal step only:
//   r: fresh motion, a fast-decaying food map. The motion pass's
//      instantaneous deadbanded frame difference (u_motion .b), blurred a
//      little, held by a max and let go with a half-life of about half a
//      second (u_freshKeep per frame). Camera noise is already under the
//      motion pass's deadband.
//   g: distance inside the matte, in matte texels, by one relaxation step
//      a frame (converges in u_domeR frames, and so moves calmly)
//   b: the depth the veins wrap around. A rounded dome over the matte's
//      distance, mixed with a pseudo-depth from heavily blurred luminance
//      (bright reads as near), eased in over time so the field breathes
//      rather than flickers
//   a: the fresh skirt: fresh motion spread outward a texel a frame and let
//      go a little slower, so a moving hand has a slope threads can feel
//      from further away
uniform sampler2D u_motion;   // r history, g luma, b instantaneous motion
uniform sampler2D u_matte;    // r = matte (softened)
uniform sampler2D u_prev;     // last frame's scene
uniform vec2 u_texel;         // one matte texel in uv
uniform float u_freshKeep;    // per-frame keep of the fresh map
uniform float u_skirtKeep;    // per-frame keep of the skirt
uniform float u_freshIn;      // motion -> fresh gain
uniform float u_domeR;        // dome radius, matte texels
uniform float u_domeW;        // share of depth that is the matte dome
uniform float u_ease;         // per-frame ease of the depth toward this frame's
uniform float u_reset;        // 1 = start over (new size, new source)
in vec2 v_uv;
layout(location = 0) out vec4 o;

float lumaAt(vec2 uv) { return texture(u_motion, uv).g; }

void main() {
  vec4 prev = texture(u_prev, v_uv);
  vec2 dx = vec2(u_texel.x, 0.0), dy = vec2(0.0, u_texel.y);
  // fresh: 3x3 blur of the instantaneous motion at 1.5 texels
  float m = 0.0;
  for (int j = -1; j <= 1; j++)
    for (int i = -1; i <= 1; i++)
      m += texture(u_motion, v_uv + 1.5 * (float(i) * dx + float(j) * dy)).b;
  m *= 1.0 / 9.0;
  float fresh = max(prev.r * u_freshKeep, clamp(u_freshIn * m, 0.0, 1.0));
  // skirt: the neighbours' skirt, held a little lower than the centre, so
  // it creeps outward and slopes down away from the motion
  float sk = 0.25 * (texture(u_prev, v_uv + dx).a + texture(u_prev, v_uv - dx).a
                   + texture(u_prev, v_uv + dy).a + texture(u_prev, v_uv - dy).a);
  float skirt = max(max(prev.a, 0.96 * sk) * u_skirtKeep, fresh);
  // matte distance, one relaxation step a frame
  float inside = step(0.5, texture(u_matte, v_uv).r);
  float nb = min(min(texture(u_prev, v_uv + dx).g, texture(u_prev, v_uv - dx).g),
                 min(texture(u_prev, v_uv + dy).g, texture(u_prev, v_uv - dy).g));
  float d = inside * min(min(prev.g + 0.5, nb + 1.0), 1.5 * u_domeR);
  float r = 1.0 - min(d / u_domeR, 1.0);
  float dome = sqrt(max(1.0 - r * r, 0.0));
  // pseudo-depth: luminance under a wide gaussian (7 x 7 bilinear taps two
  // texels apart), so only the big shapes of light survive
  float blur = 0.0, wsum = 0.0;
  for (int j = -3; j <= 3; j++)
    for (int i = -3; i <= 3; i++) {
      float w = exp(-0.18 * float(i * i + j * j));
      blur += w * lumaAt(v_uv + 2.0 * (float(i) * dx + float(j) * dy));
      wsum += w;
    }
  float pseudo = blur / wsum;
  float target = mix(pseudo, max(0.5 * pseudo, dome), u_domeW);
  float depth = u_reset > 0.5 ? target : mix(prev.b, target, u_ease);
  if (u_reset > 0.5) { fresh = 0.0; skirt = 0.0; d = 0.0; }
  o = vec4(fresh, d, depth, skirt);
}
