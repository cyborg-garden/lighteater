// browser-shared: dtouch/shaders/alive (README there).
// Alive carved composite: the display pass for H's fractal step when the
// background shows the camera (V veil or full). The veins are solid
// material set INTO the picture, not a film over it:
//   - inside the vein outline the body is opaque: palette colour from the
//     vein's own brightness, lightly tinted by the feed's hue underneath
//     (u_body.y), never the feed at reduced opacity;
//   - the carved lighting sits on that body: the wall that faces the key
//     light gets a crisp lit rim, the other wall a shadow, shallower where
//     the scene depth turns away (the facing term);
//   - a thin dark contact line rings every vein and a short shadow falls
//     away from the light, so the material sits in the surface;
//   - the feed shows around the veins, bent slightly by their slope and by
//     the scene depth, and dimly on the floor of the widest grooves;
//   - the finer levels are cut at raised thresholds (u_thrUp, on the
//     grid-size fields), so over the camera a dense network loses its
//     faint lace and twig spikes with a crisp edge and reads as drawn;
//   - only the accents (exploring tips, connection light, the cast keys)
//     stay luminous, through the palette.
// Textures are in image space (row 0 at the top); flip once here.
uniform sampler2D u_lum;      // alive picture: r whole, g veins, b accents
uniform sampler2D u_fields;   // tonemap_fractal's fields, grid size: r trunks, g veins, b threads
uniform vec3 u_thrUp;         // per-level outline threshold over the camera (trunks as drawn, finer raised)
uniform sampler2D u_lut;
uniform sampler2D u_scene;    // matte res: b = depth
uniform float u_palRow;
uniform float u_palRowFrom;
uniform float u_palMix;
uniform float u_bgLuma;       // 1 = veil (the feed through the palette), 0 = full colour
uniform float u_bgMix;        // how much of the feed shows (veil dims it)
uniform float u_blackout;
uniform vec2 u_lumTexel;
uniform vec2 u_sceneTexel;
uniform vec3 u_light;         // key light, image space (+y down)
uniform vec4 u_carve;         // groove floor, rim strength, refraction (lum texels), depth refraction (scene texels)
uniform vec2 u_bump;          // (vein normal scale, depth steepness gain)
uniform vec4 u_body;          // body brightness floor, feed hue tint, contact line strength, drop shadow strength
uniform vec2 u_contact;       // contact line radius, drop shadow length (lum texels)
uniform vec3 u_fxBurst;
uniform vec3 u_fxWave;
uniform float u_fxAspect;
in vec2 v_uv;
layout(location = 0) out vec4 o;

vec3 lut(float x) {
  return mix(texture(u_lut, vec2(x, u_palRowFrom)).rgb, texture(u_lut, vec2(x, u_palRow)).rgb, u_palMix);
}

float aastep(float t, float f) {
  float w = max(0.7 * fwidth(f), 1e-4);
  return smoothstep(t - w, t + w, f);
}

// the vein's coverage at p: its drawn outline, where the level that drew
// it clears its raised threshold (a trunk always does)
float cover(vec2 p) {
  vec3 F = texture(u_fields, p).rgb;
  float keep = max(aastep(u_thrUp.x, F.r), max(aastep(u_thrUp.y, F.g), aastep(u_thrUp.z, F.b)));
  return smoothstep(0.05, 0.14, texture(u_lum, p).g) * keep;
}

void main() {
  vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);
  vec4 L = texture(u_lum, uv);
  vec2 tx = vec2(u_lumTexel.x, 0.0), ty = vec2(0.0, u_lumTexel.y);
  // the veins' slope, over a texel and a half (crisp, not a blur)
  vec2 gV = 0.5 * vec2(texture(u_lum, uv + 1.5 * tx).g - texture(u_lum, uv - 1.5 * tx).g,
                       texture(u_lum, uv + 1.5 * ty).g - texture(u_lum, uv - 1.5 * ty).g);
  vec2 sx = vec2(u_sceneTexel.x, 0.0), sy = vec2(0.0, u_sceneTexel.y);
  vec2 gH = 0.5 * vec2(texture(u_scene, uv + sx).b - texture(u_scene, uv - sx).b,
                       texture(u_scene, uv + sy).b - texture(u_scene, uv - sy).b);
  float steep = length(gH) * u_bump.y;
  float facing = inversesqrt(1.0 + steep * steep);
  // the feed, bent through the grooves and over the depth
  vec2 off = gV * u_carve.z * u_lumTexel + gH * u_carve.w * u_sceneTexel;
  vec3 video = texture(u_video, vuv(uv + off)).rgb;
  float vl = dot(video, vec3(0.299, 0.587, 0.114));
  vec3 base = mix(video, lut(vl), u_bgLuma) * mix(1.0, u_bgMix, u_bgLuma);

  float vein = L.g;
  float body = cover(uv);
  // contact: a thin dark line just outside the outline (a ring of taps at
  // u_contact.x), and a short shadow cast away from the key light
  float ring = 0.0;
  for (int i = 0; i < 8; i++) {
    float a = 0.7853982 * float(i);
    ring = max(ring, cover(uv + u_contact.x * vec2(cos(a), sin(a)) * u_lumTexel));
  }
  vec2 ld = u_light.xy / max(length(u_light.xy), 1e-3);
  float drop = max(cover(uv + ld * u_contact.y * u_lumTexel),
                   cover(uv + ld * 0.5 * u_contact.y * u_lumTexel));
  float outside = 1.0 - body;
  // only solid bodies cast them: a faint hairline gets no dark halo
  ring = smoothstep(0.4, 0.9, ring);
  drop = smoothstep(0.4, 0.9, drop);
  vec3 bg = base * (1.0 - outside * max(u_body.z * ring, u_body.w * drop * facing));

  // the body: palette colour from the vein's own brightness, hue-tinted by
  // the feed under it, with the carved walls lit and shadowed on it
  vec3 n = normalize(vec3(gV * u_bump.x, 1.0));   // a groove: the wall normals point inward
  float lit = dot(n, u_light);
  float wall = clamp((lit - u_light.z) * 6.0, -1.0, 1.0) * smoothstep(0.02, 0.2, length(gV) * u_bump.x);
  float shade = u_body.x + (0.8 - u_body.x) * clamp(vein, 0.0, 1.0);
  vec3 pal = lut(shade);
  vec3 hue = video / max(max(video.r, max(video.g, video.b)), 0.05);
  vec3 mat = pal * mix(vec3(1.0), 0.35 + 0.65 * hue, u_body.y * (1.0 - u_bgLuma));
  mat *= 1.0 - 0.65 * max(-wall, 0.0);
  mat += u_carve.y * max(wall, 0.0) * facing * (0.4 + 0.6 * lut(1.0));
  // the floor of the widest grooves: the feed, dim, where the slope flattens
  // along a bright trunk's middle
  float floorK = u_carve.x * smoothstep(0.75, 0.95, vein) * (1.0 - smoothstep(0.0, 0.03, length(gV)));
  mat = mix(mat, base * 0.45, floorK);
  vec3 col = mix(bg, mat, body);

  // luminous accents: exploring tips, connection light, the cast keys
  float acc = L.b;
  vec2 asp = vec2(u_fxAspect, 1.0);
  float bt = u_fxBurst.z / 0.6;
  if (bt < 1.0) {
    float d = length((uv - u_fxBurst.xy) * asp);
    float sg = 0.09 + 0.10 * bt;
    acc += 2.6 * (1.0 - bt) * (1.0 - bt) * exp(-d * d / (2.0 * sg * sg));
  }
  float wt = u_fxWave.z / 0.9;
  if (wt < 1.0) {
    float d = length((uv - u_fxWave.xy) * asp);
    float q = (d - (0.02 + 0.45 * wt)) / 0.035;
    acc += 1.8 * (1.0 - wt) * exp(-q * q);
  }
  acc = min(acc, 1.0);
  col = 1.0 - (1.0 - col) * (1.0 - lut(acc) * acc);
  col *= 1.0 - u_blackout;
  o = vec4(clamp(col, 0.0, 1.0), 1.0);
}
