// browser-shared: dtouch/shaders/alive (README there).
// Alive ink: the display pass for H's fractal step (camera or not) when the
// ink look is on. The veins are drawn as calligraphy: dark, sharp strokes
// with a metallic sheen, set over a molten ground.
//   - strokes are iso-contours of the grid-size fields (r trunks, g veins,
//     b threads) at RAISED thresholds (u_inkThr x compose's u_thr), so the
//     weak lace drops out and what stays is fewer, more deliberate lines.
//     Sampled bilinear and cut with a sub-pixel anti-aliased step (u_edge), so an
//     outline is a smooth curve at any grid-to-screen ratio. Where a vein's
//     field falls off toward its end the contour closes to a point: the
//     taper of a lifted brush, by construction;
//   - broad nib: a flat pen at one angle (u_nib: fixed near 35 degrees, can drift) makes a
//     stroke full width where it travels across the nib edge and a hairline
//     where it travels along it, so widths turn on direction: thick
//     diagonals, razor hairlines, angular at every bend (blackletter);
//   - tiers: primaries (trunks), thorns (veins) and hairlines (threads) are
//     cut separately, each at its own threshold, so widths come in plateaus
//     with gaps between them (modern gothic / cyber sigil): wide letter-like
//     bodies, sharp thorns spinning off them that close to needles away from
//     the primaries (u_thorn), and fine lace;
//   - chrome: the stroke's reflection posterised into four hard values
//     (u_chrome), mostly black with small hard whites;
//   - pressure: a slow value noise over the grid scales the threshold, so a
//     stroke swells and thins along its length (u_press: amount, wavelength
//     in grid px, drift);
//   - dry brush: where the field barely clears the threshold (the lifted
//     ends, the thin flanks) streaks running along the stroke break it up
//     (u_dry: amount, streak scale across, along);
//   - the body is near-black lacquer on a tube profile (normal from the
//     field's slope): a horizon-band environment reflection (bright above,
//     dark below, a hard line between) gives the chrome bevel, Fresnel
//     weighted, plus a narrow Blinn highlight from the key light (u_light)
//     that runs along each stroke. The alive picture's brightness (fronts,
//     the shuttle pulse) lifts the sheen, so the pulse travels as light
//     along the ink;
//   - the ground (u_variant): 0 lacquer on a lit field (strokes dark on the
//     palette's light), 1 molten pool (the heat of the network, a blur of
//     its fields from their mip chain, through the palette; strokes black inside their
//     own glow), 2 liquid chrome (bright chrome strokes on near-black).
//     Over the camera (V veil or full) the feed is the ground, and the
//     molten glow is screened over it;
//   - accents (exploring tips, connection light, the cast keys) stay
//     luminous through the palette, on top;
//   - two style options, both off by default (the landing never sets
//     them; off is the picture above, pixel for pixel), after the owner's
//     TikTok study (2026-10-07, @sakrmusic):
//     paper (u_paper): black ink on warm white paper, a grey smoke wash
//       where the network has mass, ink bleeding a little past each edge,
//       and ONE accent colour (the palette's hot end) laid on like a
//       second ink; over the camera the feed prints soft grey under it.
//       u_paper is 0..1 and the host fades it (alive.json `style`): a
//       white frame never cuts in, the flash floor (3 a second) holds;
//     fold (u_fold): the whole pass reads folded coordinates, so the
//       network, the camera and the accents fold together: 2 mirrors one
//       half across the centre line, 4 one quadrant across both, 6 a 60
//       degree wedge around the centre into six, at the picture's own
//       scale (reaching past the frame reads it mirrored). The source half,
//       quadrant or wedge is the one the performer is in (u_foldSide,
//       u_foldAngle: the host follows the matte's lit centroid with a
//       hysteresis and a hold, dtouch.alive.fold_side), so nobody is
//       folded out of view.
// Textures are in image space (row 0 at the top); flip once here.
uniform sampler2D u_lum;      // alive picture: r whole, g veins, b accents
uniform sampler2D u_fields;   // tonemap_fractal's fields, grid size
uniform sampler2D u_fieldsMip; // the same, mipmapped, for the molten heat
uniform sampler2D u_lut;
uniform vec3 u_thr;           // compose's outline thresholds per level
uniform vec3 u_inkThr;        // raise per level: trunks, veins, threads
uniform vec2 u_gridTexel;
uniform vec2 u_gridSize;
uniform float u_palRow;
uniform float u_palRowFrom;
uniform float u_palMix;
uniform float u_bgOn;         // 1 = the camera is the ground
uniform float u_bgLuma;       // 1 = veil (the feed through the palette), 0 = full colour
uniform float u_bgMix;        // how much of the feed shows
uniform float u_blackout;
uniform vec3 u_light;         // key light, image space (+y down)
uniform float u_variant;      // 0 lacquer, 1 molten pool, 2 liquid chrome
uniform vec3 u_press;         // pressure amount, wavelength (grid px), drift (grid px / s)
uniform vec4 u_dry;           // dry-brush amount, streak scale across, along (grid px); glint strength
uniform vec4 u_sheen;         // bump, env strength, highlight strength, highlight power
uniform vec4 u_ground;        // heat gain, heat blur (mip level), ground level, edge halo
uniform vec2 u_edge;          // outline anti-alias width (x fwidth), glint line sharpness
uniform vec4 u_nib;           // broad nib: hairline gain, full gain, angle (rad), drift (rad / s)
uniform vec4 u_thorn;        // thorns: gain near the primaries, gain far out, mip level of 'near', nib hairline gain
uniform float u_thornRim;    // thorns' white rim on the lit side
uniform vec3 u_chrome;       // chrome posterise cuts: black / dark grey / light grey / white
uniform vec2 u_body;         // primaries' body: mip level, its share
uniform vec2 u_tierBump;     // bevel of the thorns, of the hairlines (x the primaries')
uniform float u_time;
uniform vec3 u_fxBurst;
uniform vec3 u_fxWave;
uniform float u_fxAspect;
in vec2 v_uv;
layout(location = 0) out vec4 o;
uniform float u_paper;       // 0..1: how much the paper shows (the host fades it)
uniform float u_fold;        // the mirror fold: 0 off, 2, 4, 6 (u_mirror is the camera's)
uniform vec2 u_foldSide;     // source half per axis: -1 left / top, +1 right / bottom
uniform float u_foldAngle;   // the 6-fold source wedge's centre, rad (image space, -pi/2 up)
vec2 foldUv(vec2 uv) {
  if (u_fold < 1.5) return uv;
  vec2 s = vec2(u_foldSide.x < 0.0 ? -1.0 : 1.0, u_foldSide.y < 0.0 ? -1.0 : 1.0);
  if (u_fold < 3.0) return vec2(0.5 + s.x * abs(uv.x - 0.5), uv.y);
  if (u_fold < 5.0) return 0.5 + s * abs(uv - 0.5);
  vec2 asp = vec2(u_fxAspect, 1.0);
  vec2 p = (uv - 0.5) * asp;
  float w = 3.14159265 / 3.0;
  float t = mod(atan(p.y, p.x) - u_foldAngle + 0.5 * w, 2.0 * w);
  if (t > w) t = 2.0 * w - t;
  float A = u_foldAngle - 0.5 * w + t;
  // the radius stays as it is (no zoom: a pixel inside the source wedge
  // maps to itself), and a fold that reaches past the frame reads it
  // mirrored back in, so the corners show picture, not a clamp smear
  vec2 q = vec2(cos(A), sin(A)) * length(p) / asp + 0.5;
  return 1.0 - abs(mod(q, 2.0) - 1.0);
}

vec3 lut(float x) {
  x = clamp(x, 0.0, 1.0);
  return mix(texture(u_lut, vec2(x, u_palRowFrom)).rgb, texture(u_lut, vec2(x, u_palRow)).rgb, u_palMix);
}

float hash(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}
float vnoise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x),
             mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
}

// stroke fields, one per tier: each level's excess over its raised
// threshold, 1 on the outline, > 1 inside (r primaries, g thorns, b hairlines)
vec3 tiers(vec2 p) {
  vec3 F = texture(u_fields, p).rgb;
  // the primaries' body: their field widened by a share of a mip level up,
  // so they read as broad letter bodies against the raw-sharp thorns
  F.r = mix(F.r, textureLod(u_fieldsMip, p, u_body.x).r, u_body.y);
  return F / max(u_thr * u_inkThr, vec3(1e-3));
}

// one tier's cover: a sub-pixel cut of its field, broken by the dry brush
// where the field barely clears (the lifted ends)
float cut(float e, float streak) {
  float dryCut = u_dry.x * (1.0 - smoothstep(0.0, 0.5, e - 1.0)) * (streak - 0.35);
  float w = max(u_edge.x * fwidth(e), 1e-4);
  return smoothstep(1.0 - w, 1.0 + w, e - max(dryCut, 0.0));
}

// the environment a chrome surface reflects: a horizon band across the
// light's azimuth, bright sky over dark ground, a hard line between
vec3 env(vec3 r, vec2 az) {
  float s = dot(r.xy, az) * 0.9 + 0.35 * r.z - 0.32;
  float sky = smoothstep(-0.015, 0.015, s) * (0.25 + 0.75 * exp(-s * s * 60.0));
  float gnd = (1.0 - smoothstep(-0.015, 0.015, s)) * 0.05 * exp(-s * s * 20.0);
  vec3 tint = mix(vec3(1.0), lut(0.95), 0.35);
  return tint * (sky + gnd);
}

void main() {
  vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);
  uv = foldUv(uv);
  vec4 L = texture(u_lum, uv);
  vec2 gp = uv * u_gridSize;

  // Three tiers with plateaus between them, not one continuous range of
  // widths: each level is cut at its own threshold and drawn as its own
  // stroke, the wider tier over the finer one.
  //   primaries (trunks): wide broad-nib bodies, letter-like;
  //   thorns (veins): sharp spikes off the primaries, full near them and
  //     closing to a needle as they leave (the trunks' blurred mass sets it);
  //   hairlines (threads): the fine lace, thin and secondary.
  float pn = vnoise(gp / max(u_press.y, 1.0) + vec2(0.7, -0.4) * u_press.z * u_time / max(u_press.y, 1.0));
  float press = 1.0 + u_press.x * (pn - 0.5) * 2.0;

  vec2 tx = vec2(u_gridTexel.x, 0.0), ty = vec2(0.0, u_gridTexel.y);
  vec3 Et = tiers(uv);
  // the slope over a texel and a half: smooth enough for a clean highlight
  vec3 Gx = (tiers(uv + 1.5 * tx) - tiers(uv - 1.5 * tx)) / 3.0;
  vec3 Gy = (tiers(uv + 1.5 * ty) - tiers(uv - 1.5 * ty)) / 3.0;
  vec2 g0 = vec2(Gx.x, Gy.x), g1 = vec2(Gx.y, Gy.y), g2 = vec2(Gx.z, Gy.z);
  vec2 N0 = length(g0) > 1e-5 ? normalize(g0) : vec2(0.0, 1.0);
  vec2 N1 = length(g1) > 1e-5 ? normalize(g1) : vec2(0.0, 1.0);

  // broad nib: a flat pen held at one angle makes a stroke full width when
  // it travels across the nib edge and a hairline when it travels along it,
  // so the width turns on the stroke's direction, angular at every bend.
  // Pressure swells only the primaries.
  vec2 nibDir = vec2(cos(u_nib.z + u_nib.w * u_time), sin(u_nib.z + u_nib.w * u_time));
  float e0 = Et.x * press * mix(u_nib.x, u_nib.y, smoothstep(0.1, 0.9, abs(dot(N0, nibDir))));
  // thorns: slim crescents off the spine. Width tapers with distance from
  // the primaries (their blurred mass): full at the root, a hairline floor
  // far out, so a thorn runs long and ends in a needle instead of stopping
  // short. The same pen, harder, makes them angular and slim, never knobby.
  vec3 T = max(u_thr * u_inkThr, vec3(1e-3));
  float near = smoothstep(0.15, 0.9, textureLod(u_fieldsMip, uv, u_thorn.z).r / T.x);
  float e1 = Et.y * mix(u_thorn.y, u_thorn.x, near)
           * mix(u_thorn.w, 1.0, smoothstep(0.2, 0.8, abs(dot(N1, nibDir))));
  float e2 = Et.z;

  vec2 T0 = vec2(-N0.y, N0.x);
  float streak = vnoise(vec2(dot(gp, N0) / max(u_dry.y, 0.1), dot(gp, T0) / max(u_dry.z, 0.1)));
  float c0 = cut(e0, streak);
  float c1 = cut(e1, streak);
  float c2 = cut(e2, 0.0);
  // no specks: a fleck that clears its threshold alone, with nothing of
  // substance around it, is dust, not a stroke (the fields a mip level up)
  vec3 Eb = textureLod(u_fieldsMip, uv, 1.5).rgb / T;
  c1 *= smoothstep(0.5, 0.8, max(Eb.x, Eb.y));
  c2 *= smoothstep(0.55, 0.85, max(Eb.x, max(Eb.y, Eb.z)));
  float cover = max(c0, max(c1, c2));
  // the surface of the topmost tier at this pixel
  vec2 g = mix(mix(g2 * u_tierBump.y, g1 * u_tierBump.x, c1), g0 * press, c0);
  float e = max(e0, max(e1, e2));

  // the tube: height rises from the outline to the crest
  vec3 n = normalize(vec3(-g * u_sheen.x, 1.0));
  vec3 V = vec3(0.0, 0.0, 1.0);
  vec3 r = reflect(-V, n);
  vec2 az = length(u_light.xy) > 1e-3 ? normalize(-u_light.xy) : vec2(0.0, -1.0);
  float fres = 0.04 + 0.96 * pow(1.0 - clamp(n.z, 0.0, 1.0), 5.0);
  vec3 H = normalize(normalize(u_light) + V);
  float hl = pow(max(dot(n, H), 0.0), u_sheen.w);
  float lift = 0.55 + 0.9 * clamp(L.r, 0.0, 1.0);

  // heat: the network's mass, blurred (two mip levels of the fields, so
  // a soft pool with no ring echoes)
  float l0 = u_ground.y;
  vec3 wv = vec3(0.6, 0.3, 0.1) / max(u_thr, vec3(1e-3));
  float heat = 0.55 * dot(textureLod(u_fieldsMip, uv, l0).rgb, wv)
             + 0.45 * dot(textureLod(u_fieldsMip, uv, l0 + 1.5).rgb, wv);
  heat = clamp(heat * u_ground.x, 0.0, 1.0);
  float halo = smoothstep(0.55, 1.0, e) * (1.0 - cover);

  // the ground
  vec3 ground;
  if (u_variant < 0.5) {
    ground = lut(u_ground.z + (0.92 - u_ground.z) * heat);
    ground *= 1.0 - u_ground.w * halo;             // a little ink bleeds past the edge
  } else if (u_variant < 1.5) {
    ground = lut(heat * (0.25 + u_ground.z)) + u_ground.w * halo * lut(0.9);
  } else {
    ground = lut(0.06 + 0.12 * heat) * 0.6;
  }
  if (u_bgOn > 0.5) {
    vec3 video = texture(u_video, vuv(uv)).rgb;
    float vl = dot(video, vec3(0.299, 0.587, 0.114));
    vec3 base = mix(video, lut(vl), u_bgLuma) * mix(1.0, u_bgMix, u_bgLuma);
    vec3 glow = u_variant > 0.5 && u_variant < 1.5 ? lut(heat * 0.8) * heat : vec3(0.0);
    ground = 1.0 - (1.0 - base * (1.0 - u_ground.w * 0.6 * halo)) * (1.0 - glow);
  }

  // paper (u_paper, 0..1): its own ground, ink and accent, cross-faded
  // over the ink above at the end, so the host's fade is a fade (and 0 is
  // the picture above untouched)
  vec3 groundP = ground;
  if (u_paper > 0.0) {
    // warm white, a grey smoke wash where the network has mass, a little
    // ink bleeding past each edge; over the camera the feed shows as a soft
    // grey print under the ink
    groundP = vec3(0.95, 0.94, 0.915) * (1.0 - 0.3 * heat) * (1.0 - 0.6 * halo);
    if (u_bgOn > 0.5) {
      float pl = dot(texture(u_video, vuv(uv)).rgb, vec3(0.299, 0.587, 0.114));
      groundP *= mix(1.0, 0.35 + 0.65 * pl, u_bgLuma > 0.5 ? 0.75 : 0.9);
    }
  }
  // the stroke body
  vec3 ink;
  vec3 E = env(r, az);
  if (u_variant < 1.5) {
    // one hard glint line along each stroke: where its reflection crosses
    // the horizon (the side away from the light, orbiting with it)
    float sg = dot(r.xy, az) * 0.9 + 0.35 * r.z - 0.32;
    float glint = exp(-sg * sg * u_edge.y) * smoothstep(0.03, 0.12, length(g) * u_sheen.x);
    ink = vec3(0.012, 0.011, 0.014) + E * u_sheen.y * fres * lift
        + (hl * u_sheen.z + glint * u_dry.w) * lift * mix(vec3(1.0), lut(1.0), 0.3);
    // chrome is hard value jumps, not gradients: the reflection posterised
    // into black, dark grey, light grey and white (u_chrome: the three cuts),
    // each cut anti-aliased over a pixel
    float il = dot(ink, vec3(0.299, 0.587, 0.114));
    float aa = fwidth(il) + 1e-3;
    float q = 0.16 * smoothstep(u_chrome.x - aa, u_chrome.x + aa, il)
            + 0.39 * smoothstep(u_chrome.y - aa, u_chrome.y + aa, il)
            + 0.45 * smoothstep(u_chrome.z - aa, u_chrome.z + aa, il);
    ink = max(ink * (q / max(il, 1e-3)), vec3(0.012, 0.011, 0.014));
  } else {
    // liquid chrome: all reflection, dark only at the outline
    float edge = smoothstep(1.0, 1.35, e);
    ink = (0.08 + E * (0.6 + 0.6 * fres)) * lift * mix(0.25, 1.0, edge) + hl * u_sheen.z * lift;
  }
  // thorns catch a hard white rim on the side facing the light, so a slim
  // spike reads as a chrome needle against the black bodies
  float w1 = max(fwidth(e1), 1e-4);
  float rim = smoothstep(1.0 - w1, 1.0 + w1, e1) - smoothstep(1.0 + 2.0 * w1, 1.0 + 3.5 * w1, e1);
  float lit = smoothstep(0.1, 0.4, dot(-N1, az));
  ink = mix(ink, vec3(0.93, 0.94, 0.96) * lift, clamp(rim * lit * c1 * (1.0 - c0), 0.0, 1.0) * u_thornRim);
  vec3 inkP = ink;
  if (u_paper > 0.0) {
    // wet black ink: solid, with only a faint grey sheen where the chrome
    // would have caught the light (no white rims on white paper)
    float il = dot(ink, vec3(0.299, 0.587, 0.114));
    inkP = vec3(0.018, 0.017, 0.022) + vec3(0.16) * smoothstep(0.35, 0.95, il);
  }
  vec3 col = mix(ground, ink, cover);
  vec3 colP = mix(groundP, inkP, cover);

  // luminous accents
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
  if (u_paper > 0.0) {
    // one accent colour on paper: the palette's hot end, saturated and
    // laid on like a second ink (a screen blend vanishes on white)
    vec3 ac = lut(0.8);
    float am = dot(ac, vec3(0.333));
    ac = clamp(am + (ac - am) * 1.8, 0.0, 1.0) * 0.85;
    colP = mix(colP, ac, clamp(acc, 0.0, 1.0) * 0.9);
    col = mix(col, colP, clamp(u_paper, 0.0, 1.0));
  }
  col *= 1.0 - u_blackout;
  o = vec4(clamp(col, 0.0, 1.0), 1.0);
}
