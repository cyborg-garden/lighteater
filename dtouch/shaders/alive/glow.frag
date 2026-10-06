// browser-shared: dtouch/shaders/alive (README there).
// Alive connection glow, grid resolution, ping-pong: a travelling light,
// not a flash. An excitable medium on the trail: a connection event (the
// events pass) lights the texel where it fired; each frame a lit texel
// lights its trail neighbours a little dimmer (u_step, dimmer still on
// settled trail than on young, so the light runs along the new link and
// only licks into the vein it met) and then goes refractory for u_refr
// seconds while it fades (u_fade per frame). Refractory texels cannot be
// relit, so the light moves on as a short bright front and never echoes.
//   r: light, g: refractory seconds left
uniform sampler2D u_prev;
uniform sampler2D u_life;    // grid: r = age (s), < 0 absent
uniform sampler2D u_events;  // blocks: gb = where, a = fired this frame
uniform ivec2 u_grid;
uniform int u_cell;
uniform float u_dt;
uniform vec2 u_step;         // light carried per texel: (young trail, settled trail)
uniform float u_young;       // seconds under which trail is young
uniform float u_fade;        // per-frame fade of a spent texel
uniform float u_refr;        // refractory seconds
uniform sampler2D u_scene;   // matte res: a = fresh-motion skirt
uniform float u_calm;        // share of connections away from motion that light
uniform float u_time;
uniform float u_reset;
layout(location = 0) out vec4 o;

void main() {
  ivec2 c = ivec2(gl_FragCoord.xy);
  vec4 prev = texelFetch(u_prev, c, 0);
  float g = prev.r, r = max(prev.g - u_dt, 0.0);
  if (u_reset > 0.5) { o = vec4(0.0); return; }
  vec4 ev = texelFetch(u_events, c / u_cell, 0);
  vec2 at = vec2((c / u_cell) * u_cell) + ev.gb * float(u_cell);
  if (ev.a > 0.5 && distance(vec2(c) + 0.5, at) < 2.0) {
    // full light where the mold is hunting (inside the fresh-motion skirt);
    // elsewhere only one connection in u_calm lights at all, and dimmer,
    // so the quiet network does not sparkle
    float near = smoothstep(0.05, 0.3, texture(u_scene, (vec2(c) + 0.5) / vec2(u_grid)).a);
    vec2 blk = vec2(c / u_cell);
    float pick = fract(sin(dot(blk, vec2(12.9898, 78.233)) + floor(u_time / 4.0) * 3.7) * 43758.5453);
    float lit = max(near, pick < u_calm ? 0.6 : 0.0);
    if (lit > 0.0) { o = vec4(lit, u_refr, 0.0, 1.0); return; }
  }
  float age = texelFetch(u_life, c, 0).r;
  if (r <= 0.0 && age >= 0.0) {
    // the light runs along LINES: a texel inside a blob (most of its ring
    // standing) does not carry it, so it never floods a patch into a square
    float n = 0.0, around = 0.0;
    for (int j = -1; j <= 1; j++)
      for (int i = -1; i <= 1; i++) {
        if (i == 0 && j == 0) continue;
        ivec2 q = clamp(c + ivec2(i, j), ivec2(0), u_grid - 1);
        float w = (i != 0 && j != 0) ? 0.985 : 1.0;
        n = max(n, w * texelFetch(u_prev, q, 0).r);
        around += step(0.0, texelFetch(u_life, q, 0).r);
      }
    float k = (age < u_young ? u_step.x : u_step.y) * (around > 5.5 ? 0.6 : 1.0);
    if (n * k > 0.12) { o = vec4(n * k, u_refr, 0.0, 1.0); return; }
  }
  o = vec4(g * u_fade, r, 0.0, 1.0);
}
