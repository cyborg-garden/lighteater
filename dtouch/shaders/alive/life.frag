// browser-shared: dtouch/shaders/alive (README there).
// Alive life pass, grid resolution, ping-pong, run after the deposit.
//   r: the AGE of the drawn line here in seconds, -1 where there is none.
//      A texel is line once any level's field (last frame's
//      tonemap_fractal output, the very thing compose outlines) passes
//      u_onOff.x times that level's outline threshold, and stops being one
//      under u_onOff.y times it (hysteresis, so a flickering skirt is not
//      reborn every frame). Age 0 is a texel that became line this frame:
//      the fronts, and the raw material of connection events.
//   gba: a short memory of what each level laid here (an EMA of the
//      deposits, u_fluxKeep per frame): how much a trail is still in use,
//      which the alive blur prunes on.
uniform sampler2D u_fields;  // tonemap_fractal's fields (last frame), grid
uniform sampler2D u_laid;
uniform sampler2D u_prev;
uniform vec3 u_thr;        // compose's outline threshold per level
uniform vec2 u_onOff;      // presence thresholds, x that threshold
uniform float u_dt;        // seconds this frame
uniform float u_fluxKeep;  // per-frame keep of the deposit memory
uniform float u_reset;
layout(location = 0) out vec4 o;
void main() {
  ivec2 c = ivec2(gl_FragCoord.xy);
  vec3 fv = texelFetch(u_fields, c, 0).rgb / u_thr;
  float s = max(fv.r, max(fv.g, fv.b));
  vec4 prev = texelFetch(u_prev, c, 0);
  float age = prev.r;
  if (u_reset > 0.5) age = s > u_onOff.x ? 10.0 : -1.0;
  else if (age < 0.0) age = s > u_onOff.x ? 0.0 : -1.0;
  else age = s < u_onOff.y ? -1.0 : min(age + u_dt, 30.0);
  vec3 laid = texelFetch(u_laid, c, 0).rgb;
  vec3 flux = u_reset > 0.5 ? laid : prev.gba * u_fluxKeep + laid * (1.0 - u_fluxKeep);
  o = vec4(age, flux);
}
