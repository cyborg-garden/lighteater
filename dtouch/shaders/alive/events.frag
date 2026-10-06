// browser-shared: dtouch/shaders/alive (README there).
// Alive connection events, one texel per u_cell x u_cell block of the grid,
// ping-pong. A connection is a texel that became line THIS frame (age 0 in
// u_life) and bridges two separate lines (see bridges()). One event per
// block per u_cool seconds.
//   r: seconds since this block's last event (capped)
//   gb: where it fired, as a share of the block
//   a: 1 on the frame it fired
uniform sampler2D u_life;   // grid: r = age (s), < 0 absent
uniform sampler2D u_prev;
uniform ivec2 u_grid;
uniform int u_cell;
uniform float u_dt;
uniform float u_cool;
uniform float u_young;
uniform float u_settled;
uniform float u_reach;    // how far back the thread must reach, texels
uniform float u_minAge;   // and at least this old there, s
uniform float u_reset;
layout(location = 0) out vec4 o;

float ageAt(ivec2 q) {
  if (q.x < 0 || q.y < 0 || q.x >= u_grid.x || q.y >= u_grid.y) return -1.0;
  return texelFetch(u_life, q, 0).r;
}

// A bridge, by the ring test of digital topology: walk a ring of radius 3
// around the newborn texel and split it into runs of standing line
// (separated by at least two samples of empty ground). A tip growing into
// the dark sees one run (behind it), a vein drifting sideways one run (its
// own body), a hole healing inside a patch one run all the way round. A
// CONNECTION is a thin young run (a thread under u_young seconds old, at
// most five samples wide) meeting a separate settled run (a vein standing
// for u_settled seconds or more) on the far side: something grew across
// and closed onto the network.
bool bridges(ivec2 c) {
  const int N = 20;
  // rotate the walk to start on empty ground, so no run wraps the seam
  int s0 = -1;
  for (int k = 0; k < N; k++) {
    float a = 6.2831853 * float(k) / float(N);
    if (ageAt(c + ivec2(round(3.0 * vec2(cos(a), sin(a))))) < 0.1) { s0 = k; break; }
  }
  if (s0 < 0) return false;
  bool inRun = false;
  int len = 0, gap = 2, nY = 0, nS = 0;
  float lo = 1e9, hi = -1.0;
  vec2 dir = vec2(0.0), yDir[3], sDir[3];
  for (int k = 1; k <= N; k++) {
    int sk = s0 + k;                     // never negative: s0 >= 0, k >= 1
    float a = 6.2831853 * float(sk - (sk / N) * N) / float(N);
    vec2 u = vec2(cos(a), sin(a));
    float ag = ageAt(c + ivec2(round(3.0 * u)));
    bool on = ag >= 0.1;
    if (on) {
      if (!inRun && gap >= 2) { len = 0; lo = 1e9; hi = -1.0; dir = vec2(0.0); }
      inRun = true;
      gap = 0;
      len++;
      lo = min(lo, ag);
      hi = max(hi, ag);
      dir += u;
    } else {
      gap++;
    }
    // a run closes on its second empty sample, or at the end of the walk
    if (inRun && (gap == 2 || k == N)) {
      inRun = false;
      vec2 d = normalize(dir + 1e-6);
      if (len <= 5 && hi < u_young && nY < 3) yDir[nY++] = d;
      if (lo >= u_settled && nS < 3) sDir[nS++] = d;
    }
  }
  // the thread and the vein it met lie on opposite sides of the new texel
  // (a sprout still near the trunk it grew from has both on the same side)
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      if (i < nY && j < nS && dot(yDir[i], sDir[j]) < -0.5) {
        // and the thread is a real one: still there, still young, twice as
        // far back (a one-frame spur is not a hunt)
        float back = ageAt(c + ivec2(round(u_reach * yDir[i])));
        if (back >= u_minAge && back < u_young) return true;
      }
  return false;
}

void main() {
  ivec2 cc = ivec2(gl_FragCoord.xy);
  vec4 prev = texelFetch(u_prev, cc, 0);
  float since = u_reset > 0.5 ? u_cool : min(prev.r + u_dt, 60.0);
  vec2 at = prev.gb;
  float fired = 0.0;
  if (since >= u_cool) {
    ivec2 base = cc * u_cell;
    for (int j = 0; j < 16; j++) {
      if (j >= u_cell || fired > 0.5) break;
      for (int i = 0; i < 16; i++) {
        if (i >= u_cell) break;
        ivec2 c = base + ivec2(i, j);
        float a = ageAt(c);
        if (a < 0.0 || a > 0.5 * u_dt) continue;
        if (bridges(c)) {
          fired = 1.0;
          since = 0.0;
          at = (vec2(i, j) + 0.5) / float(u_cell);
          break;
        }
      }
    }
  }
  o = vec4(since, at, fired);
}
