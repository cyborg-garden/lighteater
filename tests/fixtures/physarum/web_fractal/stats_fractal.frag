// DEMO, WEB ONLY: moves to the desktop shared unit (dtouch/shaders/physarum)
// if kept. A modified copy of vendor/physarum/stats.frag for the fractal
// veins demo, loaded only when the fractal amount is above 0.
//
// Same stride subsample, plus the two finer level channels in .b and .a so
// the host can take each level's own bright end (trunks = r - b - a). The
// stock .r (sum) and .g (laid sum) are unchanged, so the stock tonemap
// normalisation keeps working through the switch.

uniform sampler2D u_trail;
uniform sampler2D u_laid;
uniform int u_stride;
uniform ivec2 u_grid;

layout(location = 0) out vec4 f_color;

void main() {
    ivec2 c = ivec2(gl_FragCoord.xy) * u_stride;
    c = min(c, u_grid - 1);
    vec3 t = texelFetch(u_trail, c, 0).rgb;
    vec3 l = texelFetch(u_laid, c, 0).rgb;
    f_color = vec4(t.r + t.g + t.b, l.r + l.g + l.b, t.g, t.b);
}
