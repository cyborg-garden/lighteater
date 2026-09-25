// Physarum tonemap statistics — a stride-u_stride subsample of (trail, laid).
//
// Drawn with fullscreen.vert into a small float target (ceil(gw/stride) x
// ceil(gh/stride)); the host reads it back and takes the 95th percentile of
// .r (trail normalization) and the mean of .g (deposit normalization).

uniform sampler2D u_trail;
uniform sampler2D u_laid;
uniform int u_stride;
uniform ivec2 u_grid;

layout(location = 0) out vec4 f_color;

void main() {
    ivec2 c = ivec2(gl_FragCoord.xy) * u_stride;
    c = min(c, u_grid - 1);
    // the picture is the whole organism: sum the species channels
    vec3 t = texelFetch(u_trail, c, 0).rgb;
    vec3 l = texelFetch(u_laid, c, 0).rgb;
    f_color = vec4(t.r + t.g + t.b, l.r + l.g + l.b, 0.0, 1.0);
}
