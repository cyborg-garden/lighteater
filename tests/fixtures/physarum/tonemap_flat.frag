// Physarum picture — the trail tonemapped to [0,1] luminance.
//
//   x   = trail / norm + grain * laid / gnorm
//   lum = 1 - exp(-exposure * x)
//
// norm is the trail's 95th percentile and gnorm = 4 * mean(laid) (both from
// stats.frag); `grain` mixes this frame's raw deposits over the smooth trail.
// Drawn with fullscreen.vert; the host colorizes lum through a palette LUT.

uniform sampler2D u_trail;
uniform sampler2D u_laid;
uniform float u_inv_norm;
uniform float u_grain;
uniform float u_inv_gnorm;
uniform float u_exposure;

layout(location = 0) out vec4 f_color;

void main() {
    ivec2 c = ivec2(gl_FragCoord.xy);
    vec3 tr = texelFetch(u_trail, c, 0).rgb;
    vec3 la = texelFetch(u_laid, c, 0).rgb;
    float x = (tr.r + tr.g + tr.b) * u_inv_norm
            + u_grain * (la.r + la.g + la.b) * u_inv_gnorm;
    f_color = vec4(vec3(1.0 - exp(-u_exposure * x)), 1.0);
}
