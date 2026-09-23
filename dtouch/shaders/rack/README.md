# SIGNAL rack shaders — GPU port of dtouch.circuit_bent

Fragment passes for `dtouch.rack_gl.SignalRackGL` and `PhysarumOutGL`: the
CircuitBent rack (scan-drift, chroma, glitch, crush, dither, scanlines) plus
the physarum colorize/upscale/video-composite stages, run on the simulation's
own moderngl context so the trail never round-trips through numpy until the
one final uint8 readback.

Unlike `../physarum/`, these files are **not** shared with the browser port —
they carry their own `#version 330 core` line and are loaded verbatim by
`dtouch/rack_gl.py`. The behavioral contract is `dtouch/circuit_bent.py`:
every pass mirrors its numpy counterpart (same stage order, same rounding
conventions — np.round == roundEven, the final `*255` truncation == floor,
cv2 INTER_LINEAR == the manual bilinear taps, cv2 INTER_NEAREST ==
`floor(x * src/dst)`), and `tests/test_rack_gl.py` holds the two backends
together with per-stage parity tests.

| file               | pass                                                     |
|--------------------|----------------------------------------------------------|
| `fullscreen.vert`  | full-screen triangle for every pass                      |
| `colorize.frag`    | luminance -> palette LUT / video-lit RGB (grid res)      |
| `upscale.frag`     | bilinear grid -> output res + video-bg screen blend      |
| `driftchroma.frag` | per-row scan drift + R/B channel shifts (one pass)       |
| `blit.frag`        | offset texel copy (glitch tile capture + replay)         |
| `crush.frag`       | hard bit-depth quantisation (roundEven == np.round)      |
| `downsample.frag`  | bilinear full res -> dither working res                  |
| `compose.frag`     | nearest-neighbour upscale + scanlines + final u8 encode  |

The rack's ordered-dither pass, `dither.frag`, lives in the browser-shared unit
`../dither/` (loaded by `rack_gl.load_shared_shader`); see that README.
