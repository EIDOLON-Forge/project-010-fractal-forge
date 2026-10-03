#version 330 core
in vec2 uv;
out vec4 fragColor;

uniform vec2 u_center;
uniform float u_scale;
uniform int u_iterations;
uniform int u_mode;              // 0 Mandelbrot, 1 Julia
uniform vec2 u_julia_c;
uniform vec3 u_palette[6];
uniform float u_palette_shift;
uniform float u_time;

vec3 palette(float t) {
    t = fract(t);
    float p = t * 5.0;
    int i = int(floor(p));
    float f = fract(p);
    i = clamp(i, 0, 4);
    return mix(u_palette[i], u_palette[i+1], f);
}

float escapeTime(vec2 z, vec2 c) {
    for (int i = 0; i < 1200; ++i) {
        if (i >= u_iterations) break;
        float r2 = dot(z,z);
        if (r2 > 4.0) {
            float smooth_value = float(i) + 1.0 - log2(log2(sqrt(r2)));
            return smooth_value;
        }
        z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;
    }
    return -1.0;
}

void main() {
    float aspect = 1200.0 / 800.0;
    vec2 p = uv - 0.5;
    vec2 z = u_center + vec2(p.x * u_scale * aspect, p.y * u_scale);

    vec2 c;
    if (u_mode == 0) {
        c = z;
        z = vec2(0.0);
    } else {
        c = u_julia_c;
    }

    float e = escapeTime(z, c);

    if (e < 0.0) {
        fragColor = vec4(u_palette[0], 1.0);
        return;
    }

    float t = e / float(u_iterations);
    // A nonlinear mapping makes boundary detail more visible at normal zoom.
    t = pow(clamp(t, 0.0, 1.0), 0.55);
    t += u_palette_shift * 0.035;

    vec3 col = palette(t);

    // Subtle animated glow; deliberately restrained so the mathematics remains visible.
    float pulse = 0.985 + 0.015 * sin(u_time * 0.8);
    fragColor = vec4(col * pulse, 1.0);
}
"