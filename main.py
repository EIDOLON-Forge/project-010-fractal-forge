
"""
EIDOLON #010 — FRACTAL FORGE
GPU Mandelbrot + Julia explorer
Python 3.12+ / Pygame 2.6.1 / PyOpenGL 3.1.7+

Run:
    python main.py
"""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import pygame
from pygame.locals import DOUBLEBUF, OPENGL

from OpenGL.GL import (
    GL_ARRAY_BUFFER, GL_BLEND, GL_COLOR_BUFFER_BIT, GL_DEPTH_TEST,
    GL_FLOAT, GL_FRAGMENT_SHADER, GL_LINEAR, GL_ONE_MINUS_SRC_ALPHA,
    GL_RGBA, GL_SRC_ALPHA, GL_STATIC_DRAW, GL_TEXTURE0, GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_MAG_FILTER, GL_TRIANGLE_STRIP, GL_UNSIGNED_BYTE,
    GL_VERTEX_SHADER, glActiveTexture, glAttachShader, glBindBuffer,
    glBindTexture, glBindVertexArray, glBlendFunc, glBufferData,
    glClear, glClearColor, glCreateProgram, glCreateShader, glDeleteProgram,
    glDeleteShader, glDisable, glDrawArrays, glEnable, glGenBuffers,
    glGenTextures, glGenVertexArrays, glGetUniformLocation, glLinkProgram,
    glShaderSource, glTexImage2D, glTexParameteri, glUniform1f,
    glUniform1i, glUniform2f, glUniform3fv, glUseProgram, glValidateProgram,
    glVertexAttribPointer, glEnableVertexAttribArray, glReadPixels,
)
from OpenGL.GL import GL_COMPILE_STATUS, GL_LINK_STATUS, GL_VALIDATE_STATUS, glGetShaderiv, glGetShaderInfoLog, glGetProgramiv, glGetProgramInfoLog

WIDTH, HEIGHT = 1200, 800
FPS = 60
MAX_ITERATIONS, MIN_ITERATIONS, DEFAULT_ITERATIONS = 1200, 40, 180
ZOOM_FACTOR = 0.72
PAN_FACTOR = 0.06

PALETTES = (
    ("EIDOLON", ((0,0,0),(15,25,30),(25,80,90),(70,180,180),(180,255,235),(245,255,255))),
    ("FIRE",    ((0,0,0),(45,5,0),(120,15,0),(220,60,0),(255,170,20),(255,245,180))),
    ("OCEAN",   ((0,0,10),(5,25,65),(10,80,150),(35,170,220),(150,235,255),(240,255,255))),
    ("VIOLET",  ((0,0,0),(25,5,55),(75,15,130),(145,55,210),(220,150,255),(255,235,255))),
    ("MATRIX",  ((0,0,0),(0,35,8),(0,95,20),(0,180,50),(80,255,130),(220,255,225))),
)

VERTEX_SHADER = """#version 330 core
layout(location=0) in vec2 a_pos;
out vec2 uv;
void main() {
    uv = a_pos * 0.5 + 0.5;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
"""

FRACTAL_SHADER = """#version 330 core
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
"""

TEXT_VERTEX_SHADER = """#version 330 core
layout(location=0) in vec2 a_pos;
layout(location=1) in vec2 a_uv;
out vec2 uv;
void main() {
    uv = a_uv;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
"""

TEXT_FRAGMENT_SHADER = """#version 330 core
in vec2 uv;
out vec4 fragColor;
uniform sampler2D u_tex;
void main() {
    fragColor = texture(u_tex, uv);
}
"""

@dataclass
class View:
    x: float
    y: float
    scale: float

    def reset(self, x: float, y: float, scale: float):
        self.x, self.y, self.scale = x, y, scale

def compile_shader(source: str, kind: int) -> int:
    shader = glCreateShader(kind)
    glShaderSource(shader, source)
    from OpenGL.GL import glCompileShader
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        msg = glGetShaderInfoLog(shader).decode("utf-8", "replace")
        glDeleteShader(shader)
        raise RuntimeError(f"GLSL compile error:\n{msg}")
    return shader

def make_program(vertex: str, fragment: str) -> int:
    vs = compile_shader(vertex, GL_VERTEX_SHADER)
    fs = compile_shader(fragment, GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vs)
    glAttachShader(program, fs)
    glLinkProgram(program)
    glDeleteShader(vs)
    glDeleteShader(fs)
    if not glGetProgramiv(program, GL_LINK_STATUS):
        msg = glGetProgramInfoLog(program).decode("utf-8", "replace")
        glDeleteProgram(program)
        raise RuntimeError(f"GLSL link error:\n{msg}")
    glUseProgram(program)
    glValidateProgram(program)
    if not glGetProgramiv(program, GL_VALIDATE_STATUS):
        # Validation warnings are useful but should not make otherwise valid drivers fail.
        print("OpenGL validation warning:", glGetProgramInfoLog(program).decode("utf-8", "replace"))
    return program

def make_fullscreen_quad():
    # Four vertices for GL_TRIANGLE_STRIP.
    # Each vertex is [x, y, u, v]. No NULL pointer offsets are used.
    import ctypes
    data = (ctypes.c_float * 16)(
        -1, -1, 0, 0,
         1, -1, 1, 0,
        -1,  1, 0, 1,
         1,  1, 1, 1,
    )
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, ctypes.sizeof(data), data, GL_STATIC_DRAW)
    stride = 4 * ctypes.sizeof(ctypes.c_float)
    glVertexAttribPointer(0, 2, GL_FLOAT, False, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 2, GL_FLOAT, False, stride, ctypes.c_void_p(2 * ctypes.sizeof(ctypes.c_float)))
    glEnableVertexAttribArray(1)
    glBindVertexArray(0)
    return vao, vbo

def make_hud_texture(w, h):
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    # Allocate once. The pixels are replaced each frame with glTexImage2D.
    blank = bytes(w * h * 4)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, blank)
    return tex

class App:
    def __init__(self, width=WIDTH, height=HEIGHT, smoke_test=False):
        pygame.init()
        pygame.font.init()

        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)

        self.width, self.height = width, height
        self.screen = pygame.display.set_mode((width, height), OPENGL | DOUBLEBUF)
        pygame.display.set_caption("EIDOLON #010 — FRACTAL FORGE")

        self.clock = pygame.time.Clock()
        self.running = True
        self.mode = 0
        self.iterations = DEFAULT_ITERATIONS
        self.palette_index = 0
        self.palette_animation = False
        self.palette_shift = 0.0
        self.audio_enabled = True
        self.started = False
        self.countdown_until = 0.0
        self.intro_until = time.perf_counter() + 1.7
        self.mouse_drag = False
        self.last_mouse = (0, 0)

        self.mandel = View(-0.5, 0.0, 3.2)
        self.julia = View(0.0, 0.0, 3.2)
        self.julia_c = (-0.745, 0.113)
        self.random = random.Random()

        glClearColor(0, 0, 0, 1)
        glDisable(GL_DEPTH_TEST)

        self.fractal_program = make_program(VERTEX_SHADER, FRACTAL_SHADER)
        self.text_program = make_program(TEXT_VERTEX_SHADER, TEXT_FRAGMENT_SHADER)
        self.vao, self.vbo = make_fullscreen_quad()
        self.hud_tex = make_hud_texture(width, height)

        self.u_center = glGetUniformLocation(self.fractal_program, "u_center")
        self.u_scale = glGetUniformLocation(self.fractal_program, "u_scale")
        self.u_iterations = glGetUniformLocation(self.fractal_program, "u_iterations")
        self.u_mode = glGetUniformLocation(self.fractal_program, "u_mode")
        self.u_julia_c = glGetUniformLocation(self.fractal_program, "u_julia_c")
        self.u_palette = glGetUniformLocation(self.fractal_program, "u_palette")
        self.u_shift = glGetUniformLocation(self.fractal_program, "u_palette_shift")
        self.u_time = glGetUniformLocation(self.fractal_program, "u_time")

        self.u_tex = glGetUniformLocation(self.text_program, "u_tex")

        # A tiny sound palette. Sound is intentionally optional.
        try:
            pygame.mixer.init()
        except pygame.error:
            self.audio_enabled = False

        if smoke_test:
            self.run_smoke_test()
            self.running = False

    @property
    def view(self):
        return self.mandel if self.mode == 0 else self.julia

    @property
    def palette(self):
        return PALETTES[self.palette_index][1]

    def sound(self, frequency=440, duration=0.045):
        if not self.audio_enabled or not pygame.mixer.get_init():
            return
        try:
            sample_rate = 44100
            count = max(1, int(sample_rate * duration))
            import array
            buf = array.array("h")
            for i in range(count):
                env = min(1.0, i / max(1, count * 0.08)) * max(0.0, 1.0 - i / count)
                v = int(8000 * env * math.sin(2 * math.pi * frequency * i / sample_rate))
                buf.append(v)
            snd = pygame.mixer.Sound(buffer=buf.tobytes())
            snd.play()
        except Exception:
            pass

    def reset(self):
        self.mandel.reset(-0.5, 0.0, 3.2)
        self.julia.reset(0.0, 0.0, 3.2)
        self.iterations = DEFAULT_ITERATIONS
        self.palette_shift = 0.0
        self.sound(220, 0.08)

    def screen_to_complex(self, pos):
        x, y = pos
        view = self.view
        aspect = self.width / self.height
        nx = x / self.width - 0.5
        ny = y / self.height - 0.5
        return (
            view.x + nx * view.scale * aspect,
            view.y + ny * view.scale,
        )

    def zoom_at(self, mouse_pos, factor):
        before = self.screen_to_complex(mouse_pos)
        view = self.view
        view.scale = max(1e-12, min(100.0, view.scale * factor))
        after = self.screen_to_complex(mouse_pos)
        view.x += before[0] - after[0]
        view.y += before[1] - after[1]
        self.sound(620 if factor < 1 else 330, 0.025)

    def pan(self, dx, dy):
        view = self.view
        view.x += dx * view.scale
        view.y += dy * view.scale

    def random_julia(self):
        self.julia_c = (
            self.random.uniform(-1.0, 1.0),
            self.random.uniform(-1.0, 1.0),
        )
        self.mode = 1
        self.julia.reset(0.0, 0.0, 3.2)
        self.sound(520, 0.05)

    def cycle_palette(self):
        self.palette_index = (self.palette_index + 1) % len(PALETTES)
        self.sound(760, 0.04)

    def handle(self, event):
        if event.type == pygame.QUIT:
            self.running = False
            return
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.running = False
            elif event.key == pygame.K_f:
                self.mode = 1 - self.mode
                self.sound(480, 0.04)
            elif event.key == pygame.K_p:
                self.cycle_palette()
            elif event.key == pygame.K_SPACE:
                self.palette_animation = not self.palette_animation
            elif event.key == pygame.K_j:
                self.random_julia()
            elif event.key == pygame.K_r:
                self.reset()
            elif event.key == pygame.K_m:
                self.audio_enabled = not self.audio_enabled
            elif event.key == pygame.K_s:
                self.save_screenshot()
            elif event.key in (pygame.K_EQUALS, pygame.K_KP_PLUS):
                self.iterations = min(MAX_ITERATIONS, self.iterations + 20)
            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.iterations = max(MIN_ITERATIONS, self.iterations - 20)
            elif event.key == pygame.K_UP:
                self.pan(0, -PAN_FACTOR)
            elif event.key == pygame.K_DOWN:
                self.pan(0, PAN_FACTOR)
            elif event.key == pygame.K_LEFT:
                self.pan(-PAN_FACTOR, 0)
            elif event.key == pygame.K_RIGHT:
                self.pan(PAN_FACTOR, 0)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                self.zoom_at(event.pos, ZOOM_FACTOR)
            elif event.button == 3:
                self.zoom_at(event.pos, 1.0 / ZOOM_FACTOR)
            elif event.button == 2:
                self.mouse_drag = True
                self.last_mouse = event.pos
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 2:
            self.mouse_drag = False
        elif event.type == pygame.MOUSEMOTION and self.mouse_drag:
            x, y = event.pos
            lx, ly = self.last_mouse
            dx = (lx - x) / self.width * self.view.scale * (self.width / self.height)
            dy = (ly - y) / self.height * self.view.scale
            self.view.x += dx
            self.view.y += dy
            self.last_mouse = event.pos

    def save_screenshot(self):
        raw = glReadPixels(0, 0, self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE)
        surf = pygame.image.frombuffer(raw, (self.width, self.height), "RGBA")
        surf = pygame.transform.flip(surf, False, True).convert()
        path = Path(f"fractal_{time.strftime('%Y%m%d_%H%M%S')}.png")
        pygame.image.save(surf, path)
        print(f"Screenshot saved: {path}")
        self.sound(900, 0.05)

    def render_fractal(self, now):
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.fractal_program)

        view = self.view
        glUniform2f(self.u_center, view.x, view.y)
        glUniform1f(self.u_scale, view.scale)
        glUniform1i(self.u_iterations, self.iterations)
        glUniform1i(self.u_mode, self.mode)
        glUniform2f(self.u_julia_c, *self.julia_c)
        glUniform1f(self.u_shift, self.palette_shift)
        glUniform1f(self.u_time, now)

        flat = [c / 255.0 for rgb in self.palette for c in rgb]
        glUniform3fv(self.u_palette, 6, flat)

        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)

    def make_hud(self):
        surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        font = pygame.font.SysFont("consolas", 18, bold=True)
        small = pygame.font.SysFont("consolas", 14)

        title = font.render("EIDOLON // FRACTAL FORGE", True, (235, 245, 245))
        surf.blit(title, (22, 18))

        mode = "MANDELBROT" if self.mode == 0 else "JULIA"
        name = PALETTES[self.palette_index][0]
        lines = [
            f"{mode}   |   PALETTE: {name}",
            f"ITERATIONS: {self.iterations:4d}   |   SCALE: {self.view.scale:.8g}",
            f"C: ({self.julia_c[0]:+.5f}, {self.julia_c[1]:+.5f})" if self.mode else "C = pixel coordinate",
            "LMB zoom in   RMB zoom out   MMB drag",
            "ARROWS pan   +/- iterations   F mode   P palette   SPACE animate",
            "J random Julia   S screenshot   R reset   M audio   ESC quit",
        ]
        y = 48
        for i, line in enumerate(lines):
            img = (small if i else font).render(line, True, (215, 225, 225))
            surf.blit(img, (22, y))
            y += 22 if i == 0 else 20

        # A restrained translucent panel keeps text readable without hiding the fractal.
        panel = pygame.Surface((500, 142), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 135))
        surf.blit(panel, (12, 10))

        # Re-blit title/text over the panel.
        surf.blit(title, (22, 18))
        y = 48
        for i, line in enumerate(lines):
            img = (small if i else font).render(line, True, (215, 225, 225))
            surf.blit(img, (22, y))
            y += 22 if i == 0 else 20

        return surf

    def render_hud(self):
        surf = self.make_hud()
        # pygame SRCALPHA surfaces are stored in BGRA on little-endian Windows;
        # convert to an explicit RGBA byte layout before uploading.
        data = pygame.image.tostring(surf, "RGBA", True)
        glUseProgram(self.text_program)
        glBindTexture(GL_TEXTURE_2D, self.hud_tex)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA, self.width, self.height,
            0, GL_RGBA, GL_UNSIGNED_BYTE, data
        )
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.hud_tex)
        glUniform1i(self.u_tex, 0)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        glDisable(GL_BLEND)

    def draw_intro(self):
        # The same HUD path is used, so there is no separate GL text pipeline.
        pass

    def run_smoke_test(self):
        self.render_fractal(time.perf_counter())
        pygame.display.flip()
        print("GPU fractal smoke test: OK")
        print("OpenGL context: 3.3 core requested")
        print("Renderer: GPU GLSL fragment shader")

    def run(self):
        while self.running:
            for event in pygame.event.get():
                self.handle(event)

            now = time.perf_counter()
            if self.palette_animation:
                self.palette_shift = (now * 0.22) % 1.0

            self.render_fractal(now)
            self.render_hud()
            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    try:
        app = App(smoke_test=args.smoke_test)
        if not args.smoke_test:
            app.run()
        else:
            pygame.quit()
    except Exception:
        pygame.quit()
        raise

if __name__ == "__main__":
    main()
