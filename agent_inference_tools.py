from manim import (
    ThreeDScene, Prism, Square, VGroup, Rectangle, Torus, Text, Sphere, 
    Cylinder, RoundedRectangle, SurroundingRectangle, FadeIn, FadeOut, 
    Create, UpdateFromAlphaFunc, DEGREES, TAU, BLUE_E, WHITE, GREY_A, 
    YELLOW_B, YELLOW_C, RIGHT, LEFT, UP, DOWN, IN, OUT, smooth
)
import numpy as np  # needed for vector math

# 3D scene: token flow -> pause -> tool calls -> results -> resume
class InceptionToolUse3D(ThreeDScene):
    def construct(self):
        self.camera.background_color = "#0b0f14"
        self.set_camera_orientation(phi=70 * DEGREES, theta=60 * DEGREES, zoom=1.0)
        self.begin_ambient_camera_rotation(rate=0.05)

        # ------------ Helpers ------------
        def make_block(pos, w=1.6, h=1.0, d=1.0, color="#16324d"):
            block = Prism(dimensions=(w, h, d), fill_opacity=1.0, fill_color=color,
                          stroke_color=BLUE_E, stroke_width=1.5)
            block.move_to(pos)
            face = Square(side_length=0.95*w, stroke_width=0).set_fill("#102331", opacity=0.9)
            face.move_to(block.get_center() + OUT * (d/2 + 0.001))
            bars = VGroup(
                Rectangle(width=0.08*w, height=0.5*h, fill_opacity=1, fill_color="#26a69a", stroke_width=0),
                Rectangle(width=0.08*w, height=0.35*h, fill_opacity=1, fill_color="#42a5f5", stroke_width=0),
                Rectangle(width=0.08*w, height=0.25*h, fill_opacity=1, fill_color="#ab47bc", stroke_width=0),
            ).arrange(RIGHT, buff=0.06).move_to(face.get_center())
            return VGroup(block, face, bars)

        def make_tool_node(label, pos, radius=0.5, color="#14532d"):
            ring = Torus(
                major_radius=radius,
                minor_radius=0.12,
                u_range=[0, TAU],
                v_range=[0, TAU],
                fill_opacity=1,
                fill_color=color,
                stroke_color="#1a9e67",
                stroke_width=1.5,
            )

            ring.move_to(pos)
            tag = Text(label, font="DejaVu Sans").scale(0.35).set_color(GREY_A)
            tag.next_to(ring, OUT, buff=0.05)
            return VGroup(ring, tag)

        def sphere_pulse(color, r=0.08):
            return Sphere(radius=r, resolution=(16, 16), fill_opacity=1, fill_color=color,
                          stroke_width=0.5, stroke_color=WHITE)

        # ------------ Layout ------------
        n_blocks = 6
        x0 = -5.0
        dx = 2.0
        blocks = VGroup()
        for i in range(n_blocks):
            blocks.add(make_block(pos=np.array([x0 + i * dx, 0.0, 0.0])))
        self.play(*[FadeIn(b, shift=IN*0.2) for b in blocks], run_time=1.4)

        rail = Cylinder(radius=0.05, height=(n_blocks - 1) * dx + 1.2, direction=RIGHT,
                        fill_opacity=1, fill_color="#0e2233", stroke_width=0)
        rail.move_to(np.array([(x0 + (n_blocks - 1) * dx) / 2, -0.25, 0.0]))
        self.play(FadeIn(rail), run_time=0.5)

        n_tokens = 15
        token_spacing = 0.35
        tokens = VGroup(*[sphere_pulse("#34d399", r=0.06) for _ in range(n_tokens)])
        for k, s in enumerate(tokens):
            s.move_to(np.array([x0 - 0.8 - k * token_spacing, -0.25, 0.0]))
        self.add(tokens)

        center_block_idx = n_blocks // 2
        center_pos = blocks[center_block_idx].get_center()
        tools = VGroup(
            make_tool_node("Search API", center_pos + np.array([-1.8, -1.2, -1.6]), color="#0c4a3e"),
            make_tool_node("DB Query",   center_pos + np.array([ 0.0, -1.8, -2.0]), color="#12395b"),
            make_tool_node("Code Exec",  center_pos + np.array([ 1.8, -1.2, -1.6]), color="#44235b"),
        )
        self.play(*[FadeIn(t, shift=DOWN*0.2) for t in tools], run_time=0.8)

        def connector(p1, p2, color="#3a5161"):
            v = p2 - p1
            L = np.linalg.norm(v) or 1e-6
            cyl = Cylinder(radius=0.025, height=L, fill_opacity=1, fill_color=color, stroke_width=0)
            v_hat = v / L
            axis = np.cross([0, 0, 1], v_hat)
            n = np.linalg.norm(axis)
            if n > 1e-6:
                axis = axis / n
                angle = np.arccos(np.clip(np.dot([0, 0, 1], v_hat), -1, 1))
                cyl.rotate(angle, axis=axis)
            cyl.move_to((p1 + p2) / 2)
            return cyl

        link_lines = VGroup(
            connector(center_pos + np.array([0, -0.25, 0]), tools[0][0].get_center(), color="#1b2a35"),
            connector(center_pos + np.array([0, -0.25, 0]), tools[1][0].get_center(), color="#1b2a35"),
            connector(center_pos + np.array([0, -0.25, 0]), tools[2][0].get_center(), color="#1b2a35"),
        ).set_opacity(0.35)
        self.play(FadeIn(link_lines), run_time=0.5)

        # Token motion
        def shift_tokens(dt):
            for s in tokens:
                s.shift(RIGHT * dt * 1.6)
                if s.get_center()[0] > x0 + (n_blocks - 1) * dx + 0.8:
                    s.shift(LEFT * ((n_blocks - 1) * dx + 2.0))

        self.add_updater(shift_tokens)
        self.wait(1.6)

        # Pause
        self.remove_updater(shift_tokens)
        pause_plate = RoundedRectangle(width=1.1, height=0.7, corner_radius=0.08,
                                       fill_opacity=1, fill_color="#17212b",
                                       stroke_color=YELLOW_B, stroke_width=2)
        bar1 = Rectangle(width=0.18, height=0.44, fill_opacity=1, fill_color=YELLOW_C, stroke_width=0).shift(LEFT*0.18)
        bar2 = Rectangle(width=0.18, height=0.44, fill_opacity=1, fill_color=YELLOW_C, stroke_width=0).shift(RIGHT*0.18)
        pause_icon = VGroup(pause_plate, bar1, bar2).move_to(center_pos + np.array([0, 0.0, 0.6]))
        self.play(FadeIn(pause_icon, shift=OUT*0.2), run_time=0.5)

        caption1 = Text("Policy pauses to decide tools (inception-time)", font="DejaVu Sans").scale(0.35).set_color(GREY_A)
        caption1.move_to(center_pos + np.array([0, 0.9, 0.6]))
        self.play(FadeIn(caption1), run_time=0.4)

        # Outgoing pulses
        def make_pulse_anim(p1, p2, color):
            dot = Sphere(radius=0.06, fill_opacity=1, fill_color=color, stroke_width=0.5, stroke_color=WHITE)
            dot.move_to(p1)
            def updater(m, alpha):
                m.move_to(p1 + alpha * (p2 - p1))
            return dot, updater

        c_origin = center_pos + np.array([0, -0.25, 0])
        p_out = []
        for i, tool in enumerate(tools):
            color = ["#34d399", "#60a5fa", "#c084fc"][i]
            dot, up = make_pulse_anim(c_origin, tool[0].get_center(), color)
            p_out.append((dot, up))
            self.add(dot)

        self.play(
            UpdateFromAlphaFunc(p_out[0][0], p_out[0][1]),
            UpdateFromAlphaFunc(p_out[1][0], p_out[1][1]),
            UpdateFromAlphaFunc(p_out[2][0], p_out[2][1]),
            run_time=1.2, rate_func=smooth
        )

        # Tool glows
        glows = VGroup(*[
            SurroundingRectangle(t[0], color=c, buff=0.08).set_stroke(width=3).set_fill(opacity=0)
            for t, c in zip(tools, ["#2dd4bf", "#93c5fd", "#e9d5ff"])
        ])
        self.play(*[Create(g) for g in glows], run_time=0.5)
        self.play(*[FadeOut(g) for g in glows], run_time=0.4)

        # Incoming pulses
        p_in = []
        for i, tool in enumerate(tools):
            color = ["#10b981", "#3b82f6", "#a855f7"][i]
            dot, up = make_pulse_anim(tool[0].get_center(), c_origin, color)
            p_in.append((dot, up))
            self.add(dot)

        self.play(
            UpdateFromAlphaFunc(p_in[0][0], p_in[0][1]),
            UpdateFromAlphaFunc(p_in[1][0], p_in[1][1]),
            UpdateFromAlphaFunc(p_in[2][0], p_in[2][1]),
            run_time=1.4, rate_func=smooth
        )

        # Integrate pulse
        halo = SurroundingRectangle(blocks[center_block_idx][0], color=YELLOW_C, buff=0.15).set_stroke(width=5).set_fill(opacity=0)
        self.play(Create(halo), FadeOut(pause_icon), FadeOut(caption1), run_time=0.6)
        self.play(FadeOut(halo), run_time=0.4)

        # Resume
        self.add_updater(shift_tokens)
        caption2 = Text("Tool outputs integrated → inference resumes (token stream continues)", font="DejaVu Sans").scale(0.35).set_color(GREY_A).to_edge(DOWN)
        self.play(FadeIn(caption2), run_time=0.4)
        self.wait(2.0)

        self.play(*[s.animate.set_fill("#22d3ee") for s in tokens[:6]], run_time=0.4)
        self.wait(0.6)
        self.play(*[s.animate.set_fill("#34d399") for s in tokens[:6]], run_time=0.4)
        self.wait(0.8)

        self.remove_updater(shift_tokens)
        self.wait(0.3)
