# agent_inference_tools_v2.py
# Manim CE 0.19.x compatible
# Passes implemented: A) fake depth/shadows, B) curved routed paths, C) livelier token stream

from manim import (
    ThreeDScene, Prism, Square, VGroup, Rectangle, Torus, Text, Sphere, Cylinder,
    RoundedRectangle, SurroundingRectangle, FadeIn, FadeOut, Create, UpdateFromAlphaFunc,
    DEGREES, TAU, BLUE_E, WHITE, GREY_A, YELLOW_B, YELLOW_C,
    RIGHT, LEFT, UP, DOWN, IN, OUT, smooth, BackgroundRectangle, Line, Circle, CubicBezier, always_redraw
)
import numpy as np
import random


class InceptionToolUse3D(ThreeDScene):
    def construct(self):
        # ======================= THEME & TOGGLES =======================
        DETAIL_LEVEL = 1   # 0 = minimal (fast), 1 = default
        THEME = dict(
            bg="#0b0f14",
            block_fill="#16324d",
            block_edge=BLUE_E,
            block_face="#102331",
            rail="#0e2233",
            link="#1b2a35",
            label_text=GREY_A,
            label_bg="#0c1218",
            token="#34d399",          # normal token
            token_hot="#22d3ee",      # emphasized token color used briefly
            tool_colors=["#0c4a3e", "#12395b", "#44235b"],
            tool_glow=["#2dd4bf", "#93c5fd", "#e9d5ff"],
            pulse_out=["#34d399", "#60a5fa", "#c084fc"],
            pulse_in=["#10b981", "#3b82f6", "#a855f7"],
            caption=GREY_A,
            shadow="#000000",
        )

        # World & camera
        self.camera.background_color = THEME["bg"]
        self.set_camera_orientation(phi=70 * DEGREES, theta=60 * DEGREES, zoom=1.0)
        self.begin_ambient_camera_rotation(rate=0.05)

        # ======================= HELPERS =======================
        def make_block(pos, w=1.6, h=1.0, d=1.0, color=THEME["block_fill"]):
            block = Prism(
                dimensions=(w, h, d),
                fill_opacity=1.0,
                fill_color=color,
                stroke_color=THEME["block_edge"],
                stroke_width=1.5,
            ).move_to(pos)

            # Faux front face panel
            face = Square(side_length=0.95 * w, stroke_width=0).set_fill(THEME["block_face"], opacity=0.9)
            face.move_to(block.get_center() + OUT * (d / 2 + 0.001))

            # Little "bar chart" on the face
            bars = VGroup(
                Rectangle(width=0.08 * w, height=0.5 * h, fill_opacity=1, fill_color="#26a69a", stroke_width=0),
                Rectangle(width=0.08 * w, height=0.35 * h, fill_opacity=1, fill_color="#42a5f5", stroke_width=0),
                Rectangle(width=0.08 * w, height=0.25 * h, fill_opacity=1, fill_color="#ab47bc", stroke_width=0),
            ).arrange(RIGHT, buff=0.06).move_to(face.get_center())

            return VGroup(block, face, bars)

        def make_shadow(target_getter, kind="rect", base_opacity=0.18, scale=1.0, y_offset=-0.32):
            """
            Returns an always_redraw Circle used as a soft "ground shadow".
            target_getter(): function with no args that returns a point (x,y,z) to follow.
            kind kept for future (all use Circle scaled to an ellipse).
            """
            def _factory():
                p = target_getter()
                s = Circle(radius=0.35 * scale, stroke_width=0).set_fill(THEME["shadow"], opacity=base_opacity)
                s.stretch(1.8 * scale, 0)   # widen in x
                s.stretch(0.6 * scale, 1)   # flatten in y
                s.move_to(np.array([p[0], y_offset, 0.0]))
                return s

            return always_redraw(_factory)

        def make_tool_node(label, pos, radius=0.5, color="#14532d"):
            ring = Torus(
                major_radius=radius,
                minor_radius=0.12,
                fill_opacity=1,
                fill_color=color,
                stroke_color="#1a9e67",
                stroke_width=1.5,
            ).move_to(pos)

            # Label that always faces camera (billboarding later)
            tag_text = Text(label, font="DejaVu Sans").scale(0.35).set_color(THEME["label_text"])
            tag_bg = BackgroundRectangle(tag_text, fill_opacity=0.6, fill_color=THEME["label_bg"], buff=0.06)
            tag = VGroup(tag_text, tag_bg)
            tag.move_to(ring.get_center() + 0.25 * OUT + 0.18 * UP)

            connector = Line(
                tag.get_center() + 0.1 * DOWN,
                ring.get_center(),
                stroke_width=1.5,
                color="#3a5161",
            ).set_opacity(0.6)

            return VGroup(ring, tag, connector)

        def sphere_token(color, r=0.08):
            return Sphere(
                radius=r,
                resolution=(16, 16),
                fill_opacity=1,
                fill_color=color,
                stroke_width=0.5,
                stroke_color=WHITE,
            )

        def connector_cylinder(p1, p2, color=THEME["link"]):
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

        def curved_path(p_start, p_end, arch_out=0.8, arch_down=0.6):
            """
            A 3D cubic Bezier path arcing outwards and slightly down for depth.
            Control points produce a clean S-curve that avoids straight lines.
            """
            p1 = np.array(p_start)
            p4 = np.array(p_end)
            mid = 0.5 * (p1 + p4)

            # Vector perpendicular-ish to introduce an outward arch
            v = p4 - p1
            if np.linalg.norm(v) < 1e-6:
                v = np.array([1.0, 0.0, 0.0])

            # pick a sideways dir roughly vertical/OUT
            side = np.cross(v, np.array([0.0, 0.0, 1.0]))
            if np.linalg.norm(side) < 1e-6:
                side = np.array([0.0, 1.0, 0.0])
            side = side / np.linalg.norm(side)

            c1 = p1 + 0.25 * v + arch_out * side + arch_down * DOWN
            c2 = p4 - 0.25 * v + arch_out * side + arch_down * DOWN

            return CubicBezier(p1, c1, c2, p4).set_stroke(width=1.5, color=THEME["link"]).set_opacity(0.25)

        def make_pulse_along_bezier(bezier_mobj, color="#ffffff", r=0.06):
            """
            Returns (dot, updater) to move a small Sphere along a CubicBezier in [0,1].
            """
            dot = Sphere(radius=r, fill_opacity=1, fill_color=color, stroke_width=0.5, stroke_color=WHITE)

            def updater(m, alpha):
                # alpha in [0,1]; point_from_proportion works with VMobject
                m.move_to(bezier_mobj.point_from_proportion(alpha))

            return dot, updater

        # ======================= LAYOUT =======================
        n_blocks = 6
        x0 = -5.0
        dx = 2.0
        z_stagger = [0.15, 0.05, 0.0, -0.02, 0.04, 0.12]  # subtle depth offsets

        blocks = VGroup()
        for i in range(n_blocks):
            p = np.array([x0 + i * dx, 0.0, 0.0 + z_stagger[i % len(z_stagger)]])
            blocks.add(make_block(pos=p))
        self.play(*[FadeIn(b, shift=IN * 0.2) for b in blocks], run_time=1.4)

        # Ground shadows for blocks
        if DETAIL_LEVEL >= 1:
            block_shadows = VGroup(
                *[
                    make_shadow(lambda b=b: b.get_center(), scale=1.2)
                    for b in blocks
                ]
            )
            self.add(block_shadows)

        # Rail
        rail = Cylinder(
            radius=0.05,
            height=(n_blocks - 1) * dx + 1.2,
            direction=RIGHT,
            fill_opacity=1,
            fill_color=THEME["rail"],
            stroke_width=0,
        )
        rail.move_to(np.array([(x0 + (n_blocks - 1) * dx) / 2, -0.25, 0.0]))
        self.play(FadeIn(rail), run_time=0.5)

        # Tokens with micro-variation
        n_tokens = 15
        token_spacing = 0.35
        tokens = VGroup()
        token_meta = []  # per-token dicts: speed, yjit, zjit, scale

        rng = random.Random(42)
        for k in range(n_tokens):
            s = sphere_token(THEME["token"], r=0.06 * rng.uniform(0.95, 1.05))
            base_x = x0 - 0.8 - k * token_spacing
            base_y = -0.25 + rng.uniform(-0.01, 0.01)  # tiny y jitter
            base_z = rng.uniform(-0.02, 0.02)          # tiny z jitter
            s.move_to(np.array([base_x, base_y, base_z]))
            tokens.add(s)

            token_meta.append(dict(
                speed=rng.uniform(1.45, 1.75),   # slight velocity differences
                yjit=rng.uniform(0.002, 0.006),
                zjit=rng.uniform(0.002, 0.006),
                phase=rng.uniform(0, TAU),
                base_r=s.radius,
            ))

        self.add(tokens)

        # Token ground shadows
        if DETAIL_LEVEL >= 1:
            token_shadows = VGroup(
                *[
                    make_shadow(lambda s=s: s.get_center(), scale=0.35, base_opacity=0.15)
                    for s in tokens
                ]
            )
            self.add(token_shadows)

        center_block_idx = n_blocks // 2
        center_pos = blocks[center_block_idx].get_center()

        tools = VGroup(
            make_tool_node("Search API", center_pos + np.array([-1.8, -1.2, -1.6]), color=THEME["tool_colors"][0]),
            make_tool_node("DB Query",   center_pos + np.array([ 0.0, -1.8, -2.0]), color=THEME["tool_colors"][1]),
            make_tool_node("Code Exec",  center_pos + np.array([ 1.8, -1.2, -1.6]), color=THEME["tool_colors"][2]),
        )

        # Billboard labels; connectors update to follow
        for t in tools:
            self.add_fixed_orientation_mobjects(t[1])  # tag group
            t[2].add_updater(
                lambda m, t=t: m.put_start_and_end_on(
                    t[1].get_center() + 0.1 * DOWN, t[0].get_center()
                )
            )

        self.play(*[FadeIn(t, shift=DOWN * 0.2) for t in tools], run_time=0.8)

        # Faint straight link lines (kept, but subtle)
        c_origin = center_pos + np.array([0, -0.25, 0])
        link_lines = VGroup(
            connector_cylinder(c_origin, tools[0][0].get_center(), color=THEME["link"]),
            connector_cylinder(c_origin, tools[1][0].get_center(), color=THEME["link"]),
            connector_cylinder(c_origin, tools[2][0].get_center(), color=THEME["link"]),
        ).set_opacity(0.22)
        self.play(FadeIn(link_lines), run_time=0.5)

        # ======================= TOKEN MOTION (UPDATER) =======================
        x_wrap = x0 + (n_blocks - 1) * dx + 0.8
        total_span = (n_blocks - 1) * dx + 2.0

        def shift_tokens(mobj, dt):
            # micro-jitter over time
            t_now = self.time
            for i, s in enumerate(mobj):
                meta = token_meta[i]
                base_speed = meta["speed"]
                # slow sine jitter to speed (±4%)
                speed = base_speed * (1.0 + 0.04 * np.sin(t_now * 0.9 + meta["phase"]))
                s.shift(RIGHT * dt * speed)

                # gentle y/z oscillation
                y0 = s.get_center()[1]
                z0 = s.get_center()[2]
                s.move_to(np.array([
                    s.get_center()[0],
                    -0.25 + meta["yjit"] * np.sin(t_now * 2.0 + meta["phase"]),
                    0.0 + meta["zjit"] * np.cos(t_now * 1.7 + meta["phase"])
                ]))

                # wrap-around
                if s.get_center()[0] > x_wrap:
                    s.shift(LEFT * total_span)

        tokens.add_updater(shift_tokens)
        self.wait(1.6)

        # ======================= PAUSE UI =======================
        tokens.remove_updater(shift_tokens)
        pause_plate = RoundedRectangle(
            width=1.1, height=0.7, corner_radius=0.08,
            fill_opacity=1, fill_color="#17212b", stroke_color=YELLOW_B, stroke_width=2,
        )
        bar1 = Rectangle(width=0.18, height=0.44, fill_opacity=1, fill_color=YELLOW_C, stroke_width=0).shift(LEFT * 0.18)
        bar2 = Rectangle(width=0.18, height=0.44, fill_opacity=1, fill_color=YELLOW_C, stroke_width=0).shift(RIGHT * 0.18)
        pause_icon = VGroup(pause_plate, bar1, bar2).move_to(center_pos + np.array([0, 0.0, 0.6]))
        self.play(FadeIn(pause_icon, shift=OUT * 0.2), run_time=0.5)

        caption1 = Text("Policy pauses to decide tools (inference-time)", font="DejaVu Sans").scale(0.35).set_color(THEME["caption"])
        self.add_fixed_orientation_mobjects(caption1)
        caption1.move_to(center_pos + np.array([0, 0.9, 0.6]))
        self.play(FadeIn(caption1), run_time=0.4)

        # ======================= CURVED PULSES (OUTGOING) =======================
        out_paths = [
            curved_path(c_origin, tools[0][0].get_center(), arch_out=0.8, arch_down=0.5),
            curved_path(c_origin, tools[1][0].get_center(), arch_out=0.9, arch_down=0.6),
            curved_path(c_origin, tools[2][0].get_center(), arch_out=1.0, arch_down=0.5),
        ]
        for p in out_paths:
            self.add(p)

        out_pulses = []
        for i, path in enumerate(out_paths):
            dot, up = make_pulse_along_bezier(path, color=THEME["pulse_out"][i], r=0.06)
            out_pulses.append((dot, up))
            self.add(dot)

        self.play(
            UpdateFromAlphaFunc(out_pulses[0][0], out_pulses[0][1]),
            UpdateFromAlphaFunc(out_pulses[1][0], out_pulses[1][1]),
            UpdateFromAlphaFunc(out_pulses[2][0], out_pulses[2][1]),
            run_time=1.3, rate_func=smooth,
        )

        # Tool glows
        glows = VGroup(
            *[
                SurroundingRectangle(t[0], color=c, buff=0.08).set_stroke(width=3).set_fill(opacity=0)
                for t, c in zip(tools, THEME["tool_glow"])
            ]
        )
        self.play(*[Create(g) for g in glows], run_time=0.45)
        self.play(*[FadeOut(g) for g in glows], run_time=0.35)

        # ======================= CURVED PULSES (INCOMING) =======================
        in_paths = [
            curved_path(tools[0][0].get_center(), c_origin, arch_out=0.8, arch_down=0.5),
            curved_path(tools[1][0].get_center(), c_origin, arch_out=0.9, arch_down=0.6),
            curved_path(tools[2][0].get_center(), c_origin, arch_out=1.0, arch_down=0.5),
        ]
        for p in in_paths:
            self.add(p)

        in_pulses = []
        for i, path in enumerate(in_paths):
            dot, up = make_pulse_along_bezier(path, color=THEME["pulse_in"][i], r=0.06)
            in_pulses.append((dot, up))
            self.add(dot)

        self.play(
            UpdateFromAlphaFunc(in_pulses[0][0], in_pulses[0][1]),
            UpdateFromAlphaFunc(in_pulses[1][0], in_pulses[1][1]),
            UpdateFromAlphaFunc(in_pulses[2][0], in_pulses[2][1]),
            run_time=1.4, rate_func=smooth,
        )

        # Integrate pulse (halo)
        halo = SurroundingRectangle(blocks[center_block_idx][0], color=YELLOW_C, buff=0.15).set_stroke(width=5).set_fill(opacity=0)
        self.play(Create(halo), FadeOut(pause_icon), FadeOut(caption1), run_time=0.6)
        self.play(FadeOut(halo), run_time=0.4)

        # ======================= RESUME: livelier stream =======================
        tokens.add_updater(shift_tokens)

        caption2 = Text(
            "Tool outputs integrated → inference resumes (token stream continues)",
            font="DejaVu Sans",
        ).scale(0.35).set_color(THEME["caption"]).to_edge(DOWN)
        self.add_fixed_orientation_mobjects(caption2)
        self.play(FadeIn(caption2), run_time=0.4)
        self.wait(0.3)

        # Brief “energized” burst for the first few tokens
        hot_N = 6
        self.play(*[s.animate.set_fill(THEME["token_hot"]) for s in tokens[:hot_N]], run_time=0.35)
        self.wait(0.4)
        self.play(*[s.animate.set_fill(THEME["token"]) for s in tokens[:hot_N]], run_time=0.35)
        self.wait(0.6)

        # Cleanup
        tokens.remove_updater(shift_tokens)
        for t in tools:
            t[2].clear_updaters()
        self.wait(0.25)
