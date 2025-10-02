from manim import *

class HelloWorld(Scene):
    def construct(self):
        text = Text("Hello from WSL2 + Manim!")
        self.play(Write(text))
        self.wait(1)
