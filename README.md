# Manim + WSL2 Setup

This project demonstrates how to run [Manim](https://docs.manim.community/) inside **WSL2 (Ubuntu)** on Windows.

---

## 🚀 Setup

1. **Install WSL2 + Ubuntu**
   ```bash
   wsl --install -d Ubuntu
   ```

2. **Update packages**
   ```bash
   sudo apt-get update -y && sudo apt-get upgrade -y
   ```

3. **Install dependencies**
   ```bash
   sudo apt-get install -y python3-venv python3-dev ffmpeg      libcairo2-dev libpango1.0-dev libgirepository1.0-dev      libglib2.0-dev libffi-dev gir1.2-pango-1.0 pkg-config      meson ninja-build
   ```

4. **Create virtual environment**
   ```bash
   python3 -m venv ~/manim-venv
   source ~/manim-venv/bin/activate
   ```

5. **Install Python packages**
   ```bash
   python -m pip install -U pip setuptools wheel
   pip install manimpango==0.6.0 "numpy>=2.0,<3.0" "scipy>=1.14,<1.15" manim==0.19.0
   ```

---

## ▶️ Run Example

Create `example.py`:

```python
from manim import *

class HelloWorld(Scene):
    def construct(self):
        text = Text("Hello from WSL2 + Manim!")
        self.play(Write(text))
        self.wait(1)
```

Render it:

```bash
manim -pqh example.py HelloWorld
```

- Output video:  
  `media/videos/example/480p15/HelloWorld.mp4`

---

## 📦 Optional: LaTeX Support

```bash
sudo apt-get install -y texlive-latex-extra texlive-fonts-extra texlive-latex-recommended
```

---

## 💻 Use with VS Code / Cursor

1. Install **Remote - WSL** extension.  
2. Open project folder: `/home/<you>/projects/manim/`  
3. Select interpreter: `~/manim-venv/bin/python`  
4. Run/debug directly from VS Code or Cursor.  

---

## ✅ Status

- [x] WSL2 + Ubuntu installed  
- [x] Manim + dependencies installed  
- [x] Example scene rendered  
- [ ] Try LaTeX scene  
- [ ] Build a real project 🎬  
