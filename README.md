# Gravitational Interaction Simulator

An interactive simulation of bodies moving under gravitational forces, written in Python using Pygame.

## Features

- Create and delete celestial bodies
- Configure masses and initial velocities
- Collision physics
- Movement trajectory visualization
- Camera scaling and movement
- Simulation speed control

## Requirements

- Python 3.8+
- pygame
- numpy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/veroxci/gravity_simulation
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # for Linux/Mac
.venv\Scripts\activate     # for Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Launch the simulation:
```bash
python gravity-simulation.py
```

### Controls

- **Left Click**: Create new body
- **Drag**: Move body (when simulation is paused)
- **Shift + Drag**: Set body velocity
- **Ctrl + Drag**: Change body mass
- **Right Click + Drag**: Move camera
- **Mouse Wheel**: Scale view
- **Delete**: Remove selected body

### Interface

- Right slider: Display scale
- Center slider: Simulation speed
- "Start/Stop" button: Control simulation
- "Reset" button: Return to initial state
- Input fields: Precise parameter adjustment for selected body

## Physical Model

The simulation uses:
- Newton's law of universal gravitation
- 4th order Runge-Kutta method for numerical integration
- Collision model with body fragmentation

## Project Structure

- `gravity-simulation.py`: Main program file
- `body.py`: Celestial body class
- `constants.py`: Physical and program constants
- `util.py`: Utility functions and classes
- `ui_elements.py`: Interface elements

## Authors

- Dubinkina Veronika

## License

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
