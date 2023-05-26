# Generating Organic Structures with the Reaction-Diffusion Equation

This repository contains the code for a university assignment for the Advanced Computer Graphics course. The goal is to implement a reaction-diffusion system to generate and render organic structures, such as corals and coral reefs.

![Render of a coral-like structure](./demo.gif)

## Contents
1. Jupyter notebooks for exploring and testing 2D and 3D reaction-diffusion models.
2. A Blender add-on for generating and rendering organic structures using the reaction-diffusion equation.

## Installation and Setup

### Jupyter Notebooks

1. Clone the repository.
```bash
git clone https://github.com/<your-username>/reaction-diffusion.git
cd reaction-diffusion
```

2. Create a virtual environment.
```bash
python -m venv env
source env/bin/activate  # Use "env\Scripts\activate" on Windows
```

3. Install the requirements.
```bash
pip install -r requirements.txt
```

4. Start Jupyter Notebook.
```bash
jupyter notebook
```

Now you can open the `.ipynb` files and run the cells.

### Blender Add-On

1. Download the ZIP file from the `blender_addon` directory.

2. Open Blender and go to `Edit -> Preferences -> Add-ons -> Install...`.

3. Select the downloaded ZIP file and click `Install Add-on`.

4. Enable the add-on by checking the box next to it.

5. You must also install `scipy` and `scikit-image` into Blender's Python. Open the Python console in Blender (`Scripting -> Python Console`) and type:
```python
import subprocess
import sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scipy'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scikit-image'])
```
**Note:** This will install the packages globally. If you want to install them for Blender only, you need to point to Blender's Python binary.

Now you can use the add-on to generate organic structures.

### Features

The add-on provides several options for fine-tuning the reaction-diffusion simulation and the generated structures:

- **Presets**: Choose from predefined settings to generate specific organic structures like "Opening Coral", "Alien Brain", "Wasp Nest", "Radio Snowflakes", and "Psychadelic Growth". You can also choose "Custom" to adjust all parameters manually.

- **Time and Diffusion Settings**: Adjust the time step and the diffusion rates for the two substances in the reaction-diffusion equation.

- **Feed and Kill Functions**: Choose the function used to compute the feed and kill rates. Options are "Constant", "Spatial Gradient", "Temporal Decay", and "Spatiotemporal Wave".

- **Directional Bias**: Apply a directional bias to the reaction-diffusion equation.

- **Threshold**: Set the threshold for the concentration of the second substance to form the structure.

- **Grid Size and Steps**: Set the size of the simulation grid and the number of steps to simulate.

- **Visualization Mode**: Choose the method for visualizing the structure. Options are "Voxel" (voxel-based representation), "Marching Cubes" (isosurface extraction), and "Metaball" (metaball-based representation).


## Usage

For the Jupyter notebooks, simply run the cells in order. They contain explanations of each step.

The Blender add-on adds a new panel in the 3D view. To generate a structure, adjust the parameters and click `Start Generation`. Switch to material preview mode to see the colors (top-right corner button or Shift + Z).

## References
1. A. Witkin, M. Kass, “Reaction-diffusion textures”, Proceedings of the 18th annual conference on Computer graphics and interactive techniques (SIGGRAPH ‘91), 1991, doi: 10.1145/122718.122750.
2. A. R. Sanderson, R. M. Kirby, C. R. Johnson, L. Yang, “Advanced Reaction-Diffusion Models for Texture Synthesis”, Journal of Graphics Tools, 2006, doi: 10.1080/2151237X.2006.10129222.