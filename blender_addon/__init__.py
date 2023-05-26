bl_info = {
    "name": "Reaction-Diffusion Generator",
    "description": "Generate 3D structures based on the Reaction-Diffusion model",
    "author": "Tim Vučina",
    "version": (1, 0),
    "blender": (3, 5, 1),
    "location": "View3D > Tools",
    "category": "Tools"
}

import bpy
import numpy as np
import time
from .reaction_diffusion import *

compute_funcs = {
    'CONSTANT': compute_constant,
    'SPATIAL_GRADIENT': compute_spatial_gradient,
    'TEMPORAL_DECAY': compute_temporal_decay,
    'SPATIOTEMPORAL_WAVE': compute_spatiotemporal_wave,
}

# This function is called whenever a parameter changes
def update_params(self, context):
    if not self.updating_preset:
        self.preset = 'CUSTOM'

# This function is called when a preset is selected
def update_preset(self, context):
    preset = self.preset
    self.updating_preset = True
    if preset == 'OPENING_CORAL':
        self.dt = 0.1
        self.Du = 0.15
        self.Dv = 0.2
        self.compute_F_function = 'CONSTANT'
        self.F = 0.055
        self.compute_k_function = 'CONSTANT'
        self.k = 0.062
        self.bias = (0.0, 0.0, 3.0)
        self.threshold = 0.25
    elif preset == 'ALIEN_BRAIN':
        self.dt = 0.2
        self.Du = 0.16
        self.Dv = 0.21
        self.compute_F_function = 'TEMPORAL_DECAY'
        self.F = 0.051
        self.compute_k_function = 'TEMPORAL_DECAY'
        self.k = 0.065
        self.bias = (0.0, 0.2, 0.2)
        self.threshold = 0.3
    elif preset == 'WASP_NEST':
        self.dt = 0.4
        self.Du = 0.864
        self.Dv = 0.061
        self.compute_F_function = 'CONSTANT'
        self.F = 0.054
        self.compute_k_function = 'SPATIOTEMPORAL_WAVE'
        self.k = 0.061
        self.bias = (0.0, 0.0, 1.0)
        self.threshold = 0.1
    elif preset == 'RADIO_SNOWFLAKES':
        self.dt = 0.1
        self.Du = 0.18
        self.Dv = 0.31
        self.compute_F_function = 'SPATIAL_GRADIENT'
        self.F = 0.072
        self.compute_k_function = 'SPATIAL_GRADIENT'
        self.k = 0.046
        self.bias = (0.0, 0.0, 0.0)
        self.threshold = 0.27
    elif preset == 'PSYCHADELIC_GROWTH':
        self.dt = 0.2
        self.Du = 0.14
        self.Dv = 0.18
        self.compute_F_function = 'TEMPORAL_DECAY'
        self.F = 0.056
        self.compute_k_function = 'CONSTANT'
        self.k = 0.064
        self.bias = (0.0, 0.0, 0.0)
        self.threshold = 0.3
    self.updating_preset = False

class RDSettings(bpy.types.PropertyGroup):
    preset : bpy.props.EnumProperty(
        name="Presets",
        description="Choose a preset",
        items=[
            ('OPENING_CORAL', "Opening Coral", "Produces a coral-like shell that slowly opens its top"),
            ('ALIEN_BRAIN', "Alien Brain", "Produces a brain-like structure with a spiky surface"),
            ('WASP_NEST', "Wasp Nest", "Forms a wasp nest-like structure"),
            ('RADIO_SNOWFLAKES', "Radio Snowflakes", "Forms snowflake-like structures, reminiscent of radio waves"),
            ('PSYCHADELIC_GROWTH', "Psychadelic Growth", "Creates complex, interconnected mass reminiscent of festival visuals"),
            ('CUSTOM', "Custom", "Custom settings")
        ],
        default='CUSTOM',
        update=update_preset
    )
    dt: bpy.props.FloatProperty(name="Time Step", default=0.1, update=update_params)
    Du: bpy.props.FloatProperty(name="Diffusion Rate (Du)", default=0.13, update=update_params)
    Dv: bpy.props.FloatProperty(name="Diffusion Rate (Dv)", default=0.17, update=update_params)
    compute_F_function: bpy.props.EnumProperty(
        name="Feed Function",
        items=[('CONSTANT', "Constant", "Constant feed value"),
           ('SPATIAL_GRADIENT', "Spatial Gradient", "Feed changes with distance from center"),
           ('TEMPORAL_DECAY', "Temporal Decay", "Feed decays over time"),
           ('SPATIOTEMPORAL_WAVE', "Spatiotemporal Wave", "Feed changes in a wave pattern over time and space")],
        default='CONSTANT', update=update_params
    )
    F: bpy.props.FloatProperty(name="Feed", default=0.0545, update=update_params)
    compute_k_function: bpy.props.EnumProperty(
        name="Kill Function",
        items=[('CONSTANT', "Constant", "Constant kill value"),
           ('SPATIAL_GRADIENT', "Spatial Gradient", "Kill changes with distance from center"),
           ('TEMPORAL_DECAY', "Temporal Decay", "Kill decays over time"),
           ('SPATIOTEMPORAL_WAVE', "Spatiotemporal Wave", "Kill changes in a wave pattern over time and space")],
        default='CONSTANT', update=update_params
    )
    k: bpy.props.FloatProperty(name="Kill", default=0.062, update=update_params)
    bias: bpy.props.FloatVectorProperty(name="Directional Bias", default=(0.0, 0.0, 0.0), subtype='XYZ', update=update_params)
    threshold: bpy.props.FloatProperty(name="Threshold", default=0.2, min=0.0, max=1.0, update=update_params)
    N: bpy.props.IntProperty(name="Grid Size", default=30, min=10, max=500)
    steps: bpy.props.IntProperty(name="Steps", default=300, min=100, max=10000)
    visualization_mode: bpy.props.EnumProperty(
        name="Mode",
        items=[('VOXEL', "Voxel", "Voxel-based representation"),
               ('MC', "Marching Cubes", "Isosurface extraction"),
               ('MBALL', "Metaball", "Metaball-based representation")],
        default='MC'
    )
    generating: bpy.props.BoolProperty(name="Generating", default=False)
    updating_preset: bpy.props.BoolProperty(default=False)

class OBJECT_PT_rdpanel(bpy.types.Panel):
    bl_label = "Reaction-Diffusion Generator"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Tools"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        rd_tool = scene.rd_tool
        
        layout.prop(rd_tool, "preset")
        # Create a box for the time and diffusion settings
        box = layout.box()
        box.label(text="Time and Diffusion Settings")
        box.prop(rd_tool, "dt")
        box.prop(rd_tool, "Du")
        box.prop(rd_tool, "Dv")

        # Create a box for the feed and kill settings
        box = layout.box()
        box.label(text="Feed and Kill Settings")
        box.prop(rd_tool, "compute_F_function")
        box.prop(rd_tool, "F")
        box.prop(rd_tool, "compute_k_function")
        box.prop(rd_tool, "k")

        # Create a box for the grid and steps settings
        box = layout.box()
        box.label(text="Grid and Steps Settings")
        box.prop(rd_tool, "N")
        box.prop(rd_tool, "steps")

        # Create a box for the bias and threshold settings
        box = layout.box()
        box.label(text="Bias and Threshold Settings")
        box.prop(rd_tool, "bias")
        box.prop(rd_tool, "threshold")

        # Create a box for the visualization mode setting
        box = layout.box()
        box.label(text="Visualization Mode Setting")
        box.prop(rd_tool, "visualization_mode")

        # Add the progress bar and the operator button
        layout.prop(scene, "my_progress", slider=True)
        if context.scene.rd_tool.generating:
            layout.operator(OBJECT_OT_cancel_generation.bl_idname, text='Stop Generation')
        else:
            layout.operator(OBJECT_OT_generate_structure.bl_idname, text='Start Generation')

def register():
    bpy.utils.register_class(RDSettings)
    bpy.types.Scene.rd_tool = bpy.props.PointerProperty(type=RDSettings)
    bpy.utils.register_class(OBJECT_OT_generate_structure)
    bpy.utils.register_class(OBJECT_PT_rdpanel)
    bpy.types.Scene.my_progress = bpy.props.FloatProperty(
        name="Progress",
        subtype='PERCENTAGE',
        default=0.0,
        min=0.0,
        max=100.0,
        description="Task progress",
    )
    bpy.utils.register_class(OBJECT_OT_cancel_generation)

def unregister():
    bpy.utils.unregister_class(OBJECT_PT_rdpanel)
    bpy.utils.unregister_class(OBJECT_OT_generate_structure)
    del bpy.types.Scene.rd_tool
    bpy.utils.unregister_class(RDSettings)
    del bpy.types.Scene.my_progress
    bpy.utils.unregister_class(OBJECT_OT_cancel_generation)  


class OBJECT_OT_generate_structure(bpy.types.Operator):
    bl_idname = "object.generate_structure"
    bl_label = "Generate Structure"
    bl_description = "Generate a 3D structure using the Reaction-Diffusion model"
    
    _timer = None
    _progress = 0
    _step_chunk = 10  # Process this many steps at a time

    def modal(self, context, event):
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            self.cancel(context)
            return {'CANCELLED'}

        if event.type == 'TIMER' and context.scene.rd_tool.generating:
            for _ in range(self._step_chunk):  # Process multiple steps at once
                # Apply the bias to U before using it
                self.U += self.bias_grid

                # Compute the Laplacian of u and v
                deltaU, deltaV = calculate_laplacian(self.U, self.V)

                # Compute the new F and k values
                x, y, z = np.indices((self.N, self.N, self.N))
                self.F = compute_funcs[self.compute_F_function](self.F, x, y, z, self._progress)
                self.k = compute_funcs[self.compute_k_function](self.k, x, y, z, self._progress)

                # Update u and v according to the reaction-diffusion equations
                self.U, self.V = update_concentrations(self.U, self.V, self.dt, self.Du, self.Dv, self.F, self.k, deltaU, deltaV)

                # Create a new mesh at this step
                obj = create_mesh_from_grid(self.V, self.N, self.threshold, frame_start=self._progress+1, mode=self.visualization_mode)

                # Unlink the object from its original collection if it is in the collection
                if obj.name in bpy.context.collection.objects:
                    bpy.context.collection.objects.unlink(obj)
                    
                # Link the new object to the new collection
                self.new_collection.objects.link(obj)

                self._progress += 1

                if self._progress >= self.steps:
                    # Set the start and end frames
                    bpy.context.scene.frame_start = 1
                    bpy.context.scene.frame_end = self.steps

                    # Reset progress
                    context.scene.my_progress = 0
                    
                    self.cancel(context)
                    return {'FINISHED'}

            
            # Update progress
            context.scene.my_progress = (self._progress / self.steps) * 100
            return {'PASS_THROUGH'}  # Allow the UI to update

        return {'RUNNING_MODAL'}

    def execute(self, context):
        context.window_manager.modal_handler_add(self)
        self._timer = context.window_manager.event_timer_add(0.1, window=context.window)

        # Access the Reaction-Diffusion parameters
        rd_tool = context.scene.rd_tool
        self.dt = rd_tool.dt
        self.Du = rd_tool.Du
        self.Dv = rd_tool.Dv
        self.F = rd_tool.F
        self.k = rd_tool.k
        self.N = rd_tool.N
        self.steps = rd_tool.steps
        self.threshold = rd_tool.threshold
        self.bias = rd_tool.bias
        self.bias_grid = create_bias_grid(self.bias, self.N)
        self.visualization_mode = rd_tool.visualization_mode
        self.compute_F_function = rd_tool.compute_F_function
        self.compute_k_function = rd_tool.compute_k_function

        # Initialize the grid
        self.U, self.V = initialize_grid(self.N)

        # Create a new collection for this run
        unique_name = f"{self.visualization_mode}_Run_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
        self.new_collection = bpy.data.collections.new(name=unique_name)
        bpy.context.scene.collection.children.link(self.new_collection)

        # Set generating state
        context.scene.rd_tool.generating = True
        
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        context.window_manager.event_timer_remove(self._timer)
        # Reset generating state
        context.scene.rd_tool.generating = False

class OBJECT_OT_cancel_generation(bpy.types.Operator):
    bl_idname = "object.cancel_generation"
    bl_label = "Cancel Generation"
    bl_description = "Cancel the ongoing structure generation process"

    def execute(self, context):
        # Get the modal operator and cancel it
        for window in bpy.context.window_manager.windows:
            screen = window.screen

            for area in screen.areas:
                if area.type == 'VIEW_3D':
                    for region in area.regions:
                        if region.type == 'WINDOW':
                            override = {'window': window, 'screen': screen, 'area': area, 'region': region}
                            bpy.ops.object.generate_structure('INVOKE_DEFAULT', True)

                            # Cancel the operator
                            bpy.ops.object.generate_structure('CANCEL', override)
        
        # Reset generating state
        context.scene.rd_tool.generating = False
        
        return {'FINISHED'}


if __name__ == "__main__":
    register()
