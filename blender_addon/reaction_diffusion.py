import bpy
import bmesh
import numpy as np
from scipy import signal
from skimage.measure import marching_cubes
import colorsys

def laplacian(Z):
    laplacian_kernel = np.array([
                                 [[0.15, 0.05, 0.15],
                                 [0.05, -0.8, 0.05],
                                 [0.15, 0.05, 0.15]], 
                                 [[.05, .2, .05],
                                 [.2, -1, .2],
                                 [.05, .2, .05]],
                                 [[0.15, 0.05, 0.15],
                                 [0.05, -0.8, 0.05],
                                 [0.15, 0.05, 0.15]],
                                 ])
    return signal.convolve(Z, laplacian_kernel, mode='same', method='direct')

def initialize_grid(N, U_mean=0.50, U_std=0.10, V_mean=0.25, V_std=0.10):
    # U = np.random.normal(U_mean, U_std, (N, N, N))
    # V = np.random.normal(V_mean, V_std, (N, N, N))
    U = np.ones((N, N, N))  # Initial condition for U
    V = np.zeros((N, N, N))  # Initial condition for V
    v_size = 2
    V[N//2-v_size:N//2+v_size, N//2-v_size:N//2+v_size, N//2-v_size:N//2+v_size] = 1
    return U, V

def create_bias_grid(bias, N):
    # Convert the bias to a numpy array
    bias_np = np.array(bias)

    # Check if the bias is zero and return a zero grid if so
    if np.linalg.norm(bias_np) == 0:
        return np.zeros((N, N, N))

    # Normalize the bias to get the growth direction
    direction = bias_np / np.linalg.norm(bias_np)

    # Create a gradient grid
    gradient = np.zeros((N, N, N, 3))  # 3 for the 3 components of the vector
    for i in range(N):
        for j in range(N):
            for k in range(N):
                gradient[i, j, k] = np.array([i/N, j/N, k/N]) * 2 - 1  # This line creates a gradient from -1 to 1

    # Project the gradient onto the growth direction
    projection = np.dot(gradient, direction)

    # Multiply by the magnitude of the bias
    scaled_projection = projection * np.linalg.norm(bias_np)

    return scaled_projection


def calculate_laplacian(U, V):
    # Calculate Laplacian
    deltaU = laplacian(U)
    deltaV = laplacian(V)

    return deltaU, deltaV

def update_concentrations(U, V, dt, Du, Dv, F, k, deltaU, deltaV):
    reaction = U * V * V
    U_new = U + dt * (Du * deltaU - reaction + F * (1 - U))
    V_new = V + dt * (Dv * deltaV + reaction - (F + k) * V)
    U_new = np.clip(U_new, 0, 1)
    V_new = np.clip(V_new, 0, 1)
    return U_new, V_new

def compute_constant(param_init, x, y, z, t):
    return param_init  # This returns the original parameter

def compute_spatial_gradient(param_init, x, y, z, t):
    center = max(x.max(), y.max(), z.max()) / 2.0
    distance_from_center = np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)
    gradient = distance_from_center / (center * 2)  # values between 0 and 1
    return param_init * (1 + gradient)  # Increase parameter based on the distance from center

def compute_temporal_decay(param_init, x, y, z, t):
    decay_rate = 0.01  # Adjust as needed
    return param_init * np.exp(-decay_rate * t)  # Parameter decreases as time passes

def compute_spatiotemporal_wave(param_init, x, y, z, t):
    wave_speed = 0.1  # Adjust as needed
    wave_amplitude = 0.1  # Adjust as needed
    distance_from_center = np.sqrt(x**2 + y**2 + z**2)
    wave = wave_amplitude * np.sin(wave_speed * (distance_from_center - t))
    return param_init * (1 + wave)  # Parameter changes in a wave pattern over time and space

def create_mesh_from_grid(value_grid, N, threshold, frame_start=1, mode='VOXEL'):
    value_grid = value_grid.astype(np.float32)
    threshold = np.float32(threshold)
    min_val, max_val = np.min(value_grid), np.max(value_grid)

    # Check if the grid_values are invalid
    if min_val == max_val and min_val == 0.0:
        # Return empty object if all values are 0.0
        print("Warning: All values are 0.0. Returning empty object.")
        return bpy.data.objects.new("RD_Empty_Object", None)
    
    if not min_val <= threshold <= max_val:
        print(f"Warning: Threshold ({threshold}) is out of range ({min_val}, {max_val})")
        if max_val - min_val < 0.0001:
            print(f"Value range is very small. Setting threshold to avg value. ({(min_val + max_val) / 2})")
            threshold = (min_val + max_val) / 2
        else:
            print(f"Setting to mean value. ({np.mean(value_grid)})")
            threshold = np.mean(value_grid)

    if mode == 'MBALL':
        obj = create_mesh_MBALL(value_grid, N, threshold)
    elif mode == 'VOXEL':
        obj = create_mesh_VOXEL(value_grid, N, threshold)
    elif mode == 'MC':
        obj = create_mesh_MC(value_grid, N, threshold)

    apply_vertex_colors(obj.data, value_grid, N)
    add_vertex_color_material(obj)

    return animate_object(obj, frame_start)


def create_mesh_MBALL(value_grid, N, threshold):
    # Create Metaball object
    mb = bpy.data.metaballs.new("MBall")
    mb.resolution = 0.2  # Increase resolution for a more detailed mesh
    mb.threshold = 0.6  # Adjust influence threshold as necessary
    obj = bpy.data.objects.new("RD_Object", mb)
    obj.location = (0, 0, 0)

    # Create Metaball elements
    for i in range(N):
        for j in range(N):
            for k in range(N):
                if value_grid[i, j, k] > threshold:
                    element = mb.elements.new(type='BALL')
                    element.co = ((i-N/2)*1.2, (j-N/2)*1.2, (k-N/2)*1.2)
                    element.radius = 1.5 * value_grid[i, j, k]  # Adjust radius as necessary

    # Link the object to the scene's collection
    bpy.context.collection.objects.link(obj)

    # Convert Metaball to mesh
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.convert(target='MESH')

    # Update 'obj' to refer to the newly created mesh object
    obj = bpy.context.view_layer.objects.active

    return obj



def create_mesh_VOXEL(value_grid, N, threshold=0.5):
    # Create a new mesh object
    mesh = bpy.data.meshes.new(name="RD_Mesh")
    obj = bpy.data.objects.new(name="RD_Object", object_data=mesh)

    # Create a BMesh instance and a vertex color layer
    bm = bmesh.new()

    for i in range(N):
        for j in range(N):
            for k in range(N):
                # If the concentration of U exceeds the threshold, add a cube
                if value_grid[i, j, k] > threshold:
                    # Create the cube
                    bmesh.ops.create_cube(bm, size=1.0)
                    # Move it to the correct position
                    bmesh.ops.translate(bm, verts=bm.verts[-8:], vec=((i-N/2)*1.2, (j-N/2)*1.2, (k-N/2)*1.2))

    # Update the mesh with the new geometry
    bm.to_mesh(mesh)
    bm.free()

    return obj


def create_mesh_MC(value_grid, N, threshold):
    # Create a new mesh object
    mesh = bpy.data.meshes.new(name="MC_Mesh")
    obj = bpy.data.objects.new(name="MC_Object", object_data=mesh)
    
    # Use marching cubes to obtain the surface mesh vertices and indices
    verts, faces, _, _ = marching_cubes(value_grid, threshold)

    # Create a BMesh instance and add the vertices
    bm = bmesh.new()
    vert_bm = [bm.verts.new(vert) for vert in verts]

    # Loop through the faces and add them to the BMesh
    for face in faces:
        bm.faces.new([vert_bm[i] for i in face])

    # Update the mesh with the new geometry and free the BMesh
    bm.to_mesh(mesh)
    bm.free()

    # Center the object
    avg_x = sum(vert.co.x for vert in mesh.vertices) / len(mesh.vertices)
    avg_y = sum(vert.co.y for vert in mesh.vertices) / len(mesh.vertices)
    avg_z = sum(vert.co.z for vert in mesh.vertices) / len(mesh.vertices)
    for vert in mesh.vertices:
        vert.co.x -= avg_x
        vert.co.y -= avg_y
        vert.co.z -= avg_z

    # Link the object to the scene's collection
    bpy.context.collection.objects.link(obj)

    # Set smooth shading
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.shade_smooth()

    return obj



def animate_object(obj, frame_start=1):
    # Animate the object
    # Hide the object and make it non-renderable
    obj.hide_render = True
    obj.hide_viewport = True

    # Insert keyframes for hide_render and hide_viewport
    obj.keyframe_insert(data_path="hide_render", frame=frame_start)
    obj.keyframe_insert(data_path="hide_viewport", frame=frame_start)

    obj.hide_render = False
    obj.hide_viewport = False

    # Insert keyframes for hide_render and hide_viewport
    obj.keyframe_insert(data_path="hide_render", frame=frame_start+1)
    obj.keyframe_insert(data_path="hide_viewport", frame=frame_start+1)

    # After the next frame, hide the object again
    obj.hide_render = True
    obj.hide_viewport = True

    obj.keyframe_insert(data_path="hide_render", frame=frame_start+2)
    obj.keyframe_insert(data_path="hide_viewport", frame=frame_start+2)

    # Return the created object
    return obj


def apply_vertex_colors(mesh, value_grid, N):
    """Apply vertex colors to a mesh based on the value grid."""

    # Create a new vertex color layer
    mesh.vertex_colors.new()
    color_layer = mesh.vertex_colors.active

    # Find the bounding box of the mesh
    min_x = min(vert.co.x for vert in mesh.vertices)
    max_x = max(vert.co.x for vert in mesh.vertices)
    min_y = min(vert.co.y for vert in mesh.vertices)
    max_y = max(vert.co.y for vert in mesh.vertices)
    min_z = min(vert.co.z for vert in mesh.vertices)
    max_z = max(vert.co.z for vert in mesh.vertices)

    # Compute the scaling factors for each dimension
    scale_x = (N - 1) / (max_x - min_x)
    scale_y = (N - 1) / (max_y - min_y)
    scale_z = (N - 1) / (max_z - min_z)

    # Set the color of each vertex based on the value_grid
    for poly in mesh.polygons:
        for idx in poly.loop_indices:
            loop = mesh.loops[idx]
            vertex = mesh.vertices[loop.vertex_index]
            coords = vertex.co

            # Map vertex coordinates to indices in the value_grid
            i = int((coords.x - min_x) * scale_x)
            j = int((coords.y - min_y) * scale_y)
            k = int((coords.z - min_z) * scale_z)

            if 0 <= i < N and 0 <= j < N and 0 <= k < N:
                value = value_grid[i][j][k]
                hue = value  # The hue should be between 0.0 and 1.0
                color = colorsys.hls_to_rgb(hue, 0.5, 1.0)  # Convert the HLS color to RGB
                color = color + (1.0,)  # Add alpha channel
                color_layer.data[loop.index].color = color




def add_vertex_color_material(obj):
    """Add a material that uses vertex colors to an object."""
    mat = bpy.data.materials.new(name="VertexColorMaterial")

    # Enable 'Use Nodes'
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]

    # Add a new 'Vertex Color' node
    color_node = mat.node_tree.nodes.new(type='ShaderNodeVertexColor')

    # Link the 'Color' output of the 'Vertex Color' node to the 'Base Color' input of the 'Principled BSDF' node
    mat.node_tree.links.new(bsdf.inputs["Base Color"], color_node.outputs["Color"])

    # Assign material to object
    if obj.data.materials:
        # replace the first material slot
        obj.data.materials[0] = mat
    else:
        # no slots
        obj.data.materials.append(mat)




