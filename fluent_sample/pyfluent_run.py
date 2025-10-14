#-------------------------------------------------------------------------------
# Paper: Sensitivity analysis and uncertainty propagation of the time to onset
#         of natural circulation in air ingress accidents
# Journal: Nuclear Engineering and Design
# Corresponding Authors: Meredith Eaheart (eaheart@umich.edu)
#                        Majdi Radaideh (radaideh@umich.edu)
# Script: Pyfluent runner script to create perturbed ONC cases
# Date: September 24, 2025
#-------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import ansys.fluent.core as pyfluent
import os 
import warnings
import cv2
import subprocess

fluent_path = "/opt/ohpc/ansys_inc/v242/fluent/bin/fluent"    #change to fluent directory in your machine
case_file_path = "base.cas.h5"                                #base case path

warnings.filterwarnings("ignore")

# Update environment variable for Linux path
os.environ["AWP_ROOT"] = "/opt/ohpc/ansys_inc/v242/Framework"

# Check if the Fluent executable exists
if not os.path.exists(fluent_path):
    raise FileNotFoundError(f"Fluent executable not found at {fluent_path}")

# Launch Fluent solver
solver = pyfluent.launch_fluent(show_gui=False, mode="solver", processor_count=None, fluent_path=fluent_path)

# Check the health of the solver
solver.health_check.is_serving
solver.file.read(file_type="case", file_name=case_file_path)
solver.tui.solve.set.transient_controls.fixed_user_specified("Yes")
solver.tui.solve.set.transient_controls.number_of_time_steps(10)
solver.tui.solve.set.transient_controls.time_step_size(0.00001)
solver.solution.initialization.standard_initialize()

# Change simulation parameters below

#1. Heated section temperature
#Define temperature of heated section
#Range: 375C - 1000C (must convert to kelvin... 648.15K - 1273.15K)
heated_section_temperature = 1107.26 #K

#Commands to apply temperature to the heated section
thermal_state = solver.setup.boundary_conditions.wall['heated_wall'].thermal.get_state()
thermal_state['temperature']['value'] = heated_section_temperature  #K
solver.setup.boundary_conditions.wall['heated_wall'].thermal.set_state(thermal_state)
print(solver.setup.boundary_conditions.wall['heated_wall'].thermal.get_state())

#2. Unheated sections' heat transfer coefficient
#Define htc of unheated sections
#Range: 0.1 W/m^2K - 10 W/m^2K
htc = 9.51207 #W/m^2K

#Commands to change htc
unheated_wall_names = ['unheated_wall-fff_left_unheated', 'unheated_wall-fff_unheated_section']

for wall_name in unheated_wall_names:
    wall_state = solver.setup.boundary_conditions.wall[wall_name].thermal.get_state()
    wall_state['heat_transfer_coeff']['value'] = htc
    solver.setup.boundary_conditions.wall[wall_name].thermal.set_state(wall_state)


#3. Wall thickness
#Range: 0.001m - 0.004m (can extend if interested)
glass_thickness = 0.00319598  #m

for wall_name in unheated_wall_names:
    wall_state = solver.setup.boundary_conditions.wall[wall_name].get_state()
    wall_state['thermal']['wall_thickness']['value'] = glass_thickness
    solver.setup.boundary_conditions.wall[wall_name].set_state(wall_state)
print("Updated Wall Thickness for Unheated Walls.")


#4. Wall thermal conductivity
#Range: either starting from a lower bound of 25C or 375C
#25C - 1000C = 1.3 W/mK - 3.7 W/mK
#375C - 1000C = 1.8 W/mK - 3.7 W/mK
wall_k = 2.47759  #W/m-K

wall_mat_state = solver.setup.materials.solid['glass'].get_state()
wall_mat_state['thermal_conductivity']['value'] = wall_k
solver.setup.materials.solid['glass'].set_state(wall_mat_state)
print("Updated Wall Thermal Conductivity:", solver.setup.materials.solid['glass'].get_state())

#5. Helium thermal conductivity
#Range: either starting from a lower bound of 25C or 375C
#25C - 1000C = 0.24364 W/mK - 0.42706 W/mK (took the closest value to each temp from the NIST table)
#375C - 1000C = 0.26648 W/mK - 0.42706 W/mK
he_k = 0.174031  #W/m-K
helium_state = solver.setup.materials.fluid['helium'].get_state()
helium_state['thermal_conductivity']['value'] = he_k
solver.setup.materials.fluid['helium'].set_state(helium_state)
print("Updated Helium Thermal Conductivity:", solver.setup.materials.fluid['helium'].get_state())

#6. Helium viscosity
#Range: either starting from a lower bound of 25C or 375C
#25C - 1000C = 0.000031092 kg/ms - 0.000054783 kg/ms
#375C - 1000C = 0.000034022 kg/ms - 0.000054783 kg/ms
he_v = 5.59195e-05  #kg/m-s
helium_state = solver.setup.materials.fluid['helium'].get_state()
helium_state['viscosity']['value'] = he_v
solver.setup.materials.fluid['helium'].set_state(helium_state)
print("Updated Helium Viscosity:", solver.setup.materials.fluid['helium'].get_state())

#7. Air thermal conductivity
#Range: either starting from a lower bound of 25C or 375C
#25C - 1000C = 0.02551 W/mK - 0.07577 W/mK
#375C - 1000C = 0.051493825 W/mK - 0.07577 W/mK
air_k = 0.0347167  #W/m-K
air_state = solver.setup.materials.fluid['air'].get_state()
air_state['thermal_conductivity']['value'] = air_k
solver.setup.materials.fluid['air'].set_state(air_state)
print("Updated Air Thermal Conductivity:", solver.setup.materials.fluid['air'].get_state())

#8. Air viscosity
#Range: either starting from a lower bound of 25C or 375C
#25C - 1000C = 0.0000185 W/mK - 0.0000467 W/mK
#375C - 1000C = 0.0000332 W/mK - 0.0000467 W/mK
air_v = 2.36634e-05  #kg/m-s
air_state = solver.setup.materials.fluid['air'].get_state()
air_state['viscosity']['value'] = air_v
solver.setup.materials.fluid['air'].set_state(air_state)
print("Updated Air Viscosity:", solver.setup.materials.fluid['air'].get_state())

#Save the updated case file to check
solver.file.write_case(file_name="perturbed_case.cas.h5")

#---------------------------------------------
# End of parameter perturbation
#---------------------------------------------

#---------------------------------------------
# Open the Patch panel
#---------------------------------------------
#assign helium and air to different regions in the geometry
solver.solution.initialization.patch.calculate_patch( 
     cell_zones = ["fff_opening"],
     variable = "species-0",
     value = 0.0
 )
solver.solution.initialization.patch.calculate_patch(
     cell_zones = ["fff_heated_section", "fff_unheated_section", "fff_left_unheated"],
     variable = "species-0",
     value = 1.0
 )

#---------------------------------------------
# Define time steps
#---------------------------------------------
solver.tui.solve.set.transient_controls.number_of_time_steps(200)
solver.tui.solve.set.transient_controls.time_step_size(0.01)
solver.solution.run_calculation.calculate()

solver.tui.solve.set.transient_controls.number_of_time_steps(50)
solver.tui.solve.set.transient_controls.time_step_size(0.02)
solver.solution.run_calculation.calculate()

solver.tui.solve.set.transient_controls.number_of_time_steps(970)
solver.tui.solve.set.transient_controls.time_step_size(0.1)
solver.solution.run_calculation.calculate()

solver.tui.solve.set.transient_controls.number_of_time_steps(29500)
solver.tui.solve.set.transient_controls.time_step_size(0.2)
solver.solution.run_calculation.calculate()

solver.exit()

#---------------------------------------------
#Generate video of animations
#---------------------------------------------
def generate_video(image_folder, prefix_name, video_name, fps=30):
    """
    Generates a video from images in a specified folder that start with a given prefix.

    Parameters:
        image_folder (str): Path to the folder containing images.
        prefix_name (str): Prefix to filter image files.
        video_name (str): Output video file name.
        fps (int): Frames per second (default is 30).
    """
    # Get the list of images sorted by filename
    images = sorted([img for img in os.listdir(image_folder) 
                     if img.startswith(prefix_name) and (img.endswith(".png") or img.endswith(".jpeg"))])
   
    if not images:
        print("No matching images found. Please check the prefix or folder path.")
        return

    # Read the first image to get dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'XVID' for AVI alternative
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    # Iterate over images and write to video
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: Could not read {img_path}, skipping.")
            continue
        video.write(frame)

    # Release the video writer
    video.release()
    cv2.destroyAllWindows()

    print(f'Video saved as {video_name}')

#generate animations based on the frames and then delete the images
generate_video(image_folder=os.getcwd(), prefix_name="animation-1", video_name="temperature.mp4")
generate_video(image_folder=os.getcwd(), prefix_name="animation-2", video_name="he.mp4")
generate_video(image_folder=os.getcwd(), prefix_name="animation-3", video_name="velocity.mp4")
generate_video(image_folder=os.getcwd(), prefix_name="animation-4", video_name="pressure.mp4")
generate_video(image_folder=os.getcwd(), prefix_name="reynolds-5", video_name="reynolds.mp4")
generate_video(image_folder=os.getcwd(), prefix_name="turbdissipation-6", video_name="turbdissipation.mp4")
generate_video(image_folder=os.getcwd(), prefix_name="turbke-7", video_name="turbke.mp4")
generate_video(image_folder=os.getcwd(), prefix_name="turbviscosity-8", video_name="turbviscosity.mp4")


# Run the Linux command after video generation to delete all image files
subprocess.run(f'rm -f *.jpeg *.cxa', shell=True)

#---------------------------------------------
# Create directories if they don't exist to move data files to them
#---------------------------------------------
directories = ['symdata', 'alldata']
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")


# Move files using Linux shell commands
subprocess.run('mv symmetry-* symdata/', shell=True)
subprocess.run('mv fulldata-* alldata/', shell=True)
