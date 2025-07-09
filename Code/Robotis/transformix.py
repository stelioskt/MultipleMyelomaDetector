import os
import subprocess

# Update these paths according to your Elastix installation and data directories
transformix_path = "path/to/transformix.exe"
images_dir = "path/to/desired/dataset channel/to be registered"  # Directory containing MRI images
parameters_root_dir = "path/to/transforming/parameters"  # Root directory containing parameter subdirectories
output_dir = "path/to/desired/output"  # Directory to save transformed images

def run_command(command):
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT)
        return output.decode('utf-8')
    except subprocess.CalledProcessError as e:
        print(f"Command '{e.cmd}' returned non-zero exit status {e.returncode}. Output: {e.output.decode('utf-8')}")
        raise

def apply_transform_with_transformix(input_image_path, transform_parameters_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    command = [
        transformix_path,
        '-in', input_image_path,
        '-tp', transform_parameters_path,
        '-out', output_dir
    ]

    print(f"Running Transformix to apply transform on {input_image_path}...")
    try:
        output = run_command(command)
        print("Transformix completed successfully.")
        print(output)
    except Exception as e:
        print(f"Error during image transformation with Transformix: {e}")
        raise

def auto_apply_transformations(images_dir, parameters_root_dir, output_dir):
    images = [f for f in os.listdir(images_dir) if f.endswith('.nii.gz')]

    for image in images:
        base_name = os.path.splitext(os.path.splitext(image)[0])[0]  # Remove .nii.gz
        patient_dir_name = f"Patient_{base_name.split('_')[-1]}"  # Assumes format "Patient_X"
        parameters_dir = os.path.join(parameters_root_dir, patient_dir_name)

        if os.path.exists(parameters_dir):
            parameter_file_path = os.path.join(parameters_dir, f"{base_name}.txt")
            if os.path.isfile(parameter_file_path):
                image_path = os.path.join(images_dir, image)
                transformed_output_dir = os.path.join(output_dir, base_name)
                apply_transform_with_transformix(image_path, parameter_file_path, transformed_output_dir)
            else:
                print(f"No parameter file found for {image} in {parameters_dir}")
        else:
            print(f"No parameters directory found for {image}")

# Example usage
auto_apply_transformations(images_dir, parameters_root_dir, output_dir)
