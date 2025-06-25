import os
import subprocess
import itk
import tempfile
import shutil

# Define paths to Elastix and Transformix binaries
elastix_path = r"path/to/elastix.exe"

def run_elastix(fixed_image_path, moving_image_path, parameter_files, output_dir, patient_id):
    """Runs Elastix to register two images and saves the result."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    command = [
        elastix_path,
        '-f', fixed_image_path,
        '-m', moving_image_path,
        '-out', output_dir
    ]
    
    # Add parameter files to the command list correctly
    for p_file in parameter_files:
        command += ['-p', p_file]

    print(f"Registering {moving_image_path} to {fixed_image_path}")
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print("Registration completed successfully.")
        
        # Assuming the primary result image is named 'result.0.nii'
        result_image_path = os.path.join(output_dir, "result.0.nii")
        final_result_image_path = os.path.join(output_dir, "result.1.nii")
        renamed_image_path = os.path.join(output_dir, f"{patient_id}.nii")
        
        # Rename or move the final result image
        if os.path.exists(final_result_image_path):
            shutil.move(final_result_image_path, renamed_image_path)
            print(f"Renamed result image to {renamed_image_path}")
        else:
            print("Expected result image not found. Check Elastix output.")
        
        # Delete the intermediary result image if it exists
        if os.path.exists(result_image_path):
            os.remove(result_image_path)
            print(f"Deleted intermediary result image: {result_image_path}")
        
    except subprocess.CalledProcessError as e:
        print(f"Registration failed: {e.stdout.decode('utf-8')}")

def resample_to_reference_and_register(fixed_image_path, moving_image_path, parameter_files, output_dir, patient_id, seq):
    """Resample moving image to match reference image and then register using Elastix."""
    PixelType = itk.F
    Dimension = 3
    ImageType = itk.Image[PixelType, Dimension]
    
    reference_image = itk.imread(fixed_image_path)
    reference_image = itk.cast_image_filter(reference_image, ttype=(type(reference_image), itk.Image[itk.F, 3]))

    moving_image = itk.imread(moving_image_path)
    moving_image = itk.cast_image_filter(moving_image, ttype=(type(moving_image), itk.Image[itk.F, 3]))

    # Extract parameters from reference image
    reference_spacing = reference_image.GetSpacing()
    reference_size = reference_image.GetLargestPossibleRegion().GetSize()
    reference_direction = reference_image.GetDirection()
    reference_origin = reference_image.GetOrigin()

    # Set up resampling
    ResampleFilterType = itk.ResampleImageFilter[type(moving_image), type(moving_image)]
    resampler = ResampleFilterType.New()
    resampler.SetReferenceImage(reference_image)
    resampler.SetSize(reference_size)
    resampler.SetOutputSpacing(reference_spacing)
    resampler.SetOutputDirection(reference_direction)
    resampler.SetOutputOrigin(reference_origin)
    
    # Use a linear interpolator
    interpolator = itk.LinearInterpolateImageFunction[type(moving_image), itk.D].New()
    resampler.SetInterpolator(interpolator)

    # Perform resampling
    resampler.SetInput(moving_image)
    resampler.Update()

    resampled_image = resampler.GetOutput()

    # Save the resampled image to a temporary file
    with tempfile.NamedTemporaryFile(suffix=f'_{seq}_{patient_id}.nii.gz', delete=True) as tmpfile:
        itk.imwrite(resampled_image, tmpfile.name)
        # Proceed with registration using Elastix
        run_elastix(fixed_image_path, tmpfile.name, parameter_files, output_dir, patient_id)

def process_patient_sequences(root_dir, output_dir, parameter_files):
    sequences = ['T1W', 'T1MDIXON_IP']
    patients = [f for f in os.listdir(os.path.join(root_dir, 'T1W')) if f.endswith('.nii.gz')]

    for patient_file in patients:
        patient_id = patient_file.split('.')[0]  # Assuming the format "Patient_X.nii.gz"

        # Register T1MDIXON and T2W to T1W
        t1w_image_path = os.path.join(root_dir, 'T1W', patient_file)
        for seq in ['T1MDIXON_IP']:
            seq_image_path = os.path.join(root_dir, seq, patient_file)
            if os.path.exists(seq_image_path):
                registration_output_dir = os.path.join(output_dir, seq, patient_id)
                resample_to_reference_and_register(t1w_image_path, seq_image_path, parameter_files, registration_output_dir, patient_id, seq)

        # Register DWI and T2MDIXON to registered T2W
        #registered_t2w_image_path = os.path.join(output_dir, 'T2W', patient_id, f"{patient_id}.nii.gz")  # Adjust based on actual output file from Elastix
        #for seq in ['DWI', 'T2MDIXON']:
            #seq_image_path = os.path.join(root_dir, seq, patient_file)
            #if os.path.exists(seq_image_path):
                #registration_output_dir = os.path.join(output_dir, seq, patient_id)
                #resample_to_reference_and_register(registered_t2w_image_path, seq_image_path, parameter_files, registration_output_dir, patient_id, seq)

        print(f"Completed registration for {patient_id}")

# Example usage:
root_dir = r'path\to\Dataset Before Registration'
output_dir = r'path\to\Output After Elastix Registration'
parameter_files = [r'path\to\Rigid.txt', r'path\to\Deformable.txt']
process_patient_sequences(root_dir, output_dir, parameter_files)