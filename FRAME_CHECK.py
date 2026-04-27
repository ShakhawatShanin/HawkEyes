import os
from pathlib import Path
from ultralytics import YOLO
import cv2

def run_yolo_segmentation_with_visualization(model_path, input_path):
    """
    Run YOLO segmentation prediction and save visualized outputs.

    Parameters:
        model_path (str): Path to the YOLO model file.
        input_path (str): Path to the root folder or a specific subfolder containing images.
    """
    # Load YOLO model
    model = YOLO(model_path)

    input_path = Path(input_path)

    # Determine if input_path is a root folder or a specific subfolder
    if input_path.is_dir():
        folders_to_process = [input_path] if any(input_path.glob("*.jpg")) else list(input_path.iterdir())
    else:
        raise ValueError("The provided path must be a directory.")

    # Iterate through the folders
    for folder in folders_to_process:
        if folder.is_dir():
            input_folder = folder
            output_folder = input_folder.parent / f"{input_folder.name} SEGMENT"
            
            # Create output folder if it doesn't exist
            output_folder.mkdir(parents=True, exist_ok=True)
            
            # Iterate through image files in the current folder
            for file in os.listdir(input_folder):
                file_path = input_folder / file
                
                if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    print(f"Processing image: {file_path}")
                    
                    # Run prediction
                    results = model.predict(source=str(file_path))
                    
                    # Extract the annotated image
                    annotated_image = results[0].plot()
                    
                    # Save the visualized output
                    save_path = output_folder / file
                    cv2.imwrite(str(save_path), annotated_image)
                    print(f"Saved visualized predictions to: {save_path}")

if __name__ == "__main__":
    model_path = r"/home/shanin/Desktop/TRAINING/FRAME/runs/segment/train2/weights/best.pt"  # Replace with your YOLO model path
    input_path = r"/home/shanin/Downloads/dataset"  # Replace with root or specific subfolder path
    # python "D:\HawkEyes Projects\FRAME_CHECK.py"

    run_yolo_segmentation_with_visualization(model_path, input_path)
