from ultralytics import YOLO
import cv2
import numpy as np
import os
import asyncio
from PIL import Image, ImageOps
import sys
import ast  # for safely evaluating strings like None or list

# -------------------------------
# Configuration
# -------------------------------
EXPAND_RATIO = 0.1        # Inner padding
OUTER_PADDING_SIZE = 5    # Outer padding size
seg_conf = 0.70
detect_conf = 0.30        # Detection confidence threshold

# -------------------------------
# Wrapper for YOLO model
# -------------------------------
class ModelsBAT:
    def __init__(self, model_path):
        self.selectMODEL = YOLO(model_path)

# -------------------------------
# Crop model
# -------------------------------
class CropModel:
    def __init__(self, model):
        self.model = model
        self.conf_threshold = seg_conf

    async def image_prediction_mask(self, image):
        predict = self.model.predict(image)

        if not predict or not predict[0].masks:
            print(f"No detection found for image.")
            return None, "Not Found"

        mask_tensor = predict[0].masks.data[0]
        mask = (mask_tensor.cpu().numpy() * 255).astype("uint8") if mask_tensor.is_cuda else (mask_tensor.numpy() * 255).astype("uint8")
        mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))

        class_ids_tensor = predict[0].boxes.cls
        class_ids = class_ids_tensor.cpu().numpy() if class_ids_tensor.is_cuda else class_ids_tensor.numpy()
        class_names = [self.model.names[int(cls_id)] for cls_id in class_ids]
        class_name = class_names[0] if class_names else "Not Found"

        return mask_resized, class_name

    async def get_mask_corner_points(self, mask):
        _, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            return None

        cnt = max(contours, key=cv2.contourArea)
        cnt_approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)

        return cnt_approx.reshape((4, 2)) if len(cnt_approx) == 4 else None

    async def get_order_points(self, points):
        rect = np.zeros((4, 2), dtype="float32")
        s = points.sum(axis=1)
        diff = np.diff(points, axis=1)

        rect[0] = points[np.argmin(s)]
        rect[2] = points[np.argmax(s)]
        rect[1] = points[np.argmin(diff)]
        rect[3] = points[np.argmax(diff)]

        return rect

    async def expand_bounding_box(self, points, expand_ratio):
        center = np.mean(points, axis=0)
        expanded_points = points + (points - center) * expand_ratio
        return expanded_points.astype("float32")

    async def point_transform(self, image, points):
        ordered_points = await self.get_order_points(points)
        expanded_points = await self.expand_bounding_box(ordered_points, EXPAND_RATIO)

        (tl, tr, br, bl) = expanded_points
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        max_width = max(int(widthA), int(widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        max_height = max(int(heightA), int(heightB))

        dst = np.array([[0, 0], [max_width - 1, 0],
                        [max_width - 1, max_height - 1],
                        [0, max_height - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(expanded_points, dst)
        warped_image = cv2.warpPerspective(image, M, (max_width, max_height))
        return warped_image

    async def add_padding_to_image(self, image, padding_size):
        return cv2.copyMakeBorder(image, padding_size, padding_size,
                                  padding_size, padding_size,
                                  cv2.BORDER_CONSTANT, value=[0, 0, 0])

    async def get_predicted_warped_image(self, image):
        mask, class_name = await self.image_prediction_mask(image)
        if mask is None:
            return None, class_name

        corner_points = await self.get_mask_corner_points(mask)
        if corner_points is None:
            return None, class_name

        warped_image = await self.point_transform(image, corner_points)
        padded_image = await self.add_padding_to_image(warped_image, OUTER_PADDING_SIZE)
        return padded_image, class_name

# -------------------------------
# Initial padding using PIL
# -------------------------------
def pad_image_first(image, padding):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    padded = ImageOps.expand(pil_image, border=padding, fill='black')
    padded_cv = cv2.cvtColor(np.array(padded), cv2.COLOR_RGB2BGR)
    return padded_cv

# -------------------------------
# Detection helper
# -------------------------------
async def run_detection(detect_class, image, desired_class_names,
                        detect_folder, image_file, show_full_annotation):
    if desired_class_names is None:
        classes = None
    else:
        if isinstance(desired_class_names, str):
            desired_class_names = [desired_class_names]
        classes = []
        for class_name in desired_class_names:
            for i, name in detect_class.selectMODEL.names.items():
                if name == class_name:
                    classes.append(i)
        if not classes:
            print(f"⚠️ No matching class found in model for: {desired_class_names}")
            return

    detect_predictions = detect_class.selectMODEL.predict(
        image,
        conf=detect_conf,
        classes=classes
    )

    detect_save_path = os.path.join(
        detect_folder, f"{os.path.splitext(image_file)[0]}.jpg"
    )
    print(f"Saving detection result to: {detect_save_path}")
    detect_predictions[0].save(detect_save_path, labels=show_full_annotation, conf=show_full_annotation)

# -------------------------------
# Recursive processing
# -------------------------------
async def process_images_recursive(folder_path, crop_class=None, detect_class=None,
                                   desired_class_names=None, show_full_annotation=True,
                                   initial_padding=0):
    for root, _, files in os.walk(folder_path):
        for image_file in files:
            image_path = os.path.join(root, image_file)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Not readable - {image_file}")
                continue

            # --- ✅ Add initial padding if set ---
            if initial_padding > 0:
                image = pad_image_first(image, padding=initial_padding)

            detect_folder = os.path.join(root, "DETECT")
            os.makedirs(detect_folder, exist_ok=True)

            if crop_class:
                warped_predicted_image, class_name = await crop_class.get_predicted_warped_image(image)

                if warped_predicted_image is not None:
                    crop_folder = os.path.join(root, "CROP")
                    os.makedirs(crop_folder, exist_ok=True)
                    crop_save_path = os.path.join(crop_folder, f"crop_{image_file}")
                    cv2.imwrite(crop_save_path, warped_predicted_image)
                    print(f"Cropped image saved: {crop_save_path}")

                    if detect_class:
                        await run_detection(detect_class, warped_predicted_image,
                                            desired_class_names, detect_folder,
                                            image_file, show_full_annotation)
                else:
                    print(f"Skipping {image_file} - could not warp image.")

            elif detect_class:
                await run_detection(detect_class, image, desired_class_names,
                                    detect_folder, image_file, show_full_annotation)

# -------------------------------
# Main entry
# -------------------------------
async def main():
    # -------------------------------
    # Read CLI args for full configurability
    # -------------------------------
    crop_model_path = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "None" else None
    detect_model_path = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "None" else None
    root_folder_path = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "None" else None
    desired_class_names = ast.literal_eval(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "None" else None
    show_full_annotation = ast.literal_eval(sys.argv[5]) if len(sys.argv) > 5 else False
    initial_padding = int(sys.argv[6]) if len(sys.argv) > 6 else 0

    crop_model = ModelsBAT(crop_model_path) if crop_model_path else None
    detect_model = ModelsBAT(detect_model_path) if detect_model_path else None

    crop_class = CropModel(crop_model.selectMODEL) if crop_model else None
    detect_class = detect_model if detect_model else None

    if not crop_class and not detect_class:
        print("No models provided. Exiting without processing.")
        return

    await process_images_recursive(root_folder_path, crop_class, detect_class,
                                   desired_class_names, show_full_annotation,
                                   initial_padding=initial_padding)


if __name__ == "__main__":
    asyncio.run(main())

"""
conda activate office && python -c "$(curl -s https://raw.githubusercontent.com/ShakhawatShanin/HawkEyes/refs/heads/main/tool.py)" \
None \
"/home/shanin/Desktop/UNILEVER_Master/AI_Models/ublDA_v8.1.pt" \
"/home/shanin/Downloads/November CSD/TRESEMME RRO COND KERATN SMOOTH LC 190ML" \
"['tresemme_ks_white','tresemme_ks_black']" \
False \
100
"""
# crop_model_path → None or [path]
# detect_model_path → None or [path]
# root_folder_path → None or [path]
# desired_class_names → None or ["class1","class2"]
# show_full_annotation → True / False
# initial_padding → numeric value
