import json
import random
import cv2
import numpy as np
import albumentations as A
from pathlib import Path
import uuid
import supervision as sv
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Union


class PolygonAugmentation:
    def __init__(self, tolerance = 0.2, area_threshold = 0.01, debug = False):
        """
        Initialize the PolygonAugmentaion class.
        
        Args:
            tolerance (float): Tolerance for polygon simplification.
            area_threshold (float): Area threshold to filter small polygons.
            debug(bool): If True, enables detailed debug prints.
        """
        self.tolerance = tolerance
        self.area_threshold = area_threshold
        self.debug = debug
        
    def __getattr__(self, name:str) -> Any:
        """Catch missing atrtributes access."""
        raise AttributeError(f"'PolugonAugmentation' object has no atribute '{name}'")    

    def calculate_polygon_area(self, points: List[List[float]]) -> float:
        """Calculate the area of a polygon."""
        poly_np = np.array(points, dtype=np.float32) # Convert to numpy array
        area = cv2.contourArea(poly_np) # Calculate the area using OpenCV
        if self.debug:
            print(f"[DEBUG] Calcualting polygon area: {area:.2f}")
        return cv2.contourArea(poly_np)

    def load_labelme_data(self, json_path: str, image_dir: str) -> Tuple:
        """Load LabelMe JSON and corresponding image."""
        json_path = Path(json_path)
        img_path = Path(image_dir) / f"{json_path.stem}.png"
        # Add debug 
        if self.debug:
            print(f"[DEBUG] Loading image from {img_path}")
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f" Could not load at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Check 'Shapes as key is in JSON
        if 'shapes' not in data or not isinstance(data['shapes'], list):
            raise ValueError(f"Invalid JSON at {json_path}: 'shapes' key missing or not a list")
        # Store polygons, labels and areas
        polygons = []
        labels = []
        original_areas = []
        
        for shape in data['shapes']:
            if shape.get('shape_type') != 'polygon' or not shape.get('points') or len(shape['points']) < 3:
                if self.debug:  
                    print(f"[DEBUG] Skipping invalid shape in {json_path}")
                continue
            try:
                points = [[float(x), float(y)] for x, y in shape['points']]
                polygons.append(points)
                labels.append(shape['label'])
                original_areas.append(self.calculate_polygon_area(points))
            except (ValueError, TypeError):
                if self.debug:
                    print(f"[DEBUG] Error processing points in {json_path}: {shape['points']}")
                continue   
        #W Check if polygons are empty     
        if not polygons and self.debug:
            print(f"[DEBUG] Warning: No valid polygons in {json_path}")
        return image, polygons, labels, original_areas, data, json_path.stem

    def simplify_polygon(self, polygon: List[List[float]], tolerance: float = None) -> List[List[float]]:
        """
        Simplify a polygon using Douglas-Peucker algorithm.

        Args:
            polygon: List of (x, y) points representing the polygon.
            tolerance: Optional override for simplification tolerance.
        
        Returns:
            Simplified polygon as list of points.
        """
        # Use the passed tolerance or the default one from the object
        tol = tolerance if tolerance is not None else self.tolerance
        # If polygon has less than 3 points, it can't be simplified
        if len(polygon) < 3:
            if self.debug:
                print(f"[DEBUG] Polygon has fewer than 3 points, skipping simplification.")
            return polygon
        # Convert polygon to Numpy array for OpenCV
        poly_np = np.array(polygon, dtype=np.float32)
        # Apply Douglas-Peucker algorithm for simplification
        approx = cv2.approxPolyDP(poly_np, tol, closed=True)
        # Reshape back to Python list of points
        simplified = approx.reshape(-1, 2).tolist()
        
        if self.debug:
            print(f"[DEBUG] Simplified polygon from {len(polygon)} to {len(simplified)} points with tolerance {tol}")
        
        return simplified

    def create_donut_polygon(self, external_contour: np.ndarray, internal_contours: List[np.ndarray]) -> List[List[float]]:
        """
        Create a donut polygon by connecting outer contour and inner contours (holes).

        Args:
            external_contour: NumPy array of external (outer) polygon points.
            internal_contours: List of NumPy arrays of internal (hole) polygons.
        
        Returns:
            A single list of points representing the connected donut polygon.
        """
        # Convert external contour to list of points
        external_points = external_contour.reshape(-1, 2).tolist()
         # If there are no internal holes, return external points directly
        if not internal_contours:
            if self.debug:
                print("[DEBUG] No internal contours found, returning external points.")
            return external_points
        
        # Start building the result with the external contour
        result_points = external_points.copy()
        
        for internal_contour in internal_contours:
            internal_points = internal_contour.reshape(-1, 2).tolist()
            
             # Process each internal hole one by one
        for internal_contour in internal_contours:
            # Convert internal contour to list of points
            internal_points = internal_contour.reshape(-1, 2).tolist()

            # Find the closest point pair between external and internal
            min_dist = float('inf')
            ext_idx = 0
            int_idx = 0
            
            # Compare every external point with every internal point
            for i, p1 in enumerate(external_points):
                for j, p2 in enumerate(internal_points):
                    dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        ext_idx = i
                        int_idx = j
            
            # Create bridge, save the points
            bridge_to = external_points[ext_idx]
            bridge_from = internal_points[int_idx]
            
            if self.debug:
                print(f"[DEBUG] Creating bridge between external index {ext_idx} and internal index {int_idx}, distance {min_dist:.2f}")
            # Create new points:
            #  - go from external start up to ext_idx
            #  - cross bridge to internal at int_idx
            #  - follow internal contour around
            #  - cross bridge back to external
            #  - continue external contour after ext_idx
            # Connect contours
            new_points = (
                result_points[:ext_idx+1] +
                internal_points[int_idx:] + internal_points[:int_idx+1] +
                [bridge_to] +
                external_points[ext_idx+1:]
            )
            # Update result_points to include this new bridge and hole
            result_points = new_points
        
        return result_points

    def save_augmented_data(
        self,
        aug_image: np.ndarray,
        aug_polygons: List[List[List[float]]],
        aug_labels: List[str],
        original_data: Dict[str, Any],
        output_img_dir: str,
        output_json_dir: str,
        base_name: str
        ) -> Tuple[Path, Path]:
        """
        Save augmented image and JSON annotation in LabelMe format.

        Args:
            aug_image: Augmented image as numpy array.
            aug_polygons: List of augmented polygons.
            aug_labels: List of labels corresponding to polygons.
            original_data: Original JSON data to preserve some metadata.
            output_img_dir: Directory to save augmented images.
            output_json_dir: Directory to save augmented JSON files.
            base_name: Base name for the augmented files.

        Returns:
            Tuple containing paths to the saved image and JSON file.
        """
        # Create output directories if they don't exists
        output_img_dir = Path(output_img_dir)
        output_json_dir = Path(output_json_dir)
        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_json_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate a short random ID for unique filenames
        aug_id = uuid.uuid4().hex[:4] # Generate a random ID
        aug_img_name = f"{base_name}_{aug_id}_aug.png"
        aug_json_name = f"{base_name}_{aug_id}_aug.json"
        
        # Save augmented image
        if self.debug:
            print(f"[DEBUG] Saving augmented image to {output_img_dir / aug_img_name}")
            
        aug_img_path = output_img_dir / aug_img_name
        cv2.imwrite(str(aug_img_path), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
        
        # Prepare shapes list for labelME JSON
        new_shapes = []
        for poly, label in zip(aug_polygons, aug_labels):
            if not poly or len(poly) < 3:
                continue
            new_shapes.append({
                "label": label,
                "points": poly,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            })
        
        # Assemble the new JSON data
        aug_data = {
            "version": original_data.get("version", "5.0.1"),
            "flags": original_data.get("flags", {}),
            "shapes": new_shapes,
            "imagePath": f"..\\images\\{aug_img_name}", # This path is only degined to open the images and JSON in LabelME
            "imageData": None,
            "imageHeight": aug_image.shape[0],
            "imageWidth": aug_image.shape[1]
        }
        
        # Save the new JSON file
        aug_json_path = output_json_dir / aug_json_name
        with open(aug_json_path, 'w', encoding='utf-8') as f:
            json.dump(aug_data, f, indent=2)
        
        if self.debug:
            print(f"[DEBUG] Saved augmented image to {aug_img_path}")
            print(f"[DEBUG] Saved augmented JSON to {aug_json_path}")
        return aug_img_path, aug_json_path

    def polygons_to_masks(self, image: np.ndarray, polygons: List[List[List[float]]], labels: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Convert polygons into separate masks for each object.

        Args:
            image: Original image array (used for size reference).
            polygons: List of polygons (list of points).
            labels: List of corresponding labels.

        Returns:
            A tuple (masks array, labels list).
        """
        # Get image height and width
        height, width = image.shape[:2]
        all_masks: List[np.ndarray] = []
        all_labels: List[str] = []
        
        # For each polygon, create a mask
        for poly_idx, (poly, label) in enumerate(zip(polygons, labels)):
            try:
                poly_np = np.array(poly, dtype=np.int32)
                if len(poly_np) < 3:
                    # Skip polygons with fewer than 3 points
                    if self.debug:
                        print(f"[DEBUG] Skipping polygon {poly_idx}: fewer than 3 points")
                    continue
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(mask, [poly_np], 1)
                all_masks.append(mask)
                all_labels.append(label)
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] Error processing polygon {poly_idx}: {str(e)}")
        
        if not all_masks:
            # If no masks were created, return an empty array
            return np.zeros((0, height, width), dtype=np.uint8), []
         # Stack all masks into a single numpy array
        masks_array = np.array(all_masks, dtype=np.uint8)

        if self.debug:
            print(f"[DEBUG] Created {len(all_masks)} masks from polygons.")
        
        return np.array(all_masks, dtype=np.uint8), all_labels

    def process_contours(
        self,
        external_contour: np.ndarray,
        internal_contours: List[np.ndarray],
        width: int,
        height: int,
        label: str,
        all_polygons: List[List[List[float]]],
        all_labels: List[str],
        tolerance: float = None
    ) -> None:
        """
        Process external and internal contours, simplify them, 
        and store as LabelMe format polygons.

        Args:
            external_contour: Numpy array of the outer contour.
            internal_contours: List of numpy arrays of inner (hole) contours.
            width: Width of the image (for boundary clipping).
            height: Height of the image (for boundary clipping).
            label: Label name for these polygons.
            all_polygons: List to store resulting polygons.
            all_labels: List to store corresponding labels.
            tolerance: Optional override for simplification tolerance.

        Returns:
            None. Results are appended into all_polygons and all_labels.
        """
        # Use the given tolerance or fall back to the object's default
        tol = tolerance if tolerance is not None else self.tolerance
        
        # Simplify external contour
        external_points = external_contour.reshape(-1, 2).tolist()
        simplified_external = self.simplify_polygon(external_points, tolerance)
        
        if len(simplified_external) >= 3:
            # Clip points inside image boundary and round
            poly_labelme = [[round(max(0, min(float(x), width - 1)), 2),
                            round(max(0, min(float(y), height - 1)), 2)]
                            for x, y in simplified_external]
            all_polygons.append(poly_labelme)
            all_labels.append(label)
            if self.debug:
                print(f"[DEBUG] Added simplified external polygon with {len(poly_labelme)} points.")

        # Simplify internal contours
        for internal_contour in internal_contours:
            internal_points = internal_contour.reshape(-1, 2).tolist()
            simplified_internal = self.simplify_polygon(internal_points, tolerance)
            
            if len(simplified_internal) >= 3:
                poly_labelme = [[round(max(0, min(float(x), width - 1)), 2),
                                round(max(0, min(float(y), height - 1)), 2)]
                                for x, y in simplified_internal]
                all_polygons.append(poly_labelme)
                all_labels.append(label)
                if self.debug:
                    print(f"[DEBUG] Added simplified internal polygon with {len(poly_labelme)} points.")

    def masks_to_labelme_polygons(
        self,
        masks: np.ndarray,
        labels: List[str],
        original_areas: List[float],
        area_threshold: float = None,
        tolerance: float = None
    ) -> Tuple[List[List[List[float]]], List[str]]:
        """
        Convert masks back into LabelMe-style polygons, with optional simplification.

        Args:
            masks: Array of masks (N, H, W), each mask is one object.
            labels: List of labels corresponding to each mask.
            original_areas: List of original polygon areas before augmentation.
            area_threshold: Minimum relative area to keep (optional override).
            tolerance: Tolerance for simplification (optional override).

        Returns:
            Tuple (list of polygons, list of corresponding labels)
        """
        # Set tolerance and area threshold, either passed or default from class
        tol = tolerance if tolerance is not None else self.tolerance
        area_thresh = area_threshold if area_threshold is not None else self.area_threshold
        # Gety image dimensions
        height, width = masks[0].shape if len(masks) > 0 else (0, 0)
        all_polygons = []
        all_labels = []
        
        # Process each mask individually
        for mask_idx, (mask, label) in enumerate(zip(masks, labels)):
            if mask.sum() < 10:
                # Skip empty or very small masks
                if self.debug:
                    print(f"[DEBUG] Skipping mask {mask_idx}: very small or empty.")
                continue
            # Find external and internal contours
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if hierarchy is None or len(contours) == 0:
                if self.debug:
                    print(f"[DEBUG] No contours found in mask {mask_idx}.")
                continue
            
            hierarchy = hierarchy[0] # Needed because OpenCV wraps hierarchy in another list
            external_contours = []
            internal_contours_map = {}
            
            # Seperate external and intrernal contours
            for i, (contour, h) in enumerate(zip(contours, hierarchy)):
                if h[3] == -1:
                    # No parent -> external contour
                    external_contours.append(contour)
                    internal_contours_map[len(external_contours)-1] = []
                else:
                    parent_idx = h[3]
                    for j, _ in enumerate(external_contours):
                        if parent_idx == j:
                            internal_contours_map[j].append(contour)
                            break
            
            if not external_contours:
                # No external contours found
                if self.sebug:
                    print(f"[DEBUG] No external contours found in mask {mask_idx}.")
                continue
            
            # Process each external contour and its internal holes
            for ext_idx, external_contour in enumerate(external_contours):
                internal_contours = internal_contours_map.get(ext_idx, [])
                ext_area = cv2.contourArea(external_contour)
                if ext_area <= 0:
                    # Skip invalid or zero-area contours
                    continue
                # Check if area is too small compared to original area
                if mask_idx < len(original_areas) and original_areas[mask_idx] > 0:
                    relative_area = ext_area / original_areas[mask_idx]
                    if relative_area < area_thresh:
                        if self.debug:
                            print(f"[DEBUG] Skipping contour {ext_idx} (area too small: {relative_area:.4f})")
                        continue
                
                # Check if label is background (for donut creation)
                is_background = label.lower() in ['background', 'bg', 'back']
                if is_background and internal_contours:
                    try:
                        donut_points = self.create_donut_polygon(external_contour, internal_contours)
                        simplified_donut = self.simplify_polygon(donut_points, tol)
                        if len(simplified_donut) >= 3:
                            poly_labelme = [[round(max(0, min(float(x), width - 1)), 2),
                                            round(max(0, min(float(y), height - 1)), 2)]
                                            for x, y in simplified_donut]
                            all_polygons.append(poly_labelme)
                            all_labels.append(label)
                            if self.debug:
                                print(f"[DEBUG] Added donut polygon with {len(poly_labelme)} points.")

                    except Exception as e:
                        # If donut creation fails, fallback to separate processing
                        if self.debug:
                            print(f"[DEBUG] Error creating donut: {str(e)}, fallback to separate polygons.")
                        self.process_contours(
                            external_contour, internal_contours, width, height,
                            label, all_polygons, all_labels, tol
                        )
                else:
                    # Normal (non-background) case: process separately
                    self.process_contours(
                        external_contour, internal_contours, width, height,
                        label, all_polygons, all_labels, tol
                    )
        
        return all_polygons, all_labels

    def augment_dataset(
        self,
        data_dir: str,
        json_dir: str,
        save_img_dir: str,
        save_json_dir: str,
        num_augmentations: Union[int, str],
        augmentation_params: Dict[str, Any]
    ) -> None:
        """
        Perform random augmentations on images and their corresponding LabelMe annotations.

        Args:
            data_dir: Directory containing original images.
            json_dir: Directory containing original LabelMe JSON files.
            save_img_dir: Directory to save augmented images.
            save_json_dir: Directory to save augmented JSON files.
            num_augmentations: Number of augmentations to perform per image ("random" or fixed int).
            augmentation_params: Dictionary of augmentation settings.
        """
        # Convert input directories to Path objects
        data_dir_path = Path(data_dir)
        json_dir_path = Path(json_dir)

        # Define the base augmentation pipeline
        augmentation_transform = A.Compose([
            A.Rotate(limit=augmentation_params.get("angle_limit", 10), p=augmentation_params.get("p_rotate", 1.0)),
            A.HorizontalFlip(p=augmentation_params.get("p_flip_h", 0.5)),
            A.VerticalFlip(p=augmentation_params.get("p_flip_v", 0.5)),
            A.Affine(scale=(0.95, 1.05), translate_percent=(-0.05, 0.05), p=augmentation_params.get("p_scale", 0.5)),
        ])

        processed_count = 0  # Counter for successful augmentations

        # Get all JSON files
        json_list = list(json_dir_path.glob("*.json"))
        total_files = len(json_list)

        # Use tqdm for progress bar
        with tqdm(total=total_files, desc="Augmenting Images", ncols=80, unit="file") as pbar:
            for json_path in json_list:
                try:
                    # Corresponding image path
                    image_path = data_dir_path / f"{json_path.stem}.png"
                    if not image_path.exists():
                        tqdm.write(f"[WARN] Image {image_path} not found, skipping...")
                        pbar.update(1)
                        continue

                    # Load image, polygons, labels
                    image, polygons, labels, original_areas, original_data, base_name = self.load_labelme_data(
                        str(json_path), str(data_dir_path)
                    )

                    height, width = image.shape[:2]

                    # Create a dynamic crop transform
                    crop_height = int(height * augmentation_params.get("crop_scale", 0.9))
                    crop_width = int(width * augmentation_params.get("crop_scale", 0.9))
                    dynamic_transform = A.Compose([
                        augmentation_transform,
                        A.RandomCrop(width=crop_width, height=crop_height, p=augmentation_params.get("p_crop", 0.6)),
                    ])

                    # Validate polygons
                    if not polygons:
                        tqdm.write(f"[WARN] No valid polygons in {json_path}, skipping...")
                        pbar.update(1)
                        continue

                    masks, mask_labels = self.polygons_to_masks(image, polygons, labels)
                    if masks.shape[0] == 0:
                        tqdm.write(f"[WARN] No valid masks created for {json_path}, skipping...")
                        pbar.update(1)
                        continue

                    # How many augmentations per image
                    if num_augmentations == "random":
                        num_augs = random.randint(1, augmentation_params.get("random_aug_per_type", 5))
                    else:
                        num_augs = int(num_augmentations)

                    # Perform augmentations
                    for i in range(num_augs):
                        try:
                            aug_result = dynamic_transform(image=image, masks=masks)
                            aug_image = aug_result['image']
                            aug_masks = aug_result['masks']

                            aug_polygons, aug_labels = self.masks_to_labelme_polygons(
                                aug_masks,
                                mask_labels,
                                original_areas,
                                area_threshold=self.area_threshold,
                                tolerance=self.tolerance
                            )

                            if not aug_polygons:
                                tqdm.write(f"[WARN] No valid polygons after augmentation {i+1} for {json_path.stem}, skipping...")
                                continue

                            self.save_augmented_data(
                                aug_image, aug_polygons, aug_labels,
                                original_data, save_img_dir, save_json_dir, base_name
                            )

                            processed_count += 1
                            tqdm.write(f"[INFO] Saved augmentation {i+1}/{num_augs} for {json_path.stem}, {len(aug_polygons)} polygons.")

                        except Exception as e:
                            tqdm.write(f"[ERROR] Error in augmentation {i+1} for {json_path.stem}: {str(e)}")
                            continue

                    pbar.update(1)  # After processing one image

                except Exception as e:
                    tqdm.write(f"[ERROR] Error processing {json_path}: {str(e)}")
                    pbar.update(1)
                    continue

        print(f"[DONE] Augmentation complete. Processed {processed_count} augmentations.")




if __name__ == "__main__":
    augmentation_params = {
        "crop_scale": 0.8,
        "angle_limit": 20,
        "p_crop": 0.7,
        "p_rotate": 1.0,
        "p_flip_h": 0.5,
        "p_flip_v": 0.5,
        "p_scale": 0.6,
        "random_aug_per_type": 5
    }
    
    # Step 2: create an instance of your augmentor class
    augmentor = PolygonAugmentation(
        tolerance=2.0,        # you can change if needed
        area_threshold=0.01,  # you can change if needed
        debug=False            # set True to see debug prints
    )
    
    # Step 3: call the method
    augmentor.augment_dataset(
        data_dir="for_augmentation/dwg_mode/images",
        json_dir="for_augmentation/dwg_mode/labels",
        save_img_dir="for_augmentation/images",
        save_json_dir="for_augmentation/labels",
        num_augmentations="random",   # or you can say 3, 5, etc.
        augmentation_params=augmentation_params
    )
