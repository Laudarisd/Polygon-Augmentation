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
    def __init__(self, tolerance=0.2, area_threshold=0.01, debug=False):
        """
        Initialize the PolygonAugmentation class.
        
        Args:
            tolerance (float): Tolerance for polygon simplification.
            area_threshold (float): Area threshold to filter small polygons.
            debug (bool): If True, enables detailed debug prints.
        """
        self.tolerance = tolerance
        self.area_threshold = area_threshold
        self.debug = debug
        self.supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.PNG', 'JPEG']  # Supported image extensions
        
    def __getattr__(self, name: str) -> Any:
        """Catch missing attribute access."""
        raise AttributeError(f"'PolygonAugmentation' object has no attribute '{name}'")    

    def calculate_polygon_area(self, points: List[List[float]]) -> float:
        """Calculate the area of a polygon."""
        poly_np = np.array(points, dtype=np.float32)
        area = cv2.contourArea(poly_np)
        if self.debug:
            print(f"[DEBUG] Calculating polygon area: {area:.2f}")
        return area

    def load_labelme_data(self, json_path: str, image_dir: str) -> Tuple:
        """Load LabelMe JSON and corresponding image with multiple extension support."""
        json_path = Path(json_path)
        image_dir_path = Path(image_dir)
        
        # Try to find an image with supported extensions
        img_path = None
        for ext in self.supported_extensions:
            potential_img_path = image_dir_path / f"{json_path.stem}{ext}"
            if potential_img_path.exists():
                img_path = potential_img_path
                break
        
        if img_path is None:
            raise ValueError(f"No image found for {json_path.stem} with supported extensions {self.supported_extensions} in {image_dir}")
        
        if self.debug:
            print(f"[DEBUG] Loading image from {img_path}")
        
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Could not load image at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'shapes' not in data or not isinstance(data['shapes'], list):
            raise ValueError(f"Invalid JSON at {json_path}: 'shapes' key missing or not a list")
        
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
        
        if not polygons and self.debug:
            print(f"[DEBUG] Warning: No valid polygons in {json_path}")
        return image, polygons, labels, original_areas, data, json_path.stem

    def simplify_polygon(self, polygon: List[List[float]], tolerance: float = None, label: str = None) -> List[List[float]]:
        """
        Simplify a polygon using Douglas-Peucker algorithm, with class-specific tolerance for background.
        
        Args:
            polygon: List of [x, y] points defining the polygon.
            tolerance: Custom tolerance for simplification (overrides self.tolerance if provided).
            label: Label of the polygon to determine if it's background (for higher tolerance).
        
        Returns:
            Simplified polygon as a list of [x, y] points.
        """
        tol = tolerance if tolerance is not None else self.tolerance
        # Increase tolerance for background class to reduce keypoints
        if label and label.lower() in ['background', 'bg', 'back']:
            tol = tol * 3  # Double the tolerance for background (e.g., 1.0 -> 2.0)
            if self.debug:
                print(f"[DEBUG] Using increased tolerance {tol} for background label '{label}'")
        
        if len(polygon) < 3:
            if self.debug:
                print(f"[DEBUG] Polygon has fewer than 3 points, skipping simplification.")
            return polygon
        poly_np = np.array(polygon, dtype=np.float32)
        approx = cv2.approxPolyDP(poly_np, tol, closed=True)
        simplified = approx.reshape(-1, 2).tolist()
        
        if self.debug:
            print(f"[DEBUG] Simplified polygon from {len(polygon)} to {len(simplified)} points with tolerance {tol}")
        return simplified

    def create_donut_polygon(self, external_contour: np.ndarray, internal_contours: List[np.ndarray]) -> List[List[float]]:
        """
        Create a donut polygon by connecting outer contour and inner contours (holes).
        """
        external_points = external_contour.reshape(-1, 2).tolist()
        if not internal_contours:
            if self.debug:
                print("[DEBUG] No internal contours found, returning external points.")
            return external_points
        
        result_points = external_points.copy()
        
        for internal_contour in internal_contours:
            internal_points = internal_contour.reshape(-1, 2).tolist()
            min_dist = float('inf')
            ext_idx = 0
            int_idx = 0
            
            for i, p1 in enumerate(external_points):
                for j, p2 in enumerate(internal_points):
                    dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        ext_idx = i
                        int_idx = j
            
            bridge_to = external_points[ext_idx]
            bridge_from = internal_points[int_idx]
            
            if self.debug:
                print(f"[DEBUG] Creating bridge between external index {ext_idx} and internal index {int_idx}, distance {min_dist:.2f}")
            
            new_points = (
                result_points[:ext_idx+1] +
                internal_points[int_idx:] + internal_points[:int_idx+1] +
                [bridge_to] +
                external_points[ext_idx+1:]
            )
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
        """
        output_img_dir = Path(output_img_dir)
        output_json_dir = Path(output_json_dir)
        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_json_dir.mkdir(parents=True, exist_ok=True)
        
        aug_id = uuid.uuid4().hex[:4]
        aug_img_name = f"{base_name}_{aug_id}_aug.png"  # Output always saved as PNG
        aug_json_name = f"{base_name}_{aug_id}_aug.json"
        
        if self.debug:
            print(f"[DEBUG] Saving augmented image to {output_img_dir / aug_img_name}")
            
        aug_img_path = output_img_dir / aug_img_name
        cv2.imwrite(str(aug_img_path), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
        
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
        
        aug_data = {
            "version": original_data.get("version", "5.0.1"),
            "flags": original_data.get("flags", {}),
            "shapes": new_shapes,
            "imagePath": f"..\\images\\{aug_img_name}",
            "imageData": None,
            "imageHeight": aug_image.shape[0],
            "imageWidth": aug_image.shape[1]
        }
        
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
        """
        height, width = image.shape[:2]
        all_masks: List[np.ndarray] = []
        all_labels: List[str] = []
        
        for poly_idx, (poly, label) in enumerate(zip(polygons, labels)):
            try:
                poly_np = np.array(poly, dtype=np.int32)
                if len(poly_np) < 3:
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
            return np.zeros((0, height, width), dtype=np.uint8), []
        
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
        Process external and internal contours, simplify them, and store as LabelMe format polygons.
        """
        tol = tolerance if tolerance is not None else self.tolerance
        external_points = external_contour.reshape(-1, 2).tolist()
        simplified_external = self.simplify_polygon(external_points, tolerance = tol, label=label)
        
        if len(simplified_external) >= 3:
            poly_labelme = [[round(max(0, min(float(x), width - 1)), 2),
                            round(max(0, min(float(y), height - 1)), 2)]
                            for x, y in simplified_external]
            all_polygons.append(poly_labelme)
            all_labels.append(label)
            if self.debug:
                print(f"[DEBUG] Added simplified external polygon with {len(poly_labelme)} points.")

        for internal_contour in internal_contours:
            internal_points = internal_contour.reshape(-1, 2).tolist()
            simplified_internal = self.simplify_polygon(internal_points, tolerance=tol, label=label)
            
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
        """
        tol = tolerance if tolerance is not None else self.tolerance
        area_thresh = area_threshold if area_threshold is not None else self.area_threshold
        height, width = masks[0].shape if len(masks) > 0 else (0, 0)
        all_polygons = []
        all_labels = []
        
        for mask_idx, (mask, label) in enumerate(zip(masks, labels)):
            if mask.sum() < 10:
                if self.debug:
                    print(f"[DEBUG] Skipping mask {mask_idx}: very small or empty.")
                continue
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if hierarchy is None or len(contours) == 0:
                if self.debug:
                    print(f"[DEBUG] No contours found in mask {mask_idx}.")
                continue
            
            hierarchy = hierarchy[0]
            external_contours = []
            internal_contours_map = {}
            
            for i, (contour, h) in enumerate(zip(contours, hierarchy)):
                if h[3] == -1:
                    external_contours.append(contour)
                    internal_contours_map[len(external_contours)-1] = []
                else:
                    parent_idx = h[3]
                    for j, _ in enumerate(external_contours):
                        if parent_idx == j:
                            internal_contours_map[j].append(contour)
                            break
            
            if not external_contours:
                if self.debug:
                    print(f"[DEBUG] No external contours found in mask {mask_idx}.")
                continue
            
            for ext_idx, external_contour in enumerate(external_contours):
                internal_contours = internal_contours_map.get(ext_idx, [])
                ext_area = cv2.contourArea(external_contour)
                if ext_area <= 0:
                    continue
                if mask_idx < len(original_areas) and original_areas[mask_idx] > 0:
                    relative_area = ext_area / original_areas[mask_idx]
                    if relative_area < area_thresh:
                        if self.debug:
                            print(f"[DEBUG] Skipping contour {ext_idx} (area too small: {relative_area:.4f})")
                        continue
                
                is_background = label.lower() in ['background', 'bg', 'back']
                if is_background and internal_contours:
                    try:
                        donut_points = self.create_donut_polygon(external_contour, internal_contours)
                        simplified_donut = self.simplify_polygon(donut_points, tolerance=tol, label=label)
                        if len(simplified_donut) >= 3:
                            poly_labelme = [[round(max(0, min(float(x), width - 1)), 2),
                                            round(max(0, min(float(y), height - 1)), 2)]
                                            for x, y in simplified_donut]
                            all_polygons.append(poly_labelme)
                            all_labels.append(label)
                            if self.debug:
                                print(f"[DEBUG] Added donut polygon with {len(poly_labelme)} points.")
                    except Exception as e:
                        if self.debug:
                            print(f"[DEBUG] Error creating donut: {str(e)}, fallback to separate polygons.")
                        self.process_contours(
                            external_contour, internal_contours, width, height,
                            label, all_polygons, all_labels, tol
                        )
                else:
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
        Perform random augmentations on images and their corresponding LabelMe annotations with balanced augmentation types.

        Args:
            data_dir: Directory containing original images.
            json_dir: Directory containing original LabelMe JSON files.
            save_img_dir: Directory to save augmented images.
            save_json_dir: Directory to save augmented JSON files.
            num_augmentations: Number of augmentations to perform per image ("random" or fixed int).
            augmentation_params: Dictionary of augmentation settings with ranges.
        """
        data_dir_path = Path(data_dir)
        json_dir_path = Path(json_dir)
        json_file_paths = list(json_dir_path.glob("*.json"))
        total_files = len(json_file_paths)

        # Define available augmentation types with dynamic ranges
        def get_augmentation_types():
            mosaic_dropout_range = augmentation_params.get("mosaic_dropout_prob", (0.05, 0.15))
            return [
                A.Rotate(limit=augmentation_params.get("angle_limit", (-10, 10)), p=augmentation_params.get("p_rotate", 0.7)),
                A.HorizontalFlip(p=augmentation_params.get("p_flip_h", 0.5)),
                A.VerticalFlip(p=augmentation_params.get("p_flip_v", 0.5)),
                A.Affine(
                    scale=augmentation_params.get("scale_limit", (0.5, 1.5)),
                    translate_percent=augmentation_params.get("translate_limit", (-0.1, 0.1)),
                    p=augmentation_params.get("p_scale", 0.7)
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=augmentation_params.get("brightness_limit", (-0.3, 0.3)),
                    contrast_limit=augmentation_params.get("contrast_limit", (-0.3, 0.3)),
                    p=augmentation_params.get("p_brightness", 0.7)
                ),
                A.PixelDropout(
                    dropout_prob=random.uniform(*mosaic_dropout_range),
                    per_channel=False,
                    p=augmentation_params.get("p_mosaic", 0.5)
                ),
            ]

        processed_count = 0
        with tqdm(total=total_files, desc="Augmenting Images", ncols=80, unit="file") as pbar:
            for json_path in json_file_paths:
                try:
                    image, polygons, labels, original_areas, original_data, base_name = self.load_labelme_data(
                        str(json_path), str(data_dir_path)
                    )

                    height, width = image.shape[:2]
                    crop_scale_range = augmentation_params.get("crop_scale_range", (0.7, 0.9))
                    crop_height = int(height * random.uniform(*crop_scale_range))
                    crop_width = int(width * random.uniform(*crop_scale_range))

                    if not polygons:
                        tqdm.write(f"[WARN] No valid polygons in {json_path}, skipping...")
                        pbar.update(1)
                        continue

                    masks, mask_labels = self.polygons_to_masks(image, polygons, labels)
                    if masks.shape[0] == 0:
                        tqdm.write(f"[WARN] No valid masks created for {json_path}, skipping...")
                        pbar.update(1)
                        continue

                    if num_augmentations == "random":
                        num_augs = random.randint(1, augmentation_params.get("random_aug_per_type", 5))
                    else:
                        num_augs = int(num_augmentations)

                    for i in range(num_augs):
                        try:
                            augmentation_types = get_augmentation_types()
                            num_types = random.randint(2, min(4, len(augmentation_types)))
                            selected_augs = random.sample(augmentation_types, num_types)
                            dynamic_transform = A.Compose([
                                *selected_augs,
                                A.RandomCrop(width=crop_width, height=crop_height, p=augmentation_params.get("p_crop", 0.7)),
                            ])

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

                    pbar.update(1)

                except Exception as e:
                    tqdm.write(f"[ERROR] Error processing {json_path}: {str(e)}")
                    pbar.update(1)
                    continue

        print(f"[DONE] Augmentation complete. Processed {processed_count} augmentations.")


if __name__ == "__main__":
    augmentation_params = {
        "crop_scale_range": (0.8, 0.9),
        "angle_limit": (-30, 30),
        "p_crop": 0.8,
        "p_rotate": 0.9,
        "p_flip_h": 0,
        "p_flip_v": 0,
        "p_scale": 0.8,
        "scale_limit": (0.5, 1.5),
        "translate_limit": (-0.1, 0.1),
        "brightness_limit": (-0.1, 0.1),
        "contrast_limit": (-0.1, 0.1),
        "p_brightness": 0.5,
        "mosaic_dropout_prob": (0.01, 0.1),
        "p_mosaic": 0.5,
    }
    
    augmentor = PolygonAugmentation(
        tolerance=2.0,       
        area_threshold=0.01,  
        debug=False            
    )
    
    augmentor.augment_dataset(
        data_dir="for_augmentation/img_mode/images",
        json_dir="for_augmentation/img_mode/labels",
        save_img_dir="for_augmentation/images",
        save_json_dir="for_augmentation/labels",
        num_augmentations=10,
        augmentation_params=augmentation_params
    )
