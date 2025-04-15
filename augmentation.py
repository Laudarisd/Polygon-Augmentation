# --------------------------------------------
# Imports
# --------------------------------------------
import json
import cv2
import os
import albumentations as A
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from shapely import Polygon
import supervision as sv
import uuid


# --------------------------------------------
# Geometry Functions
# --------------------------------------------
def calculate_polygon_area(points):
    poly_np = np.array(points, dtype=np.float32)
    return cv2.contourArea(poly_np)


def simplify_polygon(polygon, epsilon=1.0):
    poly_np = np.array(polygon, dtype=np.float32)
    approx = cv2.approxPolyDP(poly_np, epsilon, closed=True)
    simplified = approx.reshape(-1, 2).tolist()
    return simplified


# --------------------------------------------
# Data Loading and Saving Functions
# --------------------------------------------
def load_labelme_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    img_path = Path(json_path).parent / data['imagePath']
    image = cv2.imread(str(img_path))
    if image is None:
        raise ValueError(f"Could not load image at {img_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    polygons = []
    labels = []
    original_areas = []
    for shape in data.get('shapes', []):
        if shape['shape_type'] == 'polygon':
            points = shape['points']
            area = calculate_polygon_area(points)
            polygons.append(points)
            labels.append(shape['label'])
            original_areas.append(area)
    
    return image, polygons, labels, original_areas, data
#----------------------------------------
# Save information to labelme JSON format
#----------------------------------------

def save_augmented_data(aug_image, aug_polygons, aug_labels, original_data, output_img_dir, output_json_dir, base_name):
    output_img_dir = Path(output_img_dir)
    output_json_dir = Path(output_json_dir)
    img_dir = output_img_dir
    json_dir = output_json_dir
    img_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    
    aug_id = uuid.uuid4().hex[:4]
    aug_img_name = f"{base_name}_{aug_id}_aug.png"
    aug_json_name = f"{base_name}_{aug_id}_aug.json"
    
    aug_img_path = img_dir / aug_img_name
    cv2.imwrite(str(aug_img_path), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
    
    aug_data = {
        "version": original_data.get("version", "5.0.1"),
        "flags": original_data.get("flags", {}),
        "shapes": [
            {
                "label": label,
                "points": poly,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            }
            for poly, label in zip(aug_polygons, aug_labels)
        ],
        "imagePath": f"..\\images\\{aug_img_name}",
        "imageData": None,
        "imageHeight": aug_image.shape[0],
        "imageWidth": aug_image.shape[1]
    }
    
    aug_json_path = json_dir / aug_json_name
    with open(aug_json_path, 'w', encoding='utf-8') as f:
        json.dump(aug_data, f, indent=2)
    
    return aug_img_path, aug_json_path
# --------------------------------------------
# Mask and Polygon Conversion Functions
# --------------------------------------------
def polygons_to_masks(image, polygons, labels):
    height, width = image.shape[:2]
    unique_labels = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    masks = np.zeros((len(unique_labels), height, width), dtype=np.uint8)
    
    for poly, label in zip(polygons, labels):
        poly_np = np.array(poly, dtype=np.int32)
        mask = sv.polygon_to_mask(poly_np, (width, height))
        masks[label_to_idx[label]] = np.maximum(masks[label_to_idx[label]], mask)
    
    return masks, unique_labels


def masks_to_labelme_polygons(masks, labels, original_areas, filename=None, epsilon=1.0, area_threshold=0.05):
    polygons = []
    aug_labels = []
    aug_areas = []
    shape_indices = []
    
    for mask_idx, (mask, label) in enumerate(zip(masks, sorted(set(labels)))):
        mask = (mask > 0.5).astype(np.uint8)
        orig_indices = [i for i, lbl in enumerate(labels) if lbl == label]
        
        # Convert mask to polygons (handles splitting naturally)
        poly_list = sv.mask_to_polygons(mask)
        if not poly_list:
            #print(f"Warning: No polygons found for label {label}")
            continue
        #print(f"Label {label}: {len(poly_list)} polygons from mask")
        # Adjust area threshold if many splits occur (optional)
        adjusted_threshold = area_threshold / max(1, len(poly_list) ** 0.5)
        for poly_idx, poly in enumerate(poly_list):
            if len(poly) >= 3:
                simplified_poly = simplify_polygon(poly, epsilon=epsilon)
                if len(simplified_poly) >= 3:
                    area = calculate_polygon_area(simplified_poly)
                    orig_idx = orig_indices[poly_idx % len(orig_indices)] if orig_indices else 0
                    orig_area = original_areas[orig_idx] if orig_indices else 0
                    
                    bounds = (np.min(simplified_poly, axis=0), np.max(simplified_poly, axis=0))
                    #print(f"Polygon {poly_idx} for {label}: area={area:.2f}, bounds={bounds}")
                    if orig_area > 0 and (area / orig_area) >= adjusted_threshold:
                        poly_labelme = [[round(max(0, min(float(x), masks[0].shape[1] - 1)), 2),
                                        round(max(0, min(float(y), masks[0].shape[0] - 1)), 2)]
                                       for x, y in simplified_poly]
                        polygons.append(poly_labelme)
                        aug_labels.append(label)
                        aug_areas.append(area)
                        shape_indices.append(orig_idx)
    

    # Handle overlapping walls
    to_remove = set()
    for i, (poly_i, label_i, area_i) in enumerate(zip(polygons, aug_labels, aug_areas)):
        if label_i != "wall":  # Skip non-wall polygons
            continue
        poly_i_np = np.array(poly_i, dtype=np.float32)
        for j, (poly_j, label_j, area_j) in enumerate(zip(polygons, aug_labels, aug_areas)):
            if i == j or label_j != "wall":  # Skip same polygon or non-wall polygons
                continue
            poly_j_np = np.array(poly_j, dtype=np.float32)
            
            # Check if poly_i is completely inside poly_j
            all_inside_i_in_j = True
            for point in poly_i_np:
                dist = cv2.pointPolygonTest(poly_j_np, tuple(point), False)
                if dist < 0:  # Outside
                    all_inside_i_in_j = False
                    break
            # Check if poly_j is completly inside poly_i
            all_inside_j_in_i = True
            for point in poly_j_np:
                dist = cv2.pointPolygonTest(poly_i_np, tuple(point), False)
                if dist < 0:
                    all_inside_j_in_i = False
                    break # Outside
                
            # If one is compleltely inside the other, mark the smaller one for removal
            if all_inside_i_in_j and area_i < area_j:
                print(f"file name:{filename} Smaller wall (area={area_i:.2f}) inside larger wall (area={area_j:.2f}), marking for removal")
                to_remove.add(i)
            elif all_inside_j_in_i and area_j < area_i:
                print(f"file name:{filename} Smaller wall (area={area_j:.2f}) inside larger wall (area={area_i:.2f}), marking for removal")
                to_remove.add(j)
                
    # Remove marked polygons
    for idx in sorted(to_remove, reverse=True):
        print(f"file name:{filename} Removing smaller wall at index {idx}")
        polygons.pop(idx)
        aug_labels.pop(idx)
        aug_areas.pop(idx)
        shape_indices.pop(idx)  
                
    # Handle background with exactly two polygons
    background_count = aug_labels.count('background')
    if background_count == 2:
        background_indices = [i for i, lbl in enumerate(aug_labels) if lbl == 'background']
        poly1, poly2 = polygons[background_indices[0]], polygons[background_indices[1]]
        area1, area2 = aug_areas[background_indices[0]], aug_areas[background_indices[1]]

        # Determine larger and smaller polygons
        larger_area = max(area1, area2)
        smaller_area = min(area1, area2)
        if larger_area <= smaller_area or smaller_area <= 0:
            print(f"{filename}...invalid areas: larger {larger_area:.2f}, smaller {smaller_area:.2f}")
            return polygons, aug_labels, aug_areas, shape_indices

        larger_poly = poly1 if area1 == larger_area else poly2
        smaller_poly = poly2 if area1 == larger_area else poly1

        # Validate polygon sizes
        if len(larger_poly) < 3 or len(smaller_poly) < 3:
            print(f"{filename}...invalid polygons: larger {len(larger_poly)}, smaller {len(smaller_poly)}")
            return polygons, aug_labels, aug_areas, shape_indices

        # Convert to numpy for OpenCV
        larger_poly_np = np.array(larger_poly, dtype=np.float32)
        smaller_poly_np = np.array(smaller_poly, dtype=np.float32)

        # Check if smaller polygon is inside larger
        all_inside = True
        for point in smaller_poly_np:
            dist = cv2.pointPolygonTest(larger_poly_np, tuple(point), False)
            if dist < 0:  # Outside
                all_inside = False
                break

        if all_inside:
            print(f"{filename}...contains smaller one completely inside")
            uncovered_area = larger_area - smaller_area
            #print(f"Uncovered area (larger - smaller): {uncovered_area:.2f}")

            try:
                # Find outer start (u1) and closest inner point (I_closest)
                min_dist = float('inf')
                u1_idx = 0
                I_closest_idx = 0
                for i, p1 in enumerate(larger_poly):
                    for j, p2 in enumerate(smaller_poly):
                        dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                        if dist < min_dist:
                            min_dist = dist
                            u1_idx = i
                            I_closest_idx = j
                #print(f"DEBUG: file name:{filename} Closet inner point: index={I_closest_idx}, point={smaller_poly[I_closest_idx]}")

                # Store outer and inner points
                n_large = len(larger_poly)
                n_small = len(smaller_poly)
                outer_points = []
                inner_points = []

                # Outer loop: u1, u2, ..., un, u1
                for i in range(n_large):
                    idx = (u1_idx + i) % n_large # Start at u1, go around
                    outer_points.append(larger_poly[idx])
                outer_points.append(larger_poly[u1_idx])  # End at u1
                
                #print(f"file name:{filename} Starting outer point: {outer_points[0]}, Ending outer point: {outer_points[-1]}")
                # Inner loop: I_closest, In, In-1, ..., I2
                for i in range(n_small + 1):
                    idx = (I_closest_idx - i ) % n_small  # Reverse, stop before I_closest
                    inner_points.append(smaller_poly[idx])
                #print(f"file name:{filename} Starting inner point: {inner_points[0]}, Ending outer point: {inner_points[-1]}")
                
                #print(f"file name:{filename} Starting inner point: index={I_closest_idx}, point={inner_points[0]}, "
                #        f"Ending inner point: index={I_closest_idx}, point={inner_points[-1]}")
                
                
                
                # Visualize the outer, inner key points with start and end points
                # vis_dir = './test_visualization_inner_outter'
                # os.makedirs(vis_dir, exist_ok=True)

                # img = np.zeros((masks[0].shape[0], masks[0].shape[1], 3), dtype=np.uint8)
                # # Outer: white line, red points
                # cv2.polylines(img, [np.array(outer_points, np.int32)], True, (255, 255, 255), 3)  # White, thickness 2
                # for point in outer_points[1:-1]:  # Skip start/end for normal points
                #     cv2.circle(img, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)  # Red points
                # # Annotate outer_points[0]
                # u1 = outer_points[0]
                # cv2.putText(img, "u1_s/e", (int(u1[0]), int(u1[1])), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 2)  # Font size 1.0, thickness 2
                # # Inner: green line, white points
                # cv2.polylines(img, [np.array(inner_points, np.int32)], True, (0, 255, 0), 3)  # Green, thickness 2
                # for point in inner_points[1:-1]:  # Skip start/end for normal points
                #     cv2.circle(img, (int(point[0]), int(point[1])), 5, (255, 255, 255), -1)  # White points
                # # Annotate inner start/end (I_closest)
                # I_closest = inner_points[0]
                # cv2.putText(img, "I_s/e", (int(I_closest[0]), int(I_closest[1])), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 2)  # Font size 1.0, white, thickness 2
                # cv2.imwrite(os.path.join(vis_dir, f"{filename}_poly.png"), img)          
                
                # Combine: u1, ..., u1, I_closest, In, ..., I2, I_closest
                I_closest = smaller_poly[I_closest_idx]
                uncovered_poly = outer_points + inner_points[:-1] + [I_closest] + [outer_points[0]]  # Close at u1
                
                print(f"file name:{filename} Starting point: {uncovered_poly[0]}, Ending point: {uncovered_poly[-1]}")
          
                #print(f"DEBUG: file name:{filename} Check uncovered polygon: {len(uncovered_poly)} points")
                # Visualize combined polygon
                # Visualize the combined polygon (not closed)
                # vis_dir = './test_visualization'
                # os.makedirs(vis_dir, exist_ok=True)
                # img = np.zeros((masks[0].shape[0], masks[0].shape[1], 3), dtype=np.uint8)
                # # Draw combined polygon: white line, red points
                # cv2.polylines(img, [np.array(uncovered_poly, np.int32)], False, (255, 255, 255), 2)  # White, not closed
                # for point in uncovered_poly[1:-1]:  # Skip start/end for normal points
                #     cv2.circle(img, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)  # Red points
                # # Annotate start (u1) and end (u1)
                # u1 = uncovered_poly[0]
                # cv2.putText(img, "u1_s", (int(u1[0]), int(u1[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                # u1_end = uncovered_poly[-1]
                # cv2.putText(img, "u1_e", (int(u1_end[0]), int(u1_end[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                #scv2.imwrite(os.path.join(vis_dir, f"{filename}_combined.png"), img)

                # Form the polygon from combined keypoints
                #print(f"file name:{filename} Polygon formed with {len(uncovered_poly)} keypoints")
                # Convert to NumPy array for OpenCV
                uncovered_poly_np = np.array(uncovered_poly, dtype=np.int32)
                # let's visualize the uncovered polygon
                # Visualize the uncovered polygon with a transparent white mask
                # vis_dir = './test_visualization'
                # os.makedirs(vis_dir, exist_ok=True)
                # img = np.zeros((masks[0].shape[0], masks[0].shape[1], 3), dtype=np.uint8)
                # mask = np.zeros((masks[0].shape[0], masks[0].shape[1]), dtype=np.uint8)\
                # # Draw the filled polygon on the mask
                # cv2.fillPoly(mask, [uncovered_poly_np], 255)
                # # Create a transparent white overlay (alpha = 0.5)
                # white_mask = np.zeros_like(img)
                # white_mask[mask == 255] = [255, 255, 255]
                # alpha = 0.5
                # img = cv2.addWeighted(img, 1.0, white_mask, alpha, 0.0)
                # # Draw the polygon outline
                # cv2.polylines(img, [uncovered_poly_np], False, (255, 255, 255), 2)
                # # Save the image
                # cv2.imwrite(os.path.join(vis_dir, f"{filename}_combined.png"), img)
                # Remove original background polygons
                for idx in sorted(background_indices, reverse=True):
                    polygons.pop(idx)
                    aug_labels.pop(idx)
                    aug_areas.pop(idx)
                    shape_indices.pop(idx)

                # Add merged polygon
                polygons.append(uncovered_poly)  # Use uncovered_poly directly
                aug_labels.append("background")
                aug_areas.append(calculate_polygon_area(uncovered_poly))  # Compute area of uncovered_poly
                shape_indices.append(-1)
                
                # Visualize all classes with unique colors and transparent masks
                vis_dir = './test_visualization'
                os.makedirs(vis_dir, exist_ok=True)
                img = np.zeros((masks[0].shape[0], masks[0].shape[1], 3), dtype=np.uint8)

                # Define unique colors for each class
                unique_labels = sorted(set(aug_labels))
                color_map = {}
                for idx, label in enumerate(unique_labels):
                    # Generate a color using HSV for better distinction, convert to BGR
                    h = (idx * 137 % 360)  # Spread hues
                    s = 255
                    v = 255
                    hsv_color = np.array([[[h, s, v]]], dtype=np.uint8)
                    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0].tolist()
                    color_map[label] = bgr_color

                # Draw each polygon with a transparent mask
                for poly, label in zip(polygons, aug_labels):
                    poly_np = np.array(poly, dtype=np.int32)
                    mask = np.zeros((masks[0].shape[0], masks[0].shape[1]), dtype=np.uint8)
                    
                    # Draw filled polygon on mask
                    cv2.fillPoly(mask, [poly_np], 255)
                    
                    # Create transparent colored overlay
                    color = color_map[label]
                    colored_mask = np.zeros_like(img)
                    colored_mask[mask == 255] = color
                    alpha = 0.5
                    img = cv2.addWeighted(img, 1.0, colored_mask, alpha, 0.0)
                    
                    # Draw polygon outline
                    cv2.polylines(img, [poly_np], False, color, 2)

                # Save the visualization
                cv2.imwrite(os.path.join(vis_dir, f"{filename}_all_classes.png"), img)

                print(f"{filename}...replaced background polygons with ring")

            except Exception as e:
                print(f"{filename}...error merging polygons: {str(e)}")
            
    return polygons, aug_labels, aug_areas, shape_indices

# --------------------------------------------
# Augmentation Functions
# --------------------------------------------
def augment_image_and_masks(image, masks, transformation):
    transformed = transformation(image=image, masks=masks)
    aug_image = transformed['image']
    aug_masks = transformed['masks']
    return aug_image, aug_masks


def augment_dataset(data_dir, json_dir, save_img_dir, save_json_dir, num_augmentations, augmentation_params):
    data_dir = Path(data_dir)
    json_dir = Path(json_dir)
    
    for json_path in json_dir.glob("*.json"):
        #print(f"Processing {json_path}")
        image_path = data_dir / f"{json_path.stem}.png"
        if not image_path.exists():
            #print(f"Image {image_path} not found, skipping...")
            continue
        # Load data
        image, polygons, labels, original_areas, original_data = load_labelme_data(str(json_path))
        # Dynamically calculate crop size
        crop_factor = augmentation_params.get("crop_scale", 0.8)
        crop_size = int(min(image.shape[0], image.shape[1]) * crop_factor)
        #print(f"Original image shape: {image.shape}")
        #print(f"Dynamic crop size: {crop_size}x{crop_size}")
        # Define augmentation transformation
        augmentation_transform = A.Compose([
            A.Rotate(limit=augmentation_params.get("angle_limit", 45), p=augmentation_params.get("p_rotate", 1.0)),
            A.RandomCrop(width=crop_size, height=crop_size, p=augmentation_params.get("p_crop", 1.0)),
        ])
        
        # Convert polygons to masks
        masks, unique_labels = polygons_to_masks(image, polygons, labels)
        
        # Perform augmentations
        for i in range(num_augmentations):
            #print(f"Augmentation {i + 1}/{num_augmentations} for {json_path.stem}")
            
            # Augment image and masks
            aug_image, aug_masks = augment_image_and_masks(image, masks, augmentation_transform)
            #print(f"Augmented image shape: {aug_image.shape}")
            #print(f"Augmented masks shapes: {[m.shape for m in aug_masks]}")
            
            # Convert augmented masks to LabelMe-style polygons
            aug_polygons, aug_labels, aug_areas, shape_indices = masks_to_labelme_polygons(
                aug_masks, labels, original_areas, filename=json_path.stem, epsilon=2.0, area_threshold=0.05
            )
            
            # Debug polygon coordinates
            #print(f"Number of augmented polygons: {len(aug_polygons)}")
            for j, poly in enumerate(aug_polygons):
                poly_np = np.array(poly)
                min_x, min_y = poly_np.min(axis=0)
                max_x, max_y = poly_np.max(axis=0)
                #print(f"Polygon {j} bounds: x=[{min_x}, {max_x}], y=[{min_y}, {max_y}]")
                if max_x > aug_image.shape[1] or max_y > aug_image.shape[0] or min_x < 0 or min_y < 0:
                    print(f"WARNING: Polygon {j} exceeds augmented image bounds ({aug_image.shape[1]}x{aug_image.shape[0]})")
            
            # Save augmented data
            aug_img_path, aug_json_path = save_augmented_data(
                aug_image, aug_polygons, aug_labels, original_data, save_img_dir, save_json_dir, json_path.stem
            )
            # print(f"Saved augmented image: {aug_img_path}")
            # print(f"Saved augmented JSON: {aug_json_path}")
    return aug_img_path, aug_json_path
    

# --------------------------------------------
# Test Execution
# --------------------------------------------
if __name__ == "__main__":
    data_dir = "raw_dataset/train/images"
    json_dir = "raw_dataset/train/labels"
    save_img_dir = "raw_dataset/images"
    save_json_dir = "raw_dataset/labels"
    num_augmentations = 3
    
    augmentation_params = {
        "crop": True,
        "crop_scale": 0.9,
        "rotate": True,
        "angle_limit": 10,
        "flip_h": True,
        "flip_v": False,
        "p_crop": 0.6,
        "p_rotate": 1.0,
        "p_flip_h": 0.5,
        "p_flip_v": 0.5,
        "scale": False,
        "scale_factor": 1.2,
        "p_scale": 0.5,
        "color": False,
        "hue_shift_limit": 20,
        "sat_shift_limit": 30,
        "val_shift_limit": 20,
        "p_color": 0.5
    }
    
    #print("Running test augmentation...")
    augment_dataset(data_dir, json_dir, save_img_dir, save_json_dir, num_augmentations, augmentation_params)
    #print("Test augmentation completed.")
