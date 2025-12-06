import numpy as np
import json
import colorsys
import cv2
import os
import torch
from colorthief import ColorThief


class VisionModule:

    def __init__(self, feature_extraction_model, configuration):
        self.handcraft_features = None
        self.feature_extraction_model = feature_extraction_model
        self.configuration = configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_map = {
            "normal": 0,
            "Keratoconus": 1
        }
        self.test_label_map = {
            0: "normal",
            1: "Keratoconus"
        }

    def crop(self, img, img_name="UNKNOWN", save_dir=None):

        crops = {}
        crop_cfg = self.configuration["crop_positions"]

        for key, cfg in crop_cfg.items():

            # --- 1. Crop the region ---
            x1, y1, x2, y2 = cfg["x1"], cfg["y1"], cfg["x2"], cfg["y2"]
            crop_img = img[y1:y2, x1:x2].copy()

            # --- 2. Apply circular mask ONLY if mask_radius_factor exists ---
            if cfg.get("apply_circular_mask", False) is True:
                radius = cfg.get("radius", None)
                center = cfg.get("center", None)
                center = (center[0] - x1, center[1] - y1)
                crop_img = self._apply_circular_mask(
                    crop_img,
                    radius,
                    center
                )

            # --- 3. Optionally save cropped image ---
            if save_dir is not None and key != "text_panel":
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{img_name}_{key}.png")
                cv2.imwrite(save_path, cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))

            # --- 4. Add to output dictionary ---
            crops[key] = crop_img

        return crops

    def _apply_circular_mask(self, img, radius, center):
        h, w = img.shape[:2]

        if center is not None:
            center = (int(center[0]), int(center[1]))
        else:
            center = (int(w // 2), int(h // 2))

        if radius is not None:
            radius = int(radius)
        else:
            radius = int(min(h, w) * 0.9 / 2)

        mask = np.zeros((h, w), dtype=np.uint8)

        # make white circle on a center
        cv2.circle(mask, center, radius, (255, 255, 255), -1)

        mask = mask.astype(float) / 255.0

        fill_color = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        fill_color = np.array(fill_color, dtype=np.float32)

        fill_img = np.ones_like(img, dtype=np.float32) * fill_color

        result = img.astype(float) * mask[..., None] + fill_img * (1 - mask[..., None])

        return result.astype(np.uint8)

    def preprocess_for_cnn(self, cropped_img):
        # resize
        size = self.configuration["resize"]
        img = cv2.resize(cropped_img, size)

        # convert to float32 0–1
        img = img.astype(np.float32) / 255.0

        # ImageNet norm (change if you use custom model)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        img = (img - mean) / std

        # HWC → CHW
        img = np.transpose(img, (2, 0, 1))

        return img

    def run_image_preprocessing(self, image_path):

        class_dir = os.path.dirname(image_path)  # e.g. dataset/normal/images
        base_dir = os.path.dirname(class_dir)  # e.g. dataset/normal

        # save_text_dir   = os.path.join(base_dir, "text_panels")
        save_crops_dir = os.path.join(base_dir, "crops")

        # os.makedirs(save_text_dir, exist_ok=True)
        os.makedirs(save_crops_dir, exist_ok=True)

        img_name = os.path.splitext(os.path.basename(image_path))[0]

        bgr = cv2.imread(image_path)
        if bgr is None:
            raise FileNotFoundError(f"Could not read: {image_path}")

        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        crops = self.crop(img, img_name=img_name, save_dir=save_crops_dir)

        # Save all crops for debugging / training
        for key, crop_rgb in crops.items():
            if key == "text_panel":
                continue
            else:
                out_path = os.path.join(save_crops_dir, f"{img_name}_{key}.png")
                cv2.imwrite(out_path, cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))

        # # Save text panel for OCR
        # text_crop_rgb = crops["text_panel"]
        # text_path = os.path.join(save_text_dir, f"{img_name}_textpanel.png")
        #
        # cv2.imwrite(text_path, cv2.cvtColor(text_crop_rgb, cv2.COLOR_RGB2BGR))

        return save_crops_dir

    def run_vision_preprocessing(self, folder_path):

        exts = (".jpg", ".jpeg", ".png", ".bmp")
        all_images = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(exts)
        ]

        print(f"[INFO] Found {len(all_images)} images in: {folder_path}")

        # parent class folder (normal or Keratoconus)
        class_dir = os.path.dirname(folder_path)

        # resolved main output dirs
        crops_dir = os.path.join(class_dir, "crops")
        # deep_dir = os.path.join(class_dir, "deep_features")
        # hand_dir = os.path.join(class_dir, "handcraft_features")
        # text_panel_dir = os.path.join(class_dir, "text_panels")

        processed_list = []

        for img_path in all_images:
            try:
                print(f"[INFO] Processing: {img_path}")
                self.run_image_preprocessing(img_path)

                processed_list.append(os.path.basename(img_path))

            except Exception as e:
                print(f"[ERROR] Failed on {img_path}: {e}")
                continue

        print("[INFO] Completed folder processing.")

        return {
            "crops": crops_dir,
            "processed_images": processed_list
        }

    def extract_deep_features(self, tensor, save_dir=None, img_name=None, quadrant=None):

        t = torch.from_numpy(tensor).unsqueeze(0).float().to(self.device)
        # shape (1,3,224,224) for resnet 50

        with torch.no_grad():
            deep_features = self.feature_extraction_model(t).cpu().numpy().flatten()
            # === OPTIONAL SAVE ===
        if save_dir is not None:

            # Ensure folder exists
            os.makedirs(save_dir, exist_ok=True)

            # Build filename
            # Example: "image001_Q1.npy"
            if img_name is None:
                img_name = "unknown"

            if quadrant is None:
                quadrant = "QX"

            filename = f"{img_name}_{quadrant}.npy"
            output_path = os.path.join(save_dir, filename)

            # Save features
            np.save(output_path, deep_features)
        return deep_features

    def handcrafted_features(self, cropped_img_pth, save_dir=None, img_name=None, quadrant=None):

        colors_platte = ColorThief(cropped_img_pth)
        dominant_colors = colors_platte.get_palette(color_count=2)
        # print(dominant_colors)

        ignored_colors = [
            (0, 0, 0),
            (255, 255, 255),
            (int(0.485 * 255), int(0.456 * 255), int(0.406 * 255))  # fill mask base color
        ]
        tol = 15
        cleaned_colors = []

        for R, G, B in dominant_colors:
            bad = False
            for ir, ig, ib in ignored_colors:
                if (abs(R - ir) <= tol and
                        abs(G - ig) <= tol and
                        abs(B - ib) <= tol):
                    bad = True
                    break

            if not bad:
                cleaned_colors.append((R, G, B))

        if len(cleaned_colors) == 0:
            cleaned_colors = [(0, 0, 0)]

        R, G, B = cleaned_colors[0]
        r, g, b = R / 255.0, G / 255.0, B / 255.0

        h, s, v = colorsys.rgb_to_hsv(r, g, b)

        dom_h = h
        dom_s = s
        dom_v = v

        # ---- FIXED BGR → RGB for cv2 ----
        cropped_img = cv2.imread(cropped_img_pth)
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

        img = cropped_img.astype(np.float32) / 255.0
        hsv = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2HSV).astype(np.float32)

        # Create mask of "valid" pixels matrix/grid
        valid_mask = np.ones((img.shape[0], img.shape[1]), dtype=bool)

        for ir, ig, ib in ignored_colors:
            if (ir, ig, ib) == ignored_colors[2]:
                # This is the FILL MASK → use tolerance
                invalid = (
                        (np.abs(cropped_img[:, :, 0] - ir) < tol) &
                        (np.abs(cropped_img[:, :, 1] - ig) < tol) &
                        (np.abs(cropped_img[:, :, 2] - ib) < tol)
                )
            else:
                # Black & white → exact match
                invalid = (
                        (cropped_img[:, :, 0] == ir) &
                        (cropped_img[:, :, 1] == ig) &
                        (cropped_img[:, :, 2] == ib)
                )

            valid_mask[invalid] = False

        # Valid pixels only
        valid_hsv = hsv[valid_mask]

        # Intensity Features (based on V channel)
        V = valid_hsv[:, 2] / 255.0

        avg_intensity = float(V.mean())
        std_intensity = float(V.std())

        # Center vs Periphery intensities, the center sharpness
        h_im, w_im = cropped_img.shape[:2]
        cy, cx = h_im // 2, w_im // 2
        R = min(cx, cy)

        r_center = int(R * 0.3)
        r_mid = int(R * 0.60)

        Y, X = np.ogrid[:h_im, :w_im]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

        center_mask = dist < r_center
        periphery_mask = (dist > r_mid) & (dist < R)

        center_vals = img[center_mask][:, :].mean() if center_mask.any() else 0.0
        periphery_vals = img[periphery_mask][:, :].mean() if periphery_mask.any() else 1e-6

        center_intensity = float(center_vals)
        periphery_intensity = float(periphery_vals)
        center_periphery_ratio = float(center_intensity / periphery_intensity)

        # Quadrant splits and asymmetry ratios
        top = img[:h_im // 2, :, :].mean()
        bottom = img[h_im // 2:, :, :].mean()
        left = img[:, :w_im // 2, :].mean()
        right = img[:, w_im // 2:, :].mean()

        inferior_superior_ratio = float(bottom / (top + 1e-6))
        left_right_ratio = float(left / (right + 1e-6))

        diag1 = img[:h_im // 2, :w_im // 2, :].mean() - img[h_im // 2:, w_im // 2:, :].mean()
        diag2 = img[:h_im // 2, w_im // 2:, :].mean() - img[h_im // 2:, :w_im // 2, :].mean()

        diag1_difference = float(diag1)
        diag2_difference = float(diag2)

        radial_symmetry = float(
            abs(top - bottom) +
            abs(left - right) +
            abs(diag1) +
            abs(diag2)
        )

        # ----- PACK RESULTS -----
        features = {
            "dom_h": dom_h,
            "dom_s": dom_s,
            "dom_v": dom_v,

            "avg_intensity": avg_intensity,
            "std_intensity": std_intensity,

            "center_intensity": center_intensity,
            "periphery_intensity": periphery_intensity,
            "center_periphery_ratio": center_periphery_ratio,

            "inferior_superior_ratio": inferior_superior_ratio,
            "left_right_ratio": left_right_ratio,

            "diag1_difference": diag1_difference,
            "diag2_difference": diag2_difference,

            "radial_symmetry": radial_symmetry
        }
        self.handcraft_features = features

        # ============== OPTIONAL SAVE ==============
        if save_dir is not None:

            os.makedirs(save_dir, exist_ok=True)

            if img_name is None:
                img_name = "unknown"

            if quadrant is None:
                quadrant = "QX"

            filename = f"{img_name}_{quadrant}.json"
            output_path = os.path.join(save_dir, filename)

            # save handcrafted features as JSON
            with open(output_path, "w") as f:
                json.dump(features, f, indent=4)

        return features
