import torch
import argparse
import os
import cv2
import numpy as np
import sys
from torchvision.transforms import v2
from torch.utils.data import DataLoader

# Custom imports
from model.DinoV3ViT import DinoV3ViT
from utils.utils import load_CSIRO
from dataset import CSIRODataset


# --- Helper Classes ---
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        tensor = tensor.clone()
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


# --- Visualization Helpers ---

def create_overlay(img_bgr, binary_mask_lowres, color=(0, 0, 255)):
    """
    Creates an overlay based on a low-res binary mask.
    Color is BGR. Default Red.
    """
    h, w = img_bgr.shape[:2]
    # Resize low-res mask to image size
    mask_resized = cv2.resize(binary_mask_lowres, (w, h), interpolation=cv2.INTER_NEAREST)

    # Colored overlay
    colored_mask = np.zeros_like(img_bgr)
    colored_mask[:, :, 0] = color[0]
    colored_mask[:, :, 1] = color[1]
    colored_mask[:, :, 2] = color[2]

    # Apply mask
    colored_mask = cv2.bitwise_and(colored_mask, colored_mask, mask=mask_resized)

    # Blend: img + overlay
    overlay = cv2.addWeighted(img_bgr, 1.0, colored_mask, 0.4, 0)
    return overlay


def text_on_img(img, text, pos=(20, 40), scale=1.0, color=(255, 255, 255), thickness=2):
    x, y = pos
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
    return img


# --- Inference Logic ---
def get_gates_and_metadata(model, device, data_dict, config):
    input_img_tensor = data_dict["Input_Img"].to(device)
    img_path = data_dict["image_path"][0]
    species = data_dict.get("species", data_dict.get("Species", ["Unknown"]))[0]

    # Extract ALL Ground Truths
    gts = {
        "Green": data_dict.get("Dry_Green_g", torch.tensor([0.0])).item(),
        "Clover": data_dict.get("Dry_Clover_g", torch.tensor([0.0])).item(),
        "Dead": data_dict.get("Dry_Dead_g", torch.tensor([0.0])).item()
    }

    if config["split_img"]:
        b, s, c, h, w = input_img_tensor.shape
        model_input = input_img_tensor.view(b * s, c, h, w)
    else:
        model_input = input_img_tensor

    with torch.no_grad():
        pred_dict = model(model_input, return_patch_preds=True)

    gate_keys = [k for k in pred_dict.keys() if "Tile" in k]
    gates = {k: pred_dict[k].cpu().numpy() for k in gate_keys}

    return gates, input_img_tensor.squeeze(0), img_path, species, gts


# --- OpenCV Interactive Viewers ---

def visualize_cv2_multiclass(gates, img_tensor, img_path, species, gts, config, unorm):
    # --- WINDOW SETUP ---
    # We now have two separate windows
    base_name = os.path.basename(img_path)
    win_input = f"Input View | {base_name}"
    win_masks = f"Mask Editor | {base_name}"

    cv2.namedWindow(win_input, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_masks, cv2.WINDOW_NORMAL)

    CLASSES = ["Green", "Clover", "Dead"]
    # Distinct colors for visualization overlays (BGR)
    CLASS_COLORS = {
        "Green": (0, 255, 0),  # Green
        "Clover": (255, 100, 0),  # Blueish
        "Dead": (0, 0, 255)  # Red
    }

    patch_h = config["input_h"] // 16
    patch_w = config["input_w"] // 16
    num_splits = 2 if config["split_img"] else 1

    # --- 1. Structure Data ---
    view_items = []

    for i in range(num_splits):
        # Process Input Image
        if config["split_img"]:
            img = unorm(img_tensor[i].cpu()).permute(1, 2, 0).numpy()
        else:
            img = unorm(img_tensor.cpu()).permute(1, 2, 0).numpy()

        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        item_data = {
            "img": img_bgr,
            "label": ["Left", "Right"][i] if config["split_img"] else "Full",
            "classes": {}
        }

        # Extract Class Data for this split
        for cls_name in CLASSES:
            gate_key = f"Tile_Dry_{cls_name}_g"
            if gate_key not in gates:
                continue

            raw = gates[gate_key]

            if config["split_img"]:
                if len(raw.shape) == 3 and raw.shape[0] == 2:
                    raw_split = raw[i]
                elif len(raw.shape) == 1:
                    raw_reshaped = raw.reshape(2, patch_h, patch_w)
                    raw_split = raw_reshaped[i]
                else:
                    raw_reshaped = raw.reshape(num_splits, patch_h, patch_w)
                    raw_split = raw_reshaped[i]
            else:
                raw_split = raw.reshape(patch_h, patch_w)

            item_data["classes"][cls_name] = {
                "raw_val": raw_split,
                "mask": np.zeros_like(raw_split, dtype=np.uint8)
            }

        view_items.append(item_data)

    # --- 2. Global Statistics ---
    class_stats = {}
    for cls_name in CLASSES:
        all_raw = []
        for item in view_items:
            if cls_name in item["classes"]:
                all_raw.append(item["classes"][cls_name]["raw_val"])

        if all_raw:
            concat_raw = np.concatenate([x.flatten() for x in all_raw])
            g_max = concat_raw.max()
            if g_max < 1e-5: g_max = 1.0
            class_stats[cls_name] = {"global_max": g_max, "thresh_ratio": 0.05}
        else:
            class_stats[cls_name] = {"global_max": 1.0, "thresh_ratio": 0.05}

    # --- 3. Interaction State ---
    state = {
        "view_items": view_items,
        "class_stats": class_stats,
        "brush_size": 1,
        "drawing_action": None,
        "img_h": view_items[0]["img"].shape[0],
        "img_w": view_items[0]["img"].shape[1],
        "grid_h": patch_h,
        "grid_w": patch_w,
        # UPDATED: Panel order only contains the classes, because Input is now in a separate window
        "panel_order": CLASSES
    }

    def apply_thresholds():
        for cls_name in CLASSES:
            thresh_val = class_stats[cls_name]["thresh_ratio"] * class_stats[cls_name]["global_max"]
            for item in state["view_items"]:
                if cls_name in item["classes"]:
                    raw = item["classes"][cls_name]["raw_val"]
                    item["classes"][cls_name]["mask"] = (raw > thresh_val).astype(np.uint8)

    apply_thresholds()

    # --- 4. Mouse Callback (Attached to Mask Editor) ---
    def modify_grid(x, y, action_val):
        # Determine Split (Column)
        split_idx = x // state["img_w"]
        if split_idx >= len(state["view_items"]): return

        # Determine Class (Row)
        # Since we removed "Input" from this window, y=0 starts at Green
        row_idx = y // state["img_h"]
        if row_idx >= len(state["panel_order"]): return

        target_panel = state["panel_order"][row_idx]

        # Local Coordinates
        local_x = x % state["img_w"]
        local_y = y % state["img_h"]

        # Map to Grid
        grid_x = int(local_x / state["img_w"] * state["grid_w"])
        grid_y = int(local_y / state["img_h"] * state["grid_h"])

        # Brush Logic
        bs = state["brush_size"]
        half_bs = bs // 2

        y_min = max(0, grid_y - half_bs)
        y_max = min(state["grid_h"], grid_y - half_bs + bs)
        x_min = max(0, grid_x - half_bs)
        x_max = min(state["grid_w"], grid_x - half_bs + bs)

        item = state["view_items"][split_idx]
        if target_panel in item["classes"]:
            item["classes"][target_panel]["mask"][y_min:y_max, x_min:x_max] = action_val
            update_display()

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                state["brush_size"] += 1
            else:
                state["brush_size"] = max(1, state["brush_size"] - 1)
            update_display()
        elif event == cv2.EVENT_LBUTTONDOWN:
            state["drawing_action"] = 1
            modify_grid(x, y, 1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            state["drawing_action"] = 0
            modify_grid(x, y, 0)
        elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
            state["drawing_action"] = None
        elif event == cv2.EVENT_MOUSEMOVE:
            if state["drawing_action"] is not None:
                modify_grid(x, y, state["drawing_action"])

    # Attach mouse interaction ONLY to the Mask Editor window
    cv2.setMouseCallback(win_masks, on_mouse)

    # --- 5. Trackbar Callbacks ---
    def on_trackbar_green(val):
        state["class_stats"]["Green"]["thresh_ratio"] = val / 100.0
        apply_thresholds()
        update_display()

    def on_trackbar_clover(val):
        state["class_stats"]["Clover"]["thresh_ratio"] = val / 100.0
        apply_thresholds()
        update_display()

    def on_trackbar_dead(val):
        state["class_stats"]["Dead"]["thresh_ratio"] = val / 100.0
        apply_thresholds()
        update_display()

    # --- 6. Render ---
    def update_display():
        final_input_cols = []
        final_mask_cols = []

        for item in state["view_items"]:
            # --- Build Input Window Column ---
            input_panel = item["img"].copy()
            text_on_img(input_panel, f"Input ({item['label']})")
            text_on_img(input_panel, f"Species: {species}", pos=(20, 80), scale=0.7)
            y_txt = 120
            for k, v in gts.items():
                text_on_img(input_panel, f"GT {k}: {v:.3f}g", pos=(20, y_txt), scale=1.2, color=(0, 0, 0))
                y_txt += 35
            final_input_cols.append(input_panel)

            # --- Build Mask Editor Column ---
            mask_v_stack = []
            for i, cls_name in enumerate(CLASSES):
                if cls_name in item["classes"]:
                    data = item["classes"][cls_name]
                    overlay = create_overlay(item["img"], data["mask"], color=CLASS_COLORS[cls_name])

                    # Add info
                    max_v = state["class_stats"][cls_name]["global_max"]
                    thresh_pct = int(state["class_stats"][cls_name]["thresh_ratio"] * 100)

                    # Add Brush Info to the top panel (Green) so user sees it while editing
                    if i == 0:
                        text_on_img(overlay, f"{cls_name} | Brush: {state['brush_size']} | Thresh: {thresh_pct}%",
                                    pos=(20, 40))
                    else:
                        text_on_img(overlay, f"{cls_name} | Thresh: {thresh_pct}%", pos=(20, 40))

                    mask_v_stack.append(overlay)
                else:
                    blank = np.zeros_like(item["img"])
                    text_on_img(blank, f"{cls_name} Missing")
                    mask_v_stack.append(blank)

            final_mask_cols.append(np.vstack(mask_v_stack))

        # Show Input Window
        input_canvas = np.hstack(final_input_cols)
        cv2.imshow(win_input, input_canvas)

        # Show Mask Window
        mask_canvas = np.hstack(final_mask_cols)
        cv2.imshow(win_masks, mask_canvas)

    # --- Window Init ---
    # Input Window Size
    h_in = state["img_h"]
    w_total = state["img_w"] * len(state["view_items"])
    # Resize Input window slightly smaller as it's just reference
    cv2.resizeWindow(win_input, int(w_total * 0.5), int(h_in * 0.5))

    # Mask Window Size (3 panels high)
    h_mask = state["img_h"] * 3
    # Scale to fit screen height approx 1000px
    target_h = 1000
    scale_ratio = target_h / h_mask
    target_w = int(w_total * scale_ratio)
    cv2.resizeWindow(win_masks, target_w, target_h)

    # Attach Trackbars to Mask Window
    cv2.createTrackbar("Green %", win_masks, 5, 100, on_trackbar_green)
    cv2.createTrackbar("Clover %", win_masks, 5, 100, on_trackbar_clover)
    cv2.createTrackbar("Dead %", win_masks, 5, 100, on_trackbar_dead)

    update_display()
    print(f"Controls: [Mouse] Draw on Mask Window, [Scroll] Brush Size, [n] Save & Next, [q] Quit")

    while True:
        # We need to waitKey generally; it catches events for all CV2 windows
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            cv2.destroyAllWindows()
            sys.exit(0)
        elif key == ord('n') or key == 32:
            # SAVE LOGIC (Unchanged)
            base_name = os.path.basename(img_path)
            for cls_name in CLASSES:
                masks = []
                for item in state["view_items"]:
                    if cls_name in item["classes"]:
                        masks.append(item["classes"][cls_name]["mask"])
                if not masks: continue
                final_mask = np.hstack(masks)
                final_mask = (final_mask * 255).astype(np.uint8)
                save_dir = f"data/CSIRO/new_pseudo_gates/{cls_name.lower()}"
                os.makedirs(save_dir, exist_ok=True)
                cv2.imwrite(os.path.join(save_dir, base_name), final_mask)
                print(f"Saved {cls_name} mask.")
            print(f"Finished {base_name}. Loading next...")
            break

    cv2.destroyWindow(win_input)
    cv2.destroyWindow(win_masks)


# --- Main ---
def main(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    print(f"Loading data from {config['data_folder']}...")
    df = load_CSIRO(config["data_folder"])
    dataset = CSIRODataset(config["data_folder"], df, config["input_h"], config["input_w"],
                           split_img=config["split_img"], is_train=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    print(f"Loading model: {config['model_name']}...")
    model = DinoV3ViT(
        model_name=config["model_name"],
        hidden_dim=config["hidden_dim"],
        training_mode="freeze_backbone",
        predict_height=False,
        split_img=config["split_img"]
    )

    if not os.path.exists(config["checkpoint_path"]):
        raise FileNotFoundError(f"Checkpoint not found at {config['checkpoint_path']}")

    model.load_state_dict(torch.load(config["checkpoint_path"], map_location=device))
    model.to(device)
    model.eval()

    print(f"\nStarting Multi-Class Interactive Session...")

    for i, data_dict in enumerate(dataloader):
        # Optional: Check if already processed (check if Green file exists for now)
        check_path = os.path.join(f"data/CSIRO/new_pseudo_gates/green", os.path.basename(data_dict["image_path"][0]))
        if os.path.exists(check_path):
            continue

        gates, img_tensor, img_path, species, gts = get_gates_and_metadata(model, device, data_dict, config)
        visualize_cv2_multiclass(gates, img_tensor, img_path, species, gts, config, unorm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-Class Interactive Gate Refinement")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    # target_class removed as we now do all simultaneously
    parser.add_argument("--data_folder", type=str, default="data/CSIRO")
    parser.add_argument("--model_name", type=str, default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--split_img", action="store_true")
    parser.add_argument("--input_h", type=int, default=1024)
    parser.add_argument("--input_w", type=int, default=2048)

    args = parser.parse_args()
    config = vars(args)

    main(config)