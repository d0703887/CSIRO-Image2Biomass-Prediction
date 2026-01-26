import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def view_biomass_images(data_folder='../data/CSIRO', csv_file='train.csv'):
    # 1. Load the dataset
    try:
        df = pd.read_csv(os.path.join(data_folder, csv_file))
    except FileNotFoundError:
        print(f"Error: Could not find '{csv_file}'. Please check the file path.")
        return

    # 2. Pivot the table to organize data by image_path
    # CHANGED: Added 'Height_Ave_cm' to the index so it is preserved for each image
    pivot_df = df.pivot_table(
        index=['image_path', 'Species', 'Height_Ave_cm'],
        columns='target_name',
        values='target'
    ).fillna(0).reset_index()

    # Create lists of image paths, species, and height
    image_paths = pivot_df['image_path'].tolist()
    species_list = pivot_df['Species'].tolist()
    height_list = pivot_df['Height_Ave_cm'].tolist()  # NEW: Extract height list

    # Prepare the data dictionary
    biomass_data = pivot_df[['Dry_Green_g', 'Dry_Clover_g', 'Dry_Dead_g']].to_dict('records')

    if not image_paths:
        print("No data found in the CSV.")
        return

    # 3. Setup the viewer
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use a dictionary to store the current index (mutable state)
    state = {'index': 0}

    def update_plot():
        idx = state['index']
        img_path = os.path.join(data_folder, image_paths[idx])
        data = biomass_data[idx]
        current_species = species_list[idx]
        current_height = height_list[idx]  # NEW: Get current height

        ax.clear()

        # Try to load the image
        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.set_title(f"Species: {current_species} | Image: {img_path} ({idx + 1}/{len(image_paths)})",
                         fontsize=12, fontweight='bold')
        else:
            # Handle missing image files gracefully
            ax.text(0.5, 0.5, f"Image file not found:\n{img_path}",
                    ha='center', va='center', fontsize=12, color='red')
            ax.set_title(f"Image Missing ({idx + 1}/{len(image_paths)})")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        # Hide axes ticks for a cleaner look
        ax.set_xticks([])
        ax.set_yticks([])

        # Create the label with the biomass values AND height
        # CHANGED: Added Height to the label text
        label_text = (
            f"Species:    {current_species}\n"
            f"Height:     {current_height:.2f} cm\n"
            f"Dry Green:  {data['Dry_Green_g']:.4f}\n"
            f"Dry Clover: {data['Dry_Clover_g']:.4f}\n"
            f"Dry Dead:   {data['Dry_Dead_g']:.4f}"
        )

        # Display the text below the image
        ax.set_xlabel(label_text, fontsize=14, fontfamily='monospace', labelpad=15)

        # Refresh the plot
        fig.canvas.draw()

    def on_key(event):
        # 4. Handle Navigation
        if event.key in ['right', 'd']:
            state['index'] = (state['index'] + 1) % len(image_paths)
            update_plot()
        elif event.key in ['left', 'a']:
            state['index'] = (state['index'] - 1) % len(image_paths)
            update_plot()

    # Connect the keyboard event listener
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Initial plot
    update_plot()

    print("Viewer running...")
    print("Controls: [Left/Right] arrows or [a/d] keys to navigate.")
    plt.show()

if __name__ == "__main__":
    #view_biomass_images()

    import os
    import re

    folder = "../wandb"

    for run_name in os.listdir(folder):
        run_path = os.path.join(folder, run_name, "wandb")

        if not os.path.isdir(run_path):
            continue

        # Find the directory starting with 'run'
        subdirs = [d for d in os.listdir(run_path) if d.startswith("run")]
        if not subdirs:
            continue

        # Path to the actual files
        files_dir = os.path.join(run_path, subdirs[0], "files")
        if not os.path.isdir(files_dir):
            continue

        # 1. Gather all .pth files and their scores
        pth_files = []
        for filename in os.listdir(files_dir):
            if filename.endswith(".pth") and "_best_model_" in filename:
                try:
                    # Extract score: assumes "{fold_id}_best_model_{score}.pth"
                    # This splits by '_' and takes the part before '.pth'
                    score_str = filename.split("_best_model_")[-1].replace(".pth", "")
                    score = float(score_str)
                    pth_files.append((score, filename))
                except ValueError:
                    continue

        # 2. Sort files by score descending (highest first)
        pth_files.sort(key=lambda x: x[0], reverse=True)

        # 3. Identify files to delete (everything after index 2)
        files_to_delete = pth_files[3:]

        # 4. Execute deletion
        for score, filename in files_to_delete:
            file_to_remove = os.path.join(files_dir, filename)
            os.remove(file_to_remove)
            print(f"Deleted {filename} from {run_name} (Score: {score})")

    print("Cleanup complete.")

