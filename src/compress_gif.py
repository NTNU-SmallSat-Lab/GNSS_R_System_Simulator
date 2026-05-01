import os

from PIL import Image


def compress_gif(input_path, output_path, max_width=400, colors=64, optimize=True, frame_skip=2):
    """
    =========================================================================================================
    Compress a GIF file while maintaining reasonable quality.

    INPUTS:
        input_path:  [str]   Path to input GIF file
        output_path: [str]   Path for output compressed GIF
        max_width:   [int]   Maximum width in pixels (height scales proportionally)
        colors:      [int]   Number of colors in palette (max 256)
        optimize:    [bool]  Enable GIF optimization
        frame_skip:  [int]   Skip every N frames (1=keep all, 2=skip every other frame)
    =========================================================================================================
    """

    try:
        # Open the GIF
        with Image.open(input_path) as img:
            frames = []

            # Get original dimensions
            original_width, original_height = img.size

            # Calculate new dimensions if needed
            if original_width > max_width:
                ratio = max_width / original_width
                new_width = max_width
                new_height = int(original_height * ratio)
            else:
                new_width, new_height = original_width, original_height

            # Process each frame
            frame_count = 0
            try:
                while True:
                    # Skip frames if needed
                    if frame_count % frame_skip == 0:
                        # Resize frame if needed
                        if (new_width, new_height) != (original_width, original_height):
                            frame = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        else:
                            frame = img.copy()

                        # Convert to palette mode with specified colors
                        if frame.mode != "P":
                            frame = frame.convert("P", palette=Image.ADAPTIVE, colors=colors)

                        frames.append(frame)

                    frame_count += 1
                    img.seek(img.tell() + 1)

            except EOFError:
                pass  # End of frames

            # Save compressed GIF
            if frames:
                # Adjust duration if frames were skipped
                original_duration = img.info.get("duration", 100)
                adjusted_duration = original_duration * frame_skip

                frames[0].save(
                    output_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=adjusted_duration,
                    loop=img.info.get("loop", 0),
                    optimize=optimize,
                    disposal=2,  # Clear frame before next
                )

                # Print compression results
                original_size = os.path.getsize(input_path)
                compressed_size = os.path.getsize(output_path)
                compression_ratio = (1 - compressed_size / original_size) * 100

                print(f"Original size: {original_size / 1024:.1f} KB")
                print(f"Compressed size: {compressed_size / 1024:.1f} KB")
                print(f"Size reduction: {compression_ratio:.1f}%")
                print(f"Dimensions: {original_width}x{original_height} → {new_width}x{new_height}")

            else:
                print("No frames found in GIF")

    except Exception as e:
        print(f"Error processing GIF: {e}")


def batch_compress_gifs(input_folder, output_folder, **kwargs):
    """
    =========================================================================================================
    Compress all GIF files in a folder.

    INPUTS:
        input_folder:  [str]   Path to folder containing GIF files
        output_folder: [str]   Path where compressed GIFs are written
        **kwargs:      [dict]  Forwarded to compress_gif (max_width, colors, optimize, frame_skip)
    =========================================================================================================
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    gif_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".gif")]

    for gif_file in gif_files:
        input_path = os.path.join(input_folder, gif_file)
        output_path = os.path.join(output_folder, f"compressed_{gif_file}")

        print(f"\nProcessing: {gif_file}")
        compress_gif(input_path, output_path, **kwargs)


# Example usage
if __name__ == "__main__":
    # Single file compression
    input_gif = "C:\\DocumentsLocal\\07_Code\\2_SMALLSAT\\GNSS_R\\GNSS_R_System_Sim\\OUTPUT\\Reflection_animation.gif"
    output_gif = "C:\\DocumentsLocal\\07_Code\\2_SMALLSAT\\GNSS_R\\GNSS_R_System_Sim\\OUTPUT\\Reflection_animation_compr.gif"

    # Compress with aggressive settings (target <50MB)
    compress_gif(input_gif, output_gif)

    # Or compress with ultra-aggressive settings for very small files
    # compress_gif(input_gif, output_gif, max_width=300, colors=32, frame_skip=3)

    # Batch compress all GIFs in a folder
    # batch_compress_gifs("input_folder", "output_folder")
