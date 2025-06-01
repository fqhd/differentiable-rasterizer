import cv2
import os
import subprocess

def make_video_from_frames(folder='frames', output='output_video.mp4', fps=30):
    def numerical_sort(filename):
        return int(os.path.splitext(filename)[0])  # e.g., "42.png" -> 42

    images = [img for img in os.listdir(folder) if img.endswith('.png')]
    images.sort(key=numerical_sort)

    if not images:
        print("No PNG images found in the folder.")
        return

    first_image_path = os.path.join(folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape

    out = cv2.VideoWriter('temp.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))

    for img_name in images:
        img_path = os.path.join(folder, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: Skipping frame {img_name}")
            continue
        out.write(frame)

    out.release()
    subprocess.run([
        'ffmpeg', '-i', 'temp.avi',
        '-vcodec', 'libx264',
        '-pix_fmt', 'yuv420p',
        output
    ])
    print(f"Video saved as {output}")

if __name__ == "__main__":
    make_video_from_frames()
