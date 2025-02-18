import multiprocessing
import os
import random
import albumentations as A
import cv2
import time
from multiprocessing import Process, Queue
from pathlib import Path

from tqdm import tqdm

# Your existing transformations
common_transformations = [
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=45, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
    A.RandomGamma(p=0.5),
    A.CLAHE(p=0.5),
    A.RandomCrop(width=200, height=200, p=0.5),
    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
]


def process_folder(input_path, output_path, images_per_original=10):
    """Process a single folder and maintain structure"""
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Queue for collecting results
    queue_images = Queue()

    # Process each image in the folder
    for img_path in tqdm(input_path.glob('*'), desc=f"Processing ", unit="card"):
        #TODO fargli skippere le cartelle già aumentate
        #print(str(len(list(input_path.glob('*'))))+ f"immagini nella cartella {img_path}")
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            # Read and process image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Could not read image: {img_path}")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Calculate process distribution
            num_cpus = multiprocessing.cpu_count()
            process_batch_number = images_per_original // num_cpus
            process_batch_number_extra = images_per_original % num_cpus

            processes = []

            # Start processes
            for i in range(num_cpus):
                batch = process_batch_number + (process_batch_number_extra if i == 0 else 0)
                p = Process(target=generate_transformed_images,
                            args=(image.copy(), batch, queue_images))
                processes.append(p)
                p.start()

            # Collect results
            all_transformed_images = []
            for _ in range(num_cpus):
                try:
                    images = queue_images.get(timeout=30)
                    all_transformed_images.extend(images)
                except Exception as e:
                    print(f"Queue error for {img_path}: {e}")

            # Join processes
            for p in processes:
                p.join(timeout=1)

            # Save augmented images
            for i, img in enumerate(all_transformed_images):
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                output_filename = f"{img_path.stem}_aug_{i}{img_path.suffix}"
                output_file_path = output_path / output_filename
                cv2.imwrite(str(output_file_path), img_bgr)


def process_all_folders(root_folder):
    """Process all folders recursively"""
    root_path = Path(root_folder)

    # Walk through all directories
    for dir_path in root_path.rglob('*'):
        if dir_path.is_dir():
            # Create corresponding augmented directory
            relative_path = dir_path.relative_to(root_path)
            input_dir = dir_path
            output_dir = root_path.parent / f"{root_path.name}_aug" / relative_path

            print(f"Processing folder: {input_dir}")
            process_folder(input_dir, output_dir)


def generate_transformed_images(image, process_batch_number, images_queue=None):
    """Your existing transformation function"""
    transformed_images = []

    for _ in range(process_batch_number):
        num_transforms = random.randint(1, 5)
        selected_transforms = random.sample(common_transformations, num_transforms)
        transform = A.Compose(selected_transforms)
        transformed_image = transform(image=image)["image"]

        if images_queue is not None:
            transformed_images.append(transformed_image)

    if images_queue is not None:
        images_queue.put(transformed_images)


if __name__ == "__main__":
    start_time = time.time()

    # Replace with your root folder name
    root_folder = "card_images_low_png"

    print(f"Number of available CPU cores: {multiprocessing.cpu_count()}")
    process_all_folders(root_folder)

    print(f"Total execution time: {time.time() - start_time} seconds")