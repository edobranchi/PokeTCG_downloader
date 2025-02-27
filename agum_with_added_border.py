import cv2
import numpy as np
from pathlib import Path
import albumentations as A
import random
from scipy.ndimage import gaussian_filter


# Agumentation of the card and adding a border around the card in a random way
# take an input root an output root and a number of agumented cards needed in output.

# ES. num_aguments = 30 ----> generates 30 differents augumented copy of each card


def create_diverse_background(width, height, style='random'):
    """Create different types of backgrounds"""
    background = np.zeros((height, width, 3), dtype=np.uint8)

    style = random.choice(['wood', 'fabric', 'gradient', 'noise', 'dark']) if style == 'random' else style

    if style == 'wood':
        base_color = np.array([160, 120, 80]) + np.random.randint(-20, 20, 3)
        background[:] = base_color
        for i in range(0, height, random.randint(10, 20)):
            cv2.line(background, (0, i), (width, i + random.randint(-5, 5)),
                     [int(x + random.randint(-20, 20)) for x in base_color],
                     random.randint(1, 3))
    elif style == 'fabric':
        base_color = np.random.randint(50, 200, 3)
        background[:] = base_color
        noise = np.random.randint(-15, 15, (height, width, 3))
        background = np.clip(background + noise, 0, 255).astype(np.uint8)
        background = gaussian_filter(background, sigma=0.5)
    elif style == 'gradient':
        color1 = np.random.randint(50, 200, 3)
        color2 = np.random.randint(50, 200, 3)
        for i in range(height):
            factor = i / height
            background[i] = color1 * (1 - factor) + color2 * factor
    elif style == 'dark':
        base_color = np.random.randint(20, 60, 3)
        background[:] = base_color
        noise = np.random.randint(-10, 10, (height, width, 3))
        background = np.clip(background + noise, 0, 255).astype(np.uint8)
    else:
        background = np.random.randint(30, 80, (height, width, 3), dtype=np.uint8)
        background = gaussian_filter(background, sigma=2.0)

    return background


def apply_diverse_lighting(image):
    h, w = image.shape[:2]
    effect = random.choice(['vignette', 'spotlight', 'shadow', 'none'])

    if effect == 'vignette':
        Y, X = np.ogrid[:h, :w]
        center = (h / 2, w / 2)
        dist = np.sqrt((Y - center[0]) ** 2 + (X - center[1]) ** 2)
        mask = 1 - (dist / (np.sqrt(h ** 2 + w ** 2) / 2)) ** 2
        mask = np.clip(mask * 1.5, 0, 1)
        mask = np.dstack([mask] * 3)
        image = (image * mask).astype(np.uint8)
    elif effect == 'spotlight':
        center_x = w // 2 + random.randint(-w // 4, w // 4)
        center_y = h // 2 + random.randint(-h // 4, h // 4)
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((Y - center_y) ** 2 + (X - center_x) ** 2)
        mask = 1 - np.clip(dist / (np.sqrt(h ** 2 + w ** 2) / 4), 0, 1)
        mask = np.dstack([mask] * 3)
        image = cv2.addWeighted(image, 0.7, (image * mask).astype(np.uint8), 0.3, 0)

    return image


def apply_safe_perspective(image):
    h, w = image.shape[:2]
    intensity = random.choice(['mild', 'medium', 'strong'])
    margin = {'mild': 0.03, 'medium': 0.05, 'strong': 0.07}[intensity]
    max_shift = int(min(h, w) * margin)

    src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_points = np.float32([
        [random.randint(0, max_shift), random.randint(0, max_shift)],
        [w - random.randint(0, max_shift), random.randint(0, max_shift)],
        [w - random.randint(0, max_shift), h - random.randint(0, max_shift)],
        [random.randint(0, max_shift), h - random.randint(0, max_shift)]
    ])

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


def augment_single_card(card_path):
    card = cv2.imread(str(card_path))
    card = cv2.cvtColor(card, cv2.COLOR_BGR2RGB)

    h, w = card.shape[:2]
    padding_ratio = random.uniform(0.15, 0.25)
    pad_w, pad_h = int(w * padding_ratio), int(h * padding_ratio)
    new_w, new_h = w + 2 * pad_w, h + 2 * pad_h

    transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.8),
        A.OneOf([
            A.RandomGamma(gamma_limit=(70, 130), p=1.0),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
        ], p=0.7),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.ISONoise(p=1.0),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
        ], p=0.6),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.5),
    ])

    augmented_card = transform(image=card)['image']
    augmented_card = apply_safe_perspective(augmented_card)
    background = create_diverse_background(new_w, new_h)

    mask = np.ones_like(augmented_card, dtype=np.float32)
    edge_width = random.randint(2, 4)
    mask[:edge_width, :] = np.linspace(0, 1, edge_width)[:, np.newaxis, np.newaxis]
    mask[-edge_width:, :] = np.linspace(1, 0, edge_width)[:, np.newaxis, np.newaxis]
    mask[:, :edge_width] *= np.linspace(0, 1, edge_width)[np.newaxis, :, np.newaxis]
    mask[:, -edge_width:] *= np.linspace(1, 0, edge_width)[np.newaxis, :, np.newaxis]

    result = background.copy()
    result[pad_h:pad_h + h, pad_w:pad_w + w] = (
        augmented_card * mask +
        background[pad_h:pad_h + h, pad_w:pad_w + w] * (1 - mask)
    ).astype(np.uint8)

    result = apply_diverse_lighting(result)
    overlay = result.copy()
    thickness = random.randint(1, 3)
    cv2.rectangle(overlay, (pad_w, pad_h), (pad_w + w, pad_h + h), (255, 255, 255), thickness)
    alpha = random.uniform(0.8, 0.9)
    return cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)


def process_folder(input_folder, output_folder, num_augments=10):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    for img_path in input_folder.glob('*.png'):
        for i in range(num_augments):
            augmented_img = augment_single_card(img_path)
            output_file_path = output_folder / f"{img_path.stem}_aug_{i}{img_path.suffix}"
            cv2.imwrite(str(output_file_path), cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR))


def process_all_folders(root_input_folder, root_output_folder, num_augments=10):
    root_input_folder = Path(root_input_folder)
    root_output_folder = Path(root_output_folder)

    for folder in root_input_folder.iterdir():
        print("Processing folder " + folder.name)
        if folder.is_dir():
            output_folder = root_output_folder / folder.name
            process_folder(folder, output_folder, num_augments)


if __name__ == "__main__":
    input_root = "./card_images_low_png/"
    output_root = "./card_images_low_png_aug/"
    process_all_folders(input_root, output_root, num_augments=30)
