#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoImageProcessor,
    SegformerForSemanticSegmentation,
)


PALETTE = {
    "unlabeled": (0, 0, 0),
    "road": (128, 64, 128),
    "road_edge": (244, 35, 232),
    "terrain": (200, 215, 255),
    "grass": (0, 190, 0),
    "vegetation": (107, 142, 35),
    "rock": (140, 90, 40),
    "barrier": (250, 170, 30),
    "building": (70, 70, 70),
    "stands": (220, 220, 0),
    "billboard": (0, 220, 220),
    "sky": (70, 130, 180),
    "hero_car": (255, 0, 0),
    "opponent_car": (0, 0, 230),
}

LABEL_MAP = {
    "road": "road",
    "sidewalk": "road_edge",
    "building": "building",
    "wall": "barrier",
    "fence": "barrier",
    "pole": "building",
    "traffic light": "billboard",
    "traffic sign": "billboard",
    "vegetation": "vegetation",
    "terrain": "grass",
    "sky": "sky",
    "car": "opponent_car",
    "truck": "opponent_car",
    "bus": "opponent_car",
    "train": "opponent_car",
    "motorcycle": "opponent_car",
    "bicycle": "opponent_car",
}

VEHICLE_LABELS = {
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
}


def normalize_label(value):
    return str(value).strip().lower().replace("_", " ")


def choose_hero_component(vehicle_mask):
    mask = vehicle_mask.astype(np.uint8)

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    )

    height, width = mask.shape
    best_label = 0
    best_score = 0.0

    for component in range(1, count):
        area = int(
            stats[component, cv2.CC_STAT_AREA]
        )

        if area < 64:
            continue

        center_x, center_y = centroids[component]

        if center_y < height * 0.45:
            continue

        horizontal = max(
            0.0,
            1.0
            - abs(center_x - width / 2)
            / (width / 2),
        )

        vertical = center_y / height

        score = (
            area
            * (0.5 + horizontal)
            * (0.5 + vertical)
        )

        if score > best_score:
            best_score = score
            best_label = component

    if best_label == 0:
        return np.zeros_like(
            vehicle_mask,
            dtype=bool,
        )

    return labels == best_label


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
    )
    parser.add_argument(
        "--model",
        required=True,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1344,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=768,
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--hero-mode",
        choices=("bottom-center", "none"),
        default="bottom-center",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    args = parser.parse_args()

    source = Path(
        args.input
    ).expanduser().resolve()

    output_dir = Path(
        args.output_dir
    ).expanduser().resolve()

    model_path = Path(
        args.model
    ).expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(source)

    if not model_path.is_dir():
        raise FileNotFoundError(model_path)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(args.device)

    processor = AutoImageProcessor.from_pretrained(
        str(model_path),
        local_files_only=True,
    )

    model = (
        SegformerForSemanticSegmentation
        .from_pretrained(
            str(model_path),
            local_files_only=True,
        )
        .to(device)
    )

    model.eval()

    id2label = {
        int(index): normalize_label(label)
        for index, label
        in model.config.id2label.items()
    }

    processed = 0

    with av.open(str(source)) as container:
        for frame in container.decode(video=0):
            if (
                args.max_frames
                and processed >= args.max_frames
            ):
                break

            image = (
                frame.to_image()
                .convert("RGB")
                .resize(
                    (args.width, args.height),
                    Image.Resampling.LANCZOS,
                )
            )

            inputs = processor(
                images=image,
                return_tensors="pt",
            )

            pixel_values = inputs[
                "pixel_values"
            ].to(device)

            with torch.inference_mode():
                logits = model(
                    pixel_values=pixel_values
                ).logits

                logits = F.interpolate(
                    logits,
                    size=(
                        args.height,
                        args.width,
                    ),
                    mode="bilinear",
                    align_corners=False,
                )

                prediction = (
                    logits.argmax(dim=1)[0]
                    .cpu()
                    .numpy()
                )

            semantic = np.zeros(
                (
                    args.height,
                    args.width,
                    3,
                ),
                dtype=np.uint8,
            )

            vehicle_mask = np.zeros(
                (
                    args.height,
                    args.width,
                ),
                dtype=bool,
            )

            for class_id in np.unique(
                prediction
            ):
                label = id2label.get(
                    int(class_id),
                    "unlabeled",
                )

                pixels = prediction == class_id

                target = LABEL_MAP.get(
                    label,
                    "unlabeled",
                )

                semantic[pixels] = PALETTE[
                    target
                ]

                if label in VEHICLE_LABELS:
                    vehicle_mask |= pixels

            if (
                args.hero_mode
                == "bottom-center"
            ):
                hero_mask = (
                    choose_hero_component(
                        vehicle_mask
                    )
                )

                semantic[hero_mask] = PALETTE[
                    "hero_car"
                ]

            output_path = (
                output_dir
                / f"{processed:06d}.png"
            )

            Image.fromarray(
                semantic,
                mode="RGB",
            ).save(
                output_path,
                compress_level=1,
            )

            processed += 1

            if processed % 25 == 0:
                print(
                    "PROGRESS",
                    processed,
                    flush=True,
                )

    metadata = {
        "input": str(source),
        "model": str(model_path),
        "frames": processed,
        "width": args.width,
        "height": args.height,
        "hero_mode": args.hero_mode,
        "palette": PALETTE,
    }

    (
        output_dir / "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "DONE",
        json.dumps(metadata),
    )


if __name__ == "__main__":
    main()
