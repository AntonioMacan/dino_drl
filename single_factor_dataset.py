from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import json
from collections import defaultdict


class SingleFactorDataset(Dataset):
    """Dataset for single-factor variations  with metadata"""
    def __init__(self, root_dir, split='train', transform=None, use_metadata=True):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.use_metadata = use_metadata

        # Collect all image paths with metadata
        self.samples = []  # List of dicts with image_path, factor_group, factor_name, prompt, etc.

        split_dir = self.root_dir / f'{split}_single_factor'

        # Iterate through scenes (e.g., animal_scene, vehicle_scene)
        for scene_dir in sorted(split_dir.iterdir()):
            if not scene_dir.is_dir():
                continue

            scene_name = scene_dir.name

            # Iterate through object types (e.g., generated_images_cat)
            for obj_dir in sorted(scene_dir.iterdir()):
                if not obj_dir.is_dir():
                    continue

                obj_type = obj_dir.name.replace('generated_images_', '')

                # Load metadata JSON
                metadata = {}
                json_path = obj_dir / f'{obj_type}.json'
                if json_path.exists() and use_metadata:
                    with open(json_path, 'r') as f:
                        metadata = json.load(f)

                # Iterate through factor groups (e.g., eyes color, tail position)
                for group_dir in sorted(obj_dir.iterdir()):
                    if not group_dir.is_dir() or not group_dir.name.startswith('Group'):
                        continue

                    factor_group = group_dir.name

                    # Get prompts for this group from metadata
                    group_prompts = metadata.get(factor_group.replace('_', ':', 1), [])

                    # Collect images from this group
                    image_files = sorted(group_dir.glob('*.png'))
                    for img_idx, img_path in enumerate(image_files):
                        prompt = group_prompts[img_idx] if img_idx < len(group_prompts) else ""

                        self.samples.append({
                            'image_path': img_path,
                            'scene': scene_name,
                            'object_type': obj_type,
                            'factor_group': factor_group,
                            'prompt': prompt,
                            'variation_idx': img_idx,
                            'num_variations': len(image_files)
                        })

        print(f'Loaded {len(self.samples)} images from {split} split')
        if use_metadata:
            print(f'  With metadata from JSON files')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = Image.open(sample['image_path']).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, sample

    def get_factor_statistics(self):
        """Get statistics about factors in the dataset"""
        factor_counts = defaultdict(int)
        scene_counts = defaultdict(int)

        for sample in self.samples:
            factor_counts[sample['factor_group']] += 1
            scene_counts[sample['scene']] += 1

        return {
            'total_samples': len(self.samples),
            'unique_factors': len(factor_counts),
            'unique_scenes': len(scene_counts),
            'factor_counts': dict(factor_counts),
            'scene_counts': dict(scene_counts)
        }