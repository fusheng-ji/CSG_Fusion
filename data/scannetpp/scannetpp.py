import json
import logging
import os

import cv2
import numpy as np

from data.data import crop_resize_if_necessary, DUST3RSplattingDataset, DUST3RSplattingTestDataset
from src.mast3r_src.dust3r.dust3r.utils.image import imread_cv2

logger = logging.getLogger(__name__)


class ScanNetPPData():

    def __init__(self, root, stage):

        self.root = root
        self.stage = stage
        self.png_depth_scale = 1000.0

        # Dictionaries to store the data for each scene
        self.color_paths = {}
        self.depth_paths = {}
        self.intrinsics = {}
        self.c2ws = {}

        # Fetch the sequences to use
        if stage == "train":
            sequence_file = os.path.join(self.root, "raw", "splits", "nvs_sem_train.txt")
            bad_scenes = ['303745abc7']
        elif stage == "val" or stage == "test":
            sequence_file = os.path.join(self.root, "raw", "splits", "nvs_sem_val.txt")
            bad_scenes = ['cc5237fd77']
        with open(sequence_file, "r") as f:
            self.sequences = f.read().splitlines()

        # Remove scenes that have frames with no valid depths
        logger.info(f"Removing scenes that have frames with no valid depths: {bad_scenes}")
        self.sequences = [s for s in self.sequences if s not in bad_scenes]

        # The split files can list more scenes than were downloaded / have a coverage file (the coverage files
        # define the training set). Skip those, so a partial download still trains on everything it has.
        missing = [s for s in self.sequences
                   if not os.path.isfile(os.path.join(self.root, 'raw', 'data', s, 'dslr', 'train_test_lists.json'))
                   or not os.path.isfile(os.path.join(self.root, 'coverage', f'{s}.json'))]
        if missing:
            logger.warning(f"Skipping {len(missing)} / {len(self.sequences)} {stage} scenes without downloaded data "
                           f"or coverage file")
            self.sequences = [s for s in self.sequences if s not in missing]

        P = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]]
        ).astype(np.float32)

        # Collect information for every sequence
        scenes_with_no_good_frames = []
        for sequence in self.sequences:

            input_raw_folder = os.path.join(self.root, 'raw', 'data', sequence)
            input_processed_folder = os.path.join(self.root, 'processed', sequence)

            # Load Train & Test Splits
            frame_file = os.path.join(input_raw_folder, "dslr", "train_test_lists.json")
            with open(frame_file, "r") as f:
                train_test_list = json.load(f)

            # Camera Metadata
            cams_metadata_path = f"{input_processed_folder}/dslr/nerfstudio/transforms_undistorted.json"
            with open(cams_metadata_path, "r") as f:
                cams_metadata = json.load(f)

            # Load the nerfstudio/transforms.json file to check whether each image is blurry
            nerfstudio_transforms_path = f"{input_raw_folder}/dslr/nerfstudio/transforms.json"
            with open(nerfstudio_transforms_path, "r") as f:
                nerfstudio_transforms = json.load(f)

            # Create a reverse mapping from image name to the frame information and nerfstudio transform
            # (as transforms_undistorted.json does not store the frames in the same order as train_test_lists.json)
            file_path_to_frame_metadata = {}
            file_path_to_nerfstudio_transform = {}
            for frame in cams_metadata["frames"]:
                file_path_to_frame_metadata[frame["file_path"]] = frame
            for frame in nerfstudio_transforms["frames"]:
                file_path_to_nerfstudio_transform[frame["file_path"]] = frame

            # Fetch the pose for every frame
            sequence_color_paths = []
            sequence_depth_paths = []
            sequence_c2ws = []
            for train_file_name in train_test_list["train"]:
                is_bad = file_path_to_nerfstudio_transform[train_file_name]["is_bad"]
                if is_bad:
                    continue
                sequence_color_paths.append(f"{input_processed_folder}/dslr/undistorted_images/{train_file_name}")
                sequence_depth_paths.append(f"{input_processed_folder}/dslr/undistorted_depths/{train_file_name.replace('.JPG', '.png')}")
                frame_metadata = file_path_to_frame_metadata[train_file_name]
                c2w = np.array(frame_metadata["transform_matrix"], dtype=np.float32)
                c2w = P @ c2w @ P.T
                sequence_c2ws.append(c2w)

            if len(sequence_color_paths) == 0:
                logger.info(f"No good frames for sequence: {sequence}")
                scenes_with_no_good_frames.append(sequence)
                continue

            # Get the intrinsics data for the frame
            K = np.eye(4, dtype=np.float32)
            K[0, 0] = cams_metadata["fl_x"]
            K[1, 1] = cams_metadata["fl_y"]
            K[0, 2] = cams_metadata["cx"]
            K[1, 2] = cams_metadata["cy"]

            self.color_paths[sequence] = sequence_color_paths
            self.depth_paths[sequence] = sequence_depth_paths
            self.c2ws[sequence] = sequence_c2ws
            self.intrinsics[sequence] = K

        # Remove scenes with no good frames
        self.sequences = [s for s in self.sequences if s not in scenes_with_no_good_frames]

    def get_view(self, sequence, view_idx, resolution):

        # RGB Image
        rgb_path = self.color_paths[sequence][view_idx]
        rgb_image = imread_cv2(rgb_path)

        # Depthmap
        depth_path = self.depth_paths[sequence][view_idx]
        depthmap = imread_cv2(depth_path, cv2.IMREAD_UNCHANGED)
        depthmap = depthmap.astype(np.float32)
        depthmap = depthmap / self.png_depth_scale

        # C2W Pose
        c2w = self.c2ws[sequence][view_idx]

        # Camera Intrinsics
        intrinsics = self.intrinsics[sequence]

        # Resize
        rgb_image, depthmap, intrinsics = crop_resize_if_necessary(
            rgb_image, depthmap, intrinsics, resolution
        )

        view = {
            'original_img': rgb_image,
            'depthmap': depthmap,
            'camera_pose': c2w,
            'camera_intrinsics': intrinsics,
            'dataset': 'scannet++',
            'label': f"scannet++/{sequence}",
            'instance': f'{view_idx}',
            'is_metric_scale': True,
            'sky_mask': depthmap <= 0.0,
        }
        return view


def get_scannet_dataset(root, stage, resolution, num_epochs_per_epoch=1):

    data = ScanNetPPData(root, stage)

    coverage, shifted = {}, []
    for sequence in data.sequences:
        with open(os.path.join(root, 'coverage', f'{sequence}.json'), 'r') as f:
            sequence_coverage = json.load(f)
        coverage[sequence] = sequence_coverage[sequence]
        if len(coverage[sequence]) != len(data.color_paths[sequence]):
            shifted.append(sequence)
    # The official v1 download can flag more frames as `is_bad` than the data the coverage files were made from. The
    # coverage matrix then refers to other frames: skip those scenes (the test sets drop them too).
    if shifted:
        logger.warning(f"Skipping {len(shifted)} / {len(data.sequences)} {stage} scenes whose usable frames differ from "
                       f"their coverage file: {shifted}")
        data.sequences = [s for s in data.sequences if s not in shifted]
        coverage = {s: coverage[s] for s in data.sequences}

    dataset = DUST3RSplattingDataset(
        data,
        coverage,
        resolution,
        num_epochs_per_epoch=num_epochs_per_epoch,
    )

    return dataset


def get_scannet_test_dataset(root, alpha, beta, resolution, use_every_n_sample=100):

    data = ScanNetPPData(root, 'val')

    samples_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'test_set_{alpha}_{beta}.json')
    print(f"Loading samples from: {samples_file}")
    with open(samples_file, 'r') as f:
        all_samples = json.load(f)

    # The official v1 download can flag more frames as `is_bad` than the data the test lists were made from.
    # Frame indices then shift in that scene and some point past its last usable frame. Drop every sample of
    # such a scene (for every method, so all comparisons stay on the same set): with shifted indices the
    # remaining samples are not the intended view triplets either.
    # Decided on the full list, so that the result does not depend on the subsampling.
    samples = _drop_shifted_scenes(data, all_samples[::use_every_n_sample], all_samples)

    dataset = DUST3RSplattingTestDataset(data, samples, resolution)

    return dataset


def _frame_ids(sample):
    """All frame indices of a test sample: [scene, c1, c2, target] or [scene, [c1, ..., cN], target]."""
    return [i for x in sample[1:] for i in (x if isinstance(x, list) else [x])]


def _drop_shifted_scenes(data, samples, all_samples):
    n_frames = {s: len(p) for s, p in data.color_paths.items()}
    shifted = {s[0] for s in all_samples if s[0] not in n_frames or max(_frame_ids(s)) >= n_frames[s[0]]}
    shifted = sorted(shifted & {s[0] for s in samples})
    if shifted:
        kept = [s for s in samples if s[0] not in shifted]
        print(f"Skipping {len(samples) - len(kept)} / {len(samples)} samples of scene(s) {shifted}, whose usable "
              f"frames differ from the data the test set was made from")
        samples = kept
    return samples


def get_scannet_multiview_test_dataset(root, samples_file, num_context, resolution):
    """Multi-view test set (scripts/eval/make_multiview_testset.py): [scene, [c1, ..., cN], target] samples, of
    which the first ``num_context`` context views are used (the sets are nested, so every N scores the same
    targets)."""
    data = ScanNetPPData(root, 'val')
    print(f"Loading samples from: {samples_file}")
    with open(samples_file, 'r') as f:
        samples = json.load(f)
    if num_context > min(len(s[1]) for s in samples):
        raise ValueError(f'{samples_file} has fewer than {num_context} context views per sample')
    return DUST3RSplattingTestDataset(data, _drop_shifted_scenes(data, samples, samples), resolution, num_context=num_context)
