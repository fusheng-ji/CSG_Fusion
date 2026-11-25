"""Upload the release checkpoints, the model card and the license files to the Hugging Face Hub (one repository).

    huggingface-cli login
    python scripts/release/export_checkpoint.py results/<run>/checkpoints/last.ckpt checkpoint/csg_fusion.ckpt
    python scripts/release/upload_hf.py --repo-id fushengji/CSG-Fusion [--private] [--card-only]

The model card is docs/hf_model_card.md (uploaded as README.md). demo.py downloads the checkpoints from the repository
id set in its HF_REPO constant.
"""
import argparse
import os

from huggingface_hub import HfApi

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FILES = {'README.md': 'docs/hf_model_card.md', 'LICENSE': 'LICENSE',
         'MASt3R_CHECKPOINTS_NOTICE': 'src/mast3r_src/CHECKPOINTS_NOTICE', 'VGGT_LICENSE.txt': 'third_party/vggt/LICENSE.txt'}
CHECKPOINTS = ('csg_fusion.ckpt', 'csg_fusion_vggt.ckpt')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo-id', required=True, help='e.g. <user>/CSG-Fusion')
    parser.add_argument('--card-only', action='store_true', help='upload the model card and license files only')
    parser.add_argument('--private', action='store_true', help='create the repository as private')
    args = parser.parse_args()

    files = dict(FILES)
    if not args.card_only:
        files.update({name: f'checkpoint/{name}' for name in CHECKPOINTS})
    api = HfApi()
    api.create_repo(args.repo_id, repo_type='model', private=args.private, exist_ok=True)
    for target, source in files.items():
        api.upload_file(path_or_fileobj=os.path.join(ROOT, source), path_in_repo=target, repo_id=args.repo_id,
                        commit_message=f'Add {target}')
    print(f'https://huggingface.co/{args.repo_id}')


if __name__ == '__main__':
    main()
