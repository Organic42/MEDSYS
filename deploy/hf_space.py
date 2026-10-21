"""Publish MEDSYS as a public demo on a Hugging Face Space.

The Space builds Dockerfile.web (CPU only, public demo mode, see
publicdemo.py) on Hugging Face's free hardware. Log in first with a token
that has write access, then run from the repository root:

    hf auth login
    python deploy/hf_space.py <username>/<space-name>

Uploads the working tree's files that git does not ignore (no patient data,
outputs or build folders), minus what the app does not need (the paper and
its reviews, research results, tests), creating the Space if it does not
exist.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Not needed to run the demo; the paper material stays out until it is published.
SKIP = ('docs/paper/', 'docs/tier2-report/', 'research/results/', 'research/heldout_hypotheses.md', 'tests/',
        '.github/', 'deploy/')


def files_to_upload():
    out = subprocess.run(['git', 'ls-files', '--cached', '--others', '--exclude-standard'], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.splitlines()
    return sorted(f for f in out if os.path.isfile(os.path.join(ROOT, f)) and not f.startswith(SKIP))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('space', help='<username>/<space-name>')
    ap.add_argument('--private', action='store_true', help='create the Space as private')
    ap.add_argument('--dry-run', action='store_true', help='list the files and stop')
    args = ap.parse_args()
    files = files_to_upload()
    size = sum(os.path.getsize(os.path.join(ROOT, f)) for f in files) / 1e6
    print(f'{len(files)} files, {size:.1f} MB -> https://huggingface.co/spaces/{args.space}')
    if args.dry_run:
        print('\n'.join(files))
        return
    from huggingface_hub import HfApi
    api = HfApi()
    try:
        user = api.whoami()['name']
    except Exception:
        sys.exit('Not logged in to Hugging Face: run "hf auth login" first.')
    print(f'logged in as {user}')
    api.create_repo(args.space, repo_type='space', space_sdk='docker', private=args.private, exist_ok=True)
    api.upload_folder(folder_path=ROOT, repo_id=args.space, repo_type='space', allow_patterns=files,
                      commit_message='Deploy MEDSYS public demo')
    print(f'uploaded; the Space builds in a few minutes: https://huggingface.co/spaces/{args.space}')


if __name__ == '__main__':
    main()
