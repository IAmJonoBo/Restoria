from __future__ import annotations

# TODO: Implement a simple worker using RQ/Celery/Arq.
# For now, this is a placeholder to document intended behavior:
# - Accept JobSpec
# - Resolve GPU affinity
# - Process images/videos in chunks
# - Emit progress over Redis/WS channels
# - Support resumable state files for long videos

def main():  # pragma: no cover
    print("Worker stub. Configure RQ/Celery and point to job runner here.")


if __name__ == "__main__":
    main()

