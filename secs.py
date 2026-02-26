import soundfile as sf
import sys
from pathlib import Path


def get_duration(filepath: str) -> float:
    """Return duration of an audio file in seconds."""
    with sf.SoundFile(filepath) as f:
        return len(f) / f.samplerate


def compare_audio_files(file1: str, file2: str) -> None:
    for fp in (file1, file2):
        if not Path(fp).exists():
            raise FileNotFoundError(f"File not found: {fp}")

    dur1 = get_duration(file1)
    dur2 = get_duration(file2)
    diff = abs(dur1 - dur2)

    print(f"File 1: {file1}")
    print(f"  Duration: {dur1:.4f}s")
    print(f"File 2: {file2}")
    print(f"  Duration: {dur2:.4f}s")
    print(f"Difference: {diff:.4f}s ({diff * 1000:.2f}ms)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python audio_diff.py <file1> <file2>")
        sys.exit(1)

    compare_audio_files(sys.argv[1], sys.argv[2])