from pathlib import Path
from PIL import Image

def main():
    img_dir = Path(__file__).resolve().parent.parent / 'Images'
    if not img_dir.exists():
        print('Images directory not found:', img_dir)
        return
    entries = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
            try:
                with Image.open(p) as im:
                    w, h = im.size
                entries.append((p.name, w, h))
            except Exception as e:
                print('Failed to read', p.name, e)
    for name, w, h in entries:
        print(f"{name},{w},{h}")

if __name__ == '__main__':
    main()
