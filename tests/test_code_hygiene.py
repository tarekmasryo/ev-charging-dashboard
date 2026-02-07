from pathlib import Path


def test_no_arabic_characters_in_python_sources() -> None:
    root = Path(__file__).resolve().parents[1]

    skip_dirs = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "build",
        "dist",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    }

    py_files = [
        p
        for p in root.rglob("*.py")
        if not any(part in skip_dirs for part in p.relative_to(root).parts)
    ]

    arabic_ranges = [
        (0x0600, 0x06FF),
        (0x0750, 0x077F),
        (0x08A0, 0x08FF),
        (0xFB50, 0xFDFF),
        (0xFE70, 0xFEFF),
    ]

    offenders: list[str] = []
    for p in py_files:
        text = p.read_text(encoding="utf-8", errors="ignore")
        for ch in text:
            cp = ord(ch)
            if any(lo <= cp <= hi for lo, hi in arabic_ranges):
                offenders.append(str(p.relative_to(root)))
                break

    assert offenders == []
