#!/usr/bin/env python3
"""Create separate, author-written diagnostic texts; no download or model code."""
from __future__ import annotations

import argparse
from pathlib import Path

TRAIN = """A red fox sat beside the gate. The gate was open and the fox went in.
A blue bird flew over the tree. The bird saw a small seed on the path.
Mina put a cup on the table. She filled the cup with cool water.
The sun came out after the rain. The wet leaves shone in the light.
A small dog ran to the pond. It stopped at the edge and looked at a frog.
The frog jumped into the water. The dog sat down and waited.
Ravi found a green stone near the hill. He put it back beside the path.
A child opened a book in the quiet room. The book told a story about a boat.
The boat moved slowly on the river. Two ducks followed it under a bridge.
At night the family closed the gate. The fox slept under a tall tree.
In the morning a bell rang. Mina took her bag and walked to the garden.
She saw the blue bird again. This time it carried a leaf in its beak.
The old cat watched from the wall. It stretched its legs and went to sleep.
Ravi made a little boat from paper. He placed it in a bowl of water.
The paper boat floated for a while. Then Ravi lifted it out to dry.
A warm wind moved the branches. The leaves made a soft sound above the path.
The child read one more page. Then she closed the book and turned off the lamp.
"""

VALID = """A little fox found a bowl beside the wall. It drank some water and walked away.
Mina sat under the tree with a book. A green leaf fell onto the open page.
The dog watched two birds near the pond. They flew over the gate into the garden.
Ravi took his paper boat to the table. He told a story about a long river.
When the rain stopped, the child opened the door. The sun shone on the wet path.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=False)
    for name, text in (("train.bin", TRAIN), ("valid.bin", VALID)):
        payload = text.encode("utf-8")
        (args.directory / name).write_bytes(payload)
        print(f"{name}: {len(payload)} bytes")
    (args.directory / "SOURCE.txt").write_text(
        "Author-written diagnostic fixture. Training and validation contain distinct "
        "sentences but intentionally similar vocabulary. Not TinyStories or a "
        "representative language benchmark. UTF-8, LF, no normalization.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
