This project analyzes a TeX / LaTeX project and extracts the positions of all symbols from it.

To run this command:

```bash
python extract_symbols.py \
  --sources archive.tgz
```

The output is in the form of a JSON array, where each item is a symbol:

```json
{
    "id": 1,
    "bounding_boxes": [
        {"left": ..., "top": ..., "width": ..., "height": ...},
        ...
    ],
    "tex": "\mathbf{x}",
    "mathml": "<mi mathvariant='bold'>x</mi>",
    "parent": 2
}
```