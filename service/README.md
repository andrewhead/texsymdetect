This project analyzes a TeX / LaTeX project and extracts the positions of all symbols from it.

First, get and run the TeX compilation service following the instructions in [this repository](https://github.com/andrewhead/tex-compilation-service). 

```bash
docker build -t symbol-extraction-service .
```

```bash
# Hosts the service on port 8001 of localhost.
docker run -p 8001:80 -it tex-compilation-service
```


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