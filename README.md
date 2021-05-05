# Symbol Detection Service

This project analyzes a TeX / LaTeX project and extracts the positions of all 
symbols from it.

## Setup

First, get and run the TeX compilation service following the instructions in 
[this repository](https://github.com/andrewhead/tex-compilation-service).

Then, build the Docker container for the service.

```bash
docker build -t symbol-detection-service .
```

## Start the service

```bash
# Hosts the service on port 8001 of localhost.
docker run -p 8001:80 -it symbol-detection-service
```

## Query the service

The service takes as input a directory containing a TeX/LaTeX project (without 
any prior compilation results or auxiliary files), and returns a list of 
generated PDFs and PostScript files.

Say, for example, you wish to compile the LaTeX project for arXiv paper 
1601.00978. First, fetch the sources for the project:

```bash
wget https://arxiv.org/e-print/1601.00978 --user-agent "Name <email>"
```

Then, unpack the sources into a directory:

```bash
mkdir sources
tar xzvf 1601.00978 -C sources
```

Queries to the service can be made through a dedicated client library, which you 
can install as follows:

```bash
pip install git+https://github.com/andrewhead/texsymdetect
```

Once the client library is installed, you can make a request like so:

```python
from texsymdetect.client import detect_symbols

symbols = detect_symbols(sources_dir='sources')
```

Inspect the detected symbols:

```python
print(symbols)
```

The output will look sort of like this.

```python
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

## License

Apache 2.0.
