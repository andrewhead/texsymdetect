# Symbol Detection Service

This project analyzes a TeX / LaTeX project and extracts the positions of all 
symbols from it.

## Setup

First, get and run the TeX compilation service following the instructions in 
[this repository](https://github.com/andrewhead/tex-compilation-service).

Then, build the Docker container for the service.

```bash
cd texsymdetect/service
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

The output will look like this:

```python
[Symbol(id_=0, mathml='<mi>max</mi>', tex='max', location=Location(left=850, top=741, width=39, height=9, page=1), parent=None),
 Symbol(id_=1, mathml='<msub><mi>h</mi><mi>θ</mi></msub>', tex='h_\\theta', location=Location(left=1021, top=661, width=18, height=17, page=1), parent=None),
 Symbol(id_=2, mathml='<msub><mi>h</mi><mi>θ</mi></msub>', tex='h_\\theta', location=Location(left=771, top=736, width=18, height=17, page=1), parent=None),
 Symbol(id_=3, mathml='<msub><mi>h</mi><mi>θ</mi></msub>', tex='h_\\theta', location=Location(left=908, top=1438, width=18, height=17, page=1), parent=None),
 ...
```

Each symbol can have one parent---i.e., another symbol that contains it. For instance, in the TeX symbol `x_i`, both `x` and `i` will have the parent symbol `x_i`.

## Advanced usage

By default, the service attempts to connect to a `texcompile` service hosted on 
port 8000 of http://localhost. It can be pointed to another `texcompile` 
endpoint by setting the `TEXCOMPILE_HOST` AND `TEXCOMPILE_PORT` environment 
variables.

The `detect_symbols` method in the client library can be passed additional 
options. Documentation and defaults for those options appear in the source code.

The client library also has a `parse_formulas` method, which requests for the 
service to parse the MathML for a list of LaTeX formulas.

## License

Apache 2.0.
