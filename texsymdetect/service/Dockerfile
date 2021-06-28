FROM ubuntu:bionic

RUN apt-get update

# Install Python
RUN apt-get install -y python3.7
RUN apt-get install -y python3-distutils
RUN apt-get install -y python3.7-dev
RUN apt-get install -y curl
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1
RUN python get-pip.py --force-reinstall

# Install Node.js
RUN curl -sL https://deb.nodesource.com/setup_12.x | bash
RUN apt-get install -y nodejs

# Let ImageMagick to read and write PDFs
RUN apt-get install -y imagemagick imagemagick-6-common imagemagick-6.q16 \
  libmagick++-6.q16-7 libmagickcore-6.q16-3 libmagickcore-6.q16-3-extra libmagickwand-6.q16-3
RUN sed -i '/pattern="PDF"/s/rights="none"/rights="read | write"/' /etc/ImageMagick-6/policy.xml

# Install GhostScript
RUN apt-get install -y ghostscript

# Install Node.js dependencies
WORKDIR /sources/node
RUN apt-get install -y git
COPY node/package*.json ./
RUN npm install
# As KaTeX is undergoing rapid changes, install it from sources
RUN npm install -g yarn
RUN npm run prepare-katex
RUN npm run install-katex

# Install custom fork of LaTeXML for expanding macros.
WORKDIR /sources/perl
RUN apt-get install -y cpanminus
RUN apt-get install -y build-essential
RUN apt-get install -y libxml-libxslt-perl
RUN cpanm LWP Text::Unidecode Archive::Zip JSON::XS IO::String File::Which XML::LibXSLT XML::LibXML Image::Size
COPY perl/setup.sh ./
RUN chmod +x setup.sh
RUN ./setup.sh

# Install vim for when we inevitably want to inspect files
RUN apt-get install -y vim

# Install pip requireements
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy over the source code
WORKDIR /sources
COPY . .

# Set up path to include Python scripts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
