# Fetch sources for AI2 clone of LaTeXML.
git clone https://github.com/allenai/latexml latexml

# Check out a stable commit with features for expanding TeX macros.
cd latexml
git checkout cad185fcec509977a5e45224d2de6b6f0f9f3e26

# Install Perl dependencies for LaTeXML.
cpanm .
