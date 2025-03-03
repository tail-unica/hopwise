sphinx-apidoc --ext-autodoc --ext-doctest --ext-intersphinx --ext-mathjax -d 5 -o docs/source/hopwise hopwise --separate
sphinx-build -M html docs/source docs/build