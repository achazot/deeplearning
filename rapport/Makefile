all:
	@echo "\033[0;32m*** Generating images...\033[0m"
	@inkscape -D -z --file=c1p1s2_schema.svg --export-pdf=c1p1s2_schema.pdf --export-latex
	@inkscape -D -z --file=c1p2s1_neuron.svg --export-pdf=c1p2s1_neuron.pdf --export-latex
	@echo "\033[0;32m*** First pass\033[0m"
	@pdflatex -file-line-error DeepLearning.tex
	@echo "\033[0;32m*** Glossary pass\033[0m"
	@makeglossaries DeepLearning
	@echo "\033[0;32m*** Final pass\033[0m"
	@pdflatex -file-line-error DeepLearning.tex
