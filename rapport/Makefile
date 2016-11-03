all:
	@echo "\033[0;32m[  0%]\033[0m Generating images..."
	@inkscape -D -z --file=c1p1s2_schema.svg --export-pdf=c1p1s2_schema.pdf --export-latex > /dev/null 2>&1
	@inkscape -D -z --file=c1p2s1_neuron.svg --export-pdf=c1p2s1_neuron.pdf --export-latex > /dev/null 2>&1
	@inkscape -D -z --file=c1p2s1_points.svg --export-pdf=c1p2s1_points.pdf --export-latex > /dev/null 2>&1
	@echo "\033[0;32m[ 25%]\033[0m First pass"
	@pdflatex -file-line-error DeepLearning.tex > /dev/null
	@echo "\033[0;32m[ 50%]\033[0m Creating glossary"
	@makeglossaries DeepLearning > /dev/null
	@echo "\033[0;32m[ 75%]\033[0m Final pass"
	@pdflatex -file-line-error DeepLearning.tex > /dev/null
	@echo "\033[0;32m[100%]\033[0m Completed"

clean:
	@echo "\033[0;32m*** Cleaning\033[0m"
	@rm -f *.aux
	@rm -f *.gls
	@rm -f *.out
	@rm -f *.glg
	@rm -f *.glo
	@rm -f *.glsdefs
	@rm -f *.pre
	@rm -f *.ist
	@rm -f *.log
	@rm -f *.blg
	@rm -f *.bls
	@echo "\033[0;32m*** Done\033[0m"
