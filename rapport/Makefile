all:
	@echo "\033[0;32m[  0%]\033[0m Generating images..."
	@inkscape -D -z --file=c1p1s2_schema.svg --export-pdf=c1p1s2_schema.pdf > /dev/null
	@inkscape -D -z --file=c1p2s1_neuron.svg --export-pdf=c1p2s1_neuron.pdf > /dev/null
	@inkscape -D -z --file=c1p2s1_points.svg --export-pdf=c1p2s1_points.pdf > /dev/null
	@inkscape -D -z --file=c2p1s1_rcnn.svg --export-pdf=c2p1s1_rcnn.pdf > /dev/null
	@inkscape -D -z --file=c2p2s2_hog.svg --export-pdf=c2p2s2_hog.pdf > /dev/null
	@echo "\033[0;32m[ 25%]\033[0m First pass"
	@pdflatex -file-line-error DeepLearning.tex
	@echo "\033[0;32m[ 50%]\033[0m Creating glossary"
	@makeglossaries DeepLearning
	@echo "\033[0;32m[ 75%]\033[0m Final pass"
	@pdflatex -file-line-error DeepLearning.tex
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
