.PHONY: all figures panels paper clean help

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

all: figures paper  ## Regenerate panels, compose figures, and compile the paper

panels:  ## Regenerate source panels from CSV data
	python scripts/generate_source_panels.py

figures: panels  ## Compose final multi-panel PDF figures from source panels
	python scripts/generate_figures.py

paper: figures  ## Compile the LaTeX paper (two passes for cross-references)
	cd paper && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex

clean:  ## Remove generated files
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.bbl paper/*.blg
	rm -f paper/figures/*.pdf
	rm -f source_panels/*_regen.png
