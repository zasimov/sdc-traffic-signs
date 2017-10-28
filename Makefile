writeup.pdf: writeup.md
	pandoc --from=markdown --to=latex writeup.md -o writeup.pdf
