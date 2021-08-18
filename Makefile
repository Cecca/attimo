docs/src/*.html: src/*.rs
	docco -L docco.json src/*.rs

