.PHONY: setup segment clean

# Stage 0 only. Vision and OCR deps install per stage - see pyproject.
setup:
	uv sync

# make segment CLIP=data/clips/foo.mp4
segment:
	uv run ft segment $(CLIP)

clean:
	rm -rf work/*/ && touch work/.gitkeep
