# PDF Summary Assistant

A Python CLI that reads text from a PDF and produces a concise summary.

It supports two workflows:

1. **Direct summarizer**: extract text and summarize in one pass.
2. **Agent mode**: a simple map-reduce style "AI agent" that summarizes chunks and then synthesizes a final summary.

## Features

- Reads PDF files with `pypdf`
- Cleans and chunks long documents
- Two summarization backends:
  - `heuristic` (default): no API keys required
  - `openai`: uses OpenAI chat completions if `OPENAI_API_KEY` is set
- Agent mode for large documents
- CLI output can be printed or written to a file

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1) Basic summary (heuristic)

```bash
python pdf_summary.py --pdf ./example.pdf --mode direct --backend heuristic
```

### 2) Agentic summary using OpenAI

```bash
export OPENAI_API_KEY="your_key"
python pdf_summary.py --pdf ./example.pdf --mode agent --backend openai
```

### 3) Save summary to file

```bash
python pdf_summary.py --pdf ./example.pdf --output summary.txt
```

## CLI options

- `--pdf` (required): path to a PDF file
- `--mode`: `direct` or `agent` (default: `direct`)
- `--backend`: `heuristic` or `openai` (default: `heuristic`)
- `--chunk-size`: approx words per chunk for agent mode (default: `450`)
- `--max-bullets`: target number of bullets in final summary (default: `6`)
- `--output`: optional output file path

## Notes

- The heuristic summarizer is extractive and deterministic.
- The OpenAI backend is abstractive and usually more natural.
- For scanned/image-only PDFs, OCR is not included in this script.
