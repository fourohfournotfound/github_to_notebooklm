# 📦 RepoToText

Convert GitHub repository ZIP files into clean, LLM-friendly text format. Perfect for use with **NotebookLM**, **Claude**, **ChatGPT**, and other AI tools.

## Features

- **🔒 Privacy-First**: All processing happens locally in your browser/server—your code is never sent anywhere else
- **📁 Smart Filtering**: Automatically filters out binary files, large files, and common junk directories (`__pycache__`, `node_modules`, `.git`, etc.)
- **🌳 Directory Tree**: Generates a clean ASCII tree view of your project structure
- **📝 Multiple Output Formats**: XML tags (best for LLMs), Markdown, or plain text
- **⚙️ Configurable**: Adjust file size limits, choose which file types to include
- **🐍 Python-Focused**: Optimized for Python projects, with optional support for config files, docs, web files, and more

## How to Use

### 1. Download your repo as a ZIP

1. Go to your GitHub repository
2. Click the green **Code** button
3. Select **Download ZIP**

This works with both **public and private repositories**!

### 2. Run the app

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### 3. Upload and configure

1. Upload your ZIP file
2. Adjust filters in the sidebar (file types, size limits)
3. Copy the output or download as TXT
4. Paste into NotebookLM, Claude, or your favorite LLM

## Output Format

The generated output includes:

1. **Header**: Repository name and description
2. **Statistics**: File counts and sizes
3. **Directory Tree**: ASCII visualization of project structure
4. **File Contents**: Each file wrapped in clear separators

Example output structure:
```
# my-project - Code Summary

## Statistics
- Files included: 15
- Total content size: 45.2 KB

## Directory Structure
├── src/
│   ├── main.py
│   └── utils/
│       └── helpers.py
└── tests/
    └── test_main.py

## File Contents

<file path="src/main.py">
def main():
    print("Hello, world!")
</file>
```

## Configuration Options

### File Types
- **Python** (always included): `.py`, `.pyx`, `.pyi`, `.pyw`
- **Config Files**: `.toml`, `.yaml`, `.yml`, `.json`, `.ini`
- **Documentation**: `.md`, `.rst`, `.txt`
- **Web**: `.html`, `.css`, `.js`, `.jsx`, `.ts`, `.tsx`
- **Data**: `.csv`, `.xml`
- **Shell**: `.sh`, `.bash`
- **Jupyter**: `.ipynb`

### Size Limits
- **Max file size**: Skip individual files over this limit (default: 100KB)
- **Max total output**: Warning when total output exceeds this (default: 5MB)

### Output Styles
- **XML** (default): `<file path="...">content</file>` - Best for most LLMs
- **Markdown**: Code blocks with syntax highlighting
- **Plain**: Simple separators

## Deploying to Streamlit Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy from your fork
4. Share the link!

Since all processing is client-side (happens in the Streamlit server you control), you can safely use this on Streamlit Cloud with private repos.

## Tips for NotebookLM

1. Keep output under 500KB for best results
2. Use XML format (default) for clearest file boundaries
3. Include config files to help the LLM understand dependencies
4. Add documentation files for context

## License

MIT License - feel free to use and modify!
