"""
RepoToText - Convert GitHub Repository Zips to LLM-Friendly Text
A Streamlit app for processing GitHub zip files into clean, formatted text
suitable for NotebookLM, Claude, ChatGPT, and other LLMs.
"""

import streamlit as st
import zipfile
import io
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from collections import defaultdict

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class FilterConfig:
    """Configuration for file filtering."""
    # File extensions to include (lowercase, with dot)
    include_extensions: set = field(default_factory=lambda: {
        '.py', '.pyx', '.pyi', '.pyw',  # Python
    })
    
    # Additional extensions that can be optionally included
    optional_extensions: dict = field(default_factory=lambda: {
        'Config Files': {'.toml', '.yaml', '.yml', '.json', '.ini', '.cfg', '.conf'},
        'Documentation': {'.md', '.rst', '.txt'},
        'Requirements': {'.txt'},  # requirements.txt handled specially
        'Web (HTML/CSS/JS)': {'.html', '.htm', '.css', '.js', '.jsx', '.ts', '.tsx'},
        'Data': {'.csv', '.xml'},
        'Shell Scripts': {'.sh', '.bash', '.zsh'},
        'Jupyter Notebooks': {'.ipynb'},
    })
    
    # Directories to always skip
    skip_directories: set = field(default_factory=lambda: {
        '__pycache__', '.git', '.svn', '.hg', '.bzr',
        'node_modules', 'venv', 'env', '.venv', '.env',
        '.tox', '.pytest_cache', '.mypy_cache', '.ruff_cache',
        'dist', 'build', 'eggs', '*.egg-info',
        '.idea', '.vscode', '.vs',
        'htmlcov', '.coverage', 'coverage',
        '.ipynb_checkpoints',
    })
    
    # Files to always skip
    skip_files: set = field(default_factory=lambda: {
        '.DS_Store', 'Thumbs.db', '.gitignore', '.gitattributes',
        'package-lock.json', 'yarn.lock', 'poetry.lock',
    })
    
    # Size limits
    max_file_size_kb: int = 500  # Default max file size in KB
    max_total_size_kb: int = 10000  # Default max total output size in KB


# =============================================================================
# File Processing
# =============================================================================

@dataclass
class ProcessedFile:
    """Represents a processed file from the repository."""
    path: str
    content: str
    size_bytes: int
    extension: str


def should_skip_path(path: str, config: FilterConfig) -> bool:
    """Check if a path should be skipped based on directory rules."""
    parts = Path(path).parts
    for part in parts:
        if part in config.skip_directories:
            return True
        # Handle wildcards like *.egg-info
        for skip_pattern in config.skip_directories:
            if '*' in skip_pattern:
                pattern = skip_pattern.replace('*', '')
                if part.endswith(pattern):
                    return True
    return False


def get_extension(filename: str) -> str:
    """Get the file extension, handling special cases."""
    name = filename.lower()
    # Handle special files
    if name in {'dockerfile', 'makefile', 'procfile'}:
        return f'.{name}'
    return Path(filename).suffix.lower()


def process_zip_file(
    zip_bytes: bytes,
    config: FilterConfig,
    allowed_extensions: set,
    include_requirements: bool = True,
) -> tuple[list[ProcessedFile], dict]:
    """
    Process a zip file and extract relevant files.
    
    Returns:
        Tuple of (list of ProcessedFile, stats dict)
    """
    processed_files = []
    stats = {
        'total_files': 0,
        'processed_files': 0,
        'skipped_too_large': 0,
        'skipped_binary': 0,
        'skipped_extension': 0,
        'skipped_directory': 0,
        'total_size_bytes': 0,
    }
    
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for file_info in zf.filelist:
            # Skip directories
            if file_info.is_dir():
                continue
            
            stats['total_files'] += 1
            filepath = file_info.filename
            
            # Remove the top-level directory that GitHub adds (repo-branch/)
            parts = Path(filepath).parts
            if len(parts) > 1:
                # Skip the first part (repo-branch directory)
                clean_path = str(Path(*parts[1:]))
            else:
                clean_path = filepath
            
            # Skip if empty path
            if not clean_path:
                continue
            
            filename = Path(clean_path).name
            
            # Skip hidden files (except .env.example type files)
            if filename.startswith('.') and filename not in {'.env.example', '.env.sample'}:
                if filename not in config.skip_files:
                    stats['skipped_extension'] += 1
                continue
            
            # Skip files in excluded directories
            if should_skip_path(clean_path, config):
                stats['skipped_directory'] += 1
                continue
            
            # Skip explicitly excluded files
            if filename in config.skip_files:
                continue
            
            # Check extension
            ext = get_extension(filename)
            
            # Special handling for requirements.txt
            is_requirements = filename.lower() in {
                'requirements.txt', 'requirements-dev.txt', 
                'requirements-test.txt', 'dev-requirements.txt',
                'test-requirements.txt'
            }
            
            if not (ext in allowed_extensions or (is_requirements and include_requirements)):
                stats['skipped_extension'] += 1
                continue
            
            # Check file size
            if file_info.file_size > config.max_file_size_kb * 1024:
                stats['skipped_too_large'] += 1
                continue
            
            # Try to read the file
            try:
                content = zf.read(filepath)
                # Try to decode as UTF-8
                try:
                    text_content = content.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        text_content = content.decode('latin-1')
                    except:
                        stats['skipped_binary'] += 1
                        continue
                
                # Skip binary-looking files
                if '\x00' in text_content:
                    stats['skipped_binary'] += 1
                    continue
                
                processed_files.append(ProcessedFile(
                    path=clean_path,
                    content=text_content,
                    size_bytes=len(content),
                    extension=ext,
                ))
                stats['processed_files'] += 1
                stats['total_size_bytes'] += len(content)
                
            except Exception as e:
                stats['skipped_binary'] += 1
                continue
    
    # Sort files by path for consistent output
    processed_files.sort(key=lambda f: f.path.lower())
    
    return processed_files, stats


# =============================================================================
# Output Formatting
# =============================================================================

def build_tree_structure(files: list[ProcessedFile]) -> str:
    """Build an ASCII tree representation of the directory structure."""
    if not files:
        return "No files found."
    
    # Build tree structure
    tree = {}
    for f in files:
        parts = Path(f.path).parts
        current = tree
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        # Mark files with None
        current[parts[-1]] = None
    
    def render_tree(node: dict, prefix: str = "", is_last: bool = True) -> list[str]:
        lines = []
        items = sorted(node.items(), key=lambda x: (x[1] is not None, x[0].lower()))
        
        for i, (name, subtree) in enumerate(items):
            is_last_item = (i == len(items) - 1)
            connector = "└── " if is_last_item else "├── "
            
            if subtree is None:
                # It's a file
                lines.append(f"{prefix}{connector}{name}")
            else:
                # It's a directory
                lines.append(f"{prefix}{connector}{name}/")
                extension = "    " if is_last_item else "│   "
                lines.extend(render_tree(subtree, prefix + extension, is_last_item))
        
        return lines
    
    return "\n".join(render_tree(tree))


def format_output(
    files: list[ProcessedFile],
    stats: dict,
    repo_name: str = "Repository",
    include_tree: bool = True,
    include_stats: bool = True,
    separator_style: str = "xml",
) -> str:
    """Format the processed files into LLM-friendly text output."""
    sections = []
    
    # Header
    sections.append(f"# {repo_name} - Code Summary")
    sections.append("")
    sections.append("This document contains the source code from a Python repository,")
    sections.append("formatted for analysis by language models.")
    sections.append("")
    
    # Statistics
    if include_stats:
        sections.append("## Statistics")
        sections.append(f"- Files included: {stats['processed_files']}")
        sections.append(f"- Total files scanned: {stats['total_files']}")
        sections.append(f"- Files skipped (too large): {stats['skipped_too_large']}")
        sections.append(f"- Files skipped (binary/unreadable): {stats['skipped_binary']}")
        sections.append(f"- Files skipped (excluded extension): {stats['skipped_extension']}")
        sections.append(f"- Files skipped (excluded directory): {stats['skipped_directory']}")
        total_kb = stats['total_size_bytes'] / 1024
        sections.append(f"- Total content size: {total_kb:.1f} KB")
        sections.append("")
    
    # Directory structure
    if include_tree:
        sections.append("## Directory Structure")
        sections.append("```")
        sections.append(build_tree_structure(files))
        sections.append("```")
        sections.append("")
    
    # File contents
    sections.append("## File Contents")
    sections.append("")
    
    for f in files:
        if separator_style == "xml":
            sections.append(f'<file path="{f.path}">')
            sections.append(f.content.rstrip())
            sections.append('</file>')
        elif separator_style == "markdown":
            sections.append(f"### `{f.path}`")
            sections.append("")
            # Determine language for syntax highlighting
            lang = f.extension.lstrip('.') if f.extension else ''
            lang_map = {
                'py': 'python', 'pyx': 'python', 'pyi': 'python',
                'js': 'javascript', 'jsx': 'javascript',
                'ts': 'typescript', 'tsx': 'typescript',
                'yml': 'yaml', 'md': 'markdown',
            }
            lang = lang_map.get(lang, lang)
            sections.append(f"```{lang}")
            sections.append(f.content.rstrip())
            sections.append("```")
        else:  # plain
            sections.append(f"{'='*60}")
            sections.append(f"FILE: {f.path}")
            sections.append(f"{'='*60}")
            sections.append(f.content.rstrip())
        
        sections.append("")
    
    return "\n".join(sections)


# =============================================================================
# Streamlit UI
# =============================================================================

def main():
    st.set_page_config(
        page_title="RepoToText",
        page_icon="📦",
        layout="wide",
    )
    
    st.title("📦 RepoToText")
    st.markdown("""
    Convert GitHub repository ZIP files into clean, LLM-friendly text format.
    Perfect for use with **NotebookLM**, **Claude**, **ChatGPT**, and other AI tools.
    
    **Privacy Note**: All processing happens locally in this app—your code is never sent anywhere else.
    """)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # File type selection
    st.sidebar.subheader("📁 File Types")
    
    config = FilterConfig()
    
    # Always include Python
    st.sidebar.markdown("**✅ Python files** (always included)")
    
    # Optional file types
    selected_optional = {}
    for category, extensions in config.optional_extensions.items():
        selected_optional[category] = st.sidebar.checkbox(
            f"{category}",
            value=(category == "Config Files"),
            help=f"Include: {', '.join(sorted(extensions))}"
        )
    
    include_requirements = st.sidebar.checkbox(
        "requirements*.txt files",
        value=True,
        help="Include requirements.txt and variants"
    )
    
    # Size limits
    st.sidebar.subheader("📏 Size Limits")
    max_file_size = st.sidebar.slider(
        "Max file size (KB)",
        min_value=50,
        max_value=2000,
        value=500,
        step=50,
        help="Skip files larger than this"
    )
    
    max_total_size = st.sidebar.slider(
        "Max total output (KB)",
        min_value=500,
        max_value=20000,
        value=10000,
        step=500,
        help="Warning threshold for total output size"
    )
    
    # Output format options
    st.sidebar.subheader("📝 Output Format")
    separator_style = st.sidebar.selectbox(
        "File separator style",
        options=["xml", "markdown", "plain"],
        index=0,
        help="XML tags work best for most LLMs"
    )
    
    include_tree = st.sidebar.checkbox("Include directory tree", value=True)
    include_stats = st.sidebar.checkbox("Include statistics", value=True)
    
    # Build allowed extensions set
    allowed_extensions = config.include_extensions.copy()
    for category, is_selected in selected_optional.items():
        if is_selected:
            allowed_extensions.update(config.optional_extensions[category])
    
    # Update config with user settings
    config.max_file_size_kb = max_file_size
    config.max_total_size_kb = max_total_size
    
    # Main content area
    st.header("📤 Upload Repository ZIP")
    
    st.markdown("""
    **How to get a ZIP from GitHub:**
    1. Go to your repository on GitHub
    2. Click the green **Code** button
    3. Select **Download ZIP**
    """)
    
    uploaded_file = st.file_uploader(
        "Drop your GitHub ZIP file here",
        type=['zip'],
        help="Upload a ZIP file downloaded from GitHub"
    )
    
    if uploaded_file is not None:
        # Get repo name from filename
        repo_name = uploaded_file.name.replace('.zip', '').replace('-main', '').replace('-master', '')
        repo_name = st.text_input("Repository name", value=repo_name)
        
        with st.spinner("Processing repository..."):
            try:
                zip_bytes = uploaded_file.read()
                files, stats = process_zip_file(
                    zip_bytes,
                    config,
                    allowed_extensions,
                    include_requirements,
                )
                
                if not files:
                    st.warning("No matching files found in the ZIP. Try adjusting the file type filters.")
                    return
                
                # Show results
                st.success(f"✅ Processed {stats['processed_files']} files!")
                
                # Statistics in columns
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Files Included", stats['processed_files'])
                col2.metric("Files Scanned", stats['total_files'])
                col3.metric("Skipped (Size)", stats['skipped_too_large'])
                col4.metric("Output Size", f"{stats['total_size_bytes']/1024:.1f} KB")
                
                # Size warning
                if stats['total_size_bytes'] > max_total_size * 1024:
                    st.warning(f"⚠️ Output size ({stats['total_size_bytes']/1024:.1f} KB) exceeds your limit ({max_total_size} KB). Consider reducing file size limits or excluding more file types.")
                
                # Generate output
                output = format_output(
                    files,
                    stats,
                    repo_name=repo_name,
                    include_tree=include_tree,
                    include_stats=include_stats,
                    separator_style=separator_style,
                )
                
                # Preview
                st.header("📋 Preview")
                
                with st.expander("Directory Structure", expanded=True):
                    st.code(build_tree_structure(files), language=None)
                
                with st.expander("Files Included", expanded=False):
                    for f in files:
                        size_str = f"{f.size_bytes/1024:.1f} KB" if f.size_bytes > 1024 else f"{f.size_bytes} B"
                        st.text(f"{f.path} ({size_str})")
                
                # Output and download
                st.header("📥 Output")
                
                # Copy button and download
                col1, col2 = st.columns([3, 1])
                
                with col2:
                    st.download_button(
                        label="⬇️ Download as TXT",
                        data=output,
                        file_name=f"{repo_name}_code.txt",
                        mime="text/plain",
                    )
                
                # Text area with output
                st.text_area(
                    "Generated output (copy this to NotebookLM or your LLM)",
                    value=output,
                    height=400,
                    help="Select all (Ctrl/Cmd+A) and copy (Ctrl/Cmd+C)"
                )
                
                # Token estimate
                token_estimate = len(output) / 4  # Rough estimate
                st.caption(f"📊 Estimated tokens: ~{token_estimate:,.0f} (rough estimate, actual may vary)")
                
            except zipfile.BadZipFile:
                st.error("❌ Invalid ZIP file. Please upload a valid ZIP file.")
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                raise e


if __name__ == "__main__":
    main()
