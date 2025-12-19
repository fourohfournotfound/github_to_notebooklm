"""
RepoToText - Convert GitHub Repository Zips to LLM-Friendly Text
A Streamlit app for processing GitHub zip files into clean, formatted text
suitable for NotebookLM, Claude, ChatGPT, and other LLMs.
"""

import streamlit as st
import zipfile
import io
import os
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from collections import defaultdict
from urllib.parse import urlparse

# =============================================================================
# Configuration
# =============================================================================

def parse_github_url(url: str) -> Optional[str]:
    """
    Parse a GitHub URL and extract the subfolder path.
    
    Examples:
        https://github.com/user/repo/tree/main/src/module -> src/module
        https://github.com/user/repo/tree/master/lib -> lib
        https://github.com/user/repo -> None (root)
    """
    if not url or not url.strip():
        return None
    
    url = url.strip()
    
    # Match GitHub tree URLs
    # Pattern: github.com/user/repo/tree/branch/path/to/folder
    pattern = r'github\.com/[^/]+/[^/]+/tree/[^/]+/(.+?)/?$'
    match = re.search(pattern, url)
    
    if match:
        return match.group(1)
    
    return None


def get_folder_structure(zip_bytes: bytes) -> dict:
    """
    Extract the folder structure from a ZIP file.
    Returns a nested dict representing the directory tree.
    """
    folders = set()
    files_by_folder = defaultdict(list)
    
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for file_info in zf.filelist:
            filepath = file_info.filename
            
            # Remove top-level GitHub directory (repo-branch/)
            parts = Path(filepath).parts
            if len(parts) > 1:
                clean_path = str(Path(*parts[1:]))
                clean_parts = parts[1:]
            else:
                continue
            
            if file_info.is_dir():
                if clean_path and clean_path != '.':
                    folders.add(clean_path.rstrip('/'))
            else:
                # Add all parent folders
                for i in range(1, len(clean_parts)):
                    folder = str(Path(*clean_parts[:i]))
                    folders.add(folder)
                
                # Track which folder this file is in
                if len(clean_parts) > 1:
                    parent = str(Path(*clean_parts[:-1]))
                    files_by_folder[parent].append(clean_parts[-1])
                else:
                    files_by_folder[''].append(clean_parts[0])
    
    return {
        'folders': sorted(folders),
        'files_by_folder': dict(files_by_folder),
    }


def build_folder_tree(folders: list[str]) -> dict:
    """Build a nested tree structure from a flat list of folder paths."""
    tree = {}
    for folder in folders:
        parts = Path(folder).parts
        current = tree
        for part in parts:
            if part not in current:
                current[part] = {}
            current = current[part]
    return tree


def render_folder_selector(folders: list[str], files_by_folder: dict) -> tuple[set, set]:
    """
    Render an interactive folder selector in Streamlit.
    Returns (included_folders, excluded_folders).
    """
    if not folders:
        return set(), set()
    
    # Get top-level folders
    top_level = sorted(set(Path(f).parts[0] for f in folders if f))
    
    # Add root files indicator if there are root files
    root_files = files_by_folder.get('', [])
    
    included = set()
    excluded = set()
    
    st.markdown("**Select folders to include:**")
    
    # Root level files
    if root_files:
        st.markdown(f"📄 *{len(root_files)} files in root*")
    
    # Create columns for better layout
    for folder in top_level:
        # Count subfolders and files
        subfolders = [f for f in folders if f.startswith(folder + '/') or f == folder]
        file_count = sum(len(files_by_folder.get(sf, [])) for sf in subfolders)
        file_count += len(files_by_folder.get(folder, []))
        
        col1, col2 = st.columns([3, 1])
        with col1:
            is_included = st.checkbox(
                f"📁 {folder}/",
                value=True,
                key=f"folder_{folder}"
            )
        with col2:
            st.caption(f"{len(subfolders)} folders, {file_count} files")
        
        if is_included:
            included.add(folder)
        else:
            excluded.add(folder)
    
    return included, excluded

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
    include_paths: Optional[set] = None,
    exclude_paths: Optional[set] = None,
    root_path_filter: Optional[str] = None,
) -> tuple[list[ProcessedFile], dict]:
    """
    Process a zip file and extract relevant files.
    
    Args:
        zip_bytes: The ZIP file contents
        config: Filter configuration
        allowed_extensions: Set of allowed file extensions
        include_requirements: Whether to include requirements.txt files
        include_paths: If set, only include files under these paths
        exclude_paths: If set, exclude files under these paths
        root_path_filter: If set, only include files under this root path
    
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
        'skipped_path_filter': 0,
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
            
            # Apply root path filter (from GitHub URL or manual input)
            if root_path_filter:
                if not clean_path.startswith(root_path_filter):
                    stats['skipped_path_filter'] += 1
                    continue
                # Adjust the path to be relative to the filter
                clean_path = clean_path[len(root_path_filter):].lstrip('/')
                if not clean_path:
                    continue
            
            filename = Path(clean_path).name
            
            # Check include/exclude paths
            if include_paths or exclude_paths:
                path_parts = Path(clean_path).parts
                top_level = path_parts[0] if path_parts else ''
                
                # Check exclusions first
                if exclude_paths and top_level in exclude_paths:
                    stats['skipped_path_filter'] += 1
                    continue
                
                # If we have explicit includes, check them (but allow root files)
                if include_paths and top_level and top_level not in include_paths:
                    stats['skipped_path_filter'] += 1
                    continue
            
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
        # Read the zip once
        zip_bytes = uploaded_file.read()
        
        # Get repo name from filename
        repo_name = uploaded_file.name.replace('.zip', '').replace('-main', '').replace('-master', '')
        repo_name = st.text_input("Repository name", value=repo_name)
        
        # Path filtering section
        st.header("📂 Path Filtering")
        
        filter_tab1, filter_tab2, filter_tab3 = st.tabs([
            "🌳 Browse Folders", 
            "🔗 GitHub URL", 
            "✏️ Manual Path"
        ])
        
        root_path_filter = None
        include_paths = None
        exclude_paths = None
        
        # Get folder structure for the explorer
        try:
            folder_info = get_folder_structure(zip_bytes)
            folders = folder_info['folders']
            files_by_folder = folder_info['files_by_folder']
        except Exception as e:
            st.error(f"Error reading ZIP structure: {e}")
            folders = []
            files_by_folder = {}
        
        with filter_tab1:
            st.markdown("Select which top-level folders to include:")
            
            if folders:
                # Get unique top-level folders
                top_level_folders = sorted(set(
                    Path(f).parts[0] for f in folders if f
                ))
                
                if top_level_folders:
                    # Show root files count
                    root_files = files_by_folder.get('', [])
                    if root_files:
                        st.caption(f"📄 {len(root_files)} files in repository root")
                    
                    # Create checkboxes for each top-level folder
                    included_folders = set()
                    
                    cols = st.columns(2)
                    for i, folder in enumerate(top_level_folders):
                        # Count contents
                        subfolders = [f for f in folders if f == folder or f.startswith(folder + '/')]
                        subfolder_count = len(subfolders)
                        
                        # Count files in this folder tree
                        file_count = len(files_by_folder.get(folder, []))
                        for sf in subfolders:
                            file_count += len(files_by_folder.get(sf, []))
                        
                        with cols[i % 2]:
                            is_included = st.checkbox(
                                f"📁 **{folder}/**",
                                value=True,
                                key=f"folder_{folder}",
                                help=f"{subfolder_count} subfolders, ~{file_count} files"
                            )
                            if is_included:
                                included_folders.add(folder)
                    
                    # Set the filter
                    if len(included_folders) < len(top_level_folders):
                        include_paths = included_folders
                        excluded_folders = set(top_level_folders) - included_folders
                        if excluded_folders:
                            st.info(f"Excluding: {', '.join(sorted(excluded_folders))}")
                else:
                    st.info("All files are in the repository root.")
            else:
                st.info("Upload a ZIP to see folder structure.")
        
        with filter_tab2:
            st.markdown("""
            Paste a GitHub URL pointing to a specific folder, and only files 
            under that path will be included.
            
            **Example:** `https://github.com/user/repo/tree/main/src/mymodule`
            """)
            
            github_url = st.text_input(
                "GitHub folder URL",
                placeholder="https://github.com/user/repo/tree/main/src/mymodule",
                key="github_url"
            )
            
            if github_url:
                parsed_path = parse_github_url(github_url)
                if parsed_path:
                    st.success(f"✅ Will filter to: `{parsed_path}/`")
                    root_path_filter = parsed_path
                else:
                    st.warning("Could not parse path from URL. Make sure it points to a folder (contains `/tree/branch/path`).")
        
        with filter_tab3:
            st.markdown("Manually specify a path prefix to filter to:")
            
            manual_path = st.text_input(
                "Path filter",
                placeholder="src/mymodule",
                key="manual_path",
                help="Only include files under this path"
            )
            
            if manual_path:
                manual_path = manual_path.strip().strip('/')
                if manual_path:
                    st.info(f"Will filter to: `{manual_path}/`")
                    root_path_filter = manual_path
        
        # Process button
        st.header("🚀 Process")
        
        if st.button("Process Repository", type="primary"):
            with st.spinner("Processing repository..."):
                try:
                    files, stats = process_zip_file(
                        zip_bytes,
                        config,
                        allowed_extensions,
                        include_requirements,
                        include_paths=include_paths,
                        exclude_paths=exclude_paths,
                        root_path_filter=root_path_filter,
                    )
                    
                    # Store in session state
                    st.session_state['processed_files'] = files
                    st.session_state['stats'] = stats
                    st.session_state['repo_name'] = repo_name
                    
                except zipfile.BadZipFile:
                    st.error("❌ Invalid ZIP file. Please upload a valid ZIP file.")
                    return
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
                    raise e
        
        # Show results if we have them
        if 'processed_files' in st.session_state:
            files = st.session_state['processed_files']
            stats = st.session_state['stats']
            repo_name = st.session_state.get('repo_name', 'Repository')
            
            if not files:
                st.warning("No matching files found. Try adjusting the filters or path settings.")
                return
            
            # Show results
            st.success(f"✅ Processed {stats['processed_files']} files!")
            
            # Statistics in columns
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Files Included", stats['processed_files'])
            col2.metric("Files Scanned", stats['total_files'])
            col3.metric("Skipped (Size)", stats['skipped_too_large'])
            col4.metric("Output Size", f"{stats['total_size_bytes']/1024:.1f} KB")
            
            # Additional stats row
            if stats.get('skipped_path_filter', 0) > 0:
                st.caption(f"📁 {stats['skipped_path_filter']} files filtered out by path selection")
            
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


if __name__ == "__main__":
    main()
