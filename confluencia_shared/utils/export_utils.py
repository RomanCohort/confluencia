"""Multi-format Data Export Utilities.

Provides functions to export dataframes and plots in various formats
for sharing, publication, and archival.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Any


def export_dataframe(df, format: str, path: str, **kwargs) -> str:
    """Export a pandas DataFrame to various formats.

    Args:
        df: pandas DataFrame
        format: Export format (csv, excel, json, latex, parquet, hdf5, tsv, markdown)
        path: Output file path
        **kwargs: Format-specific options

    Returns:
        Path to exported file
    """
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    format = format.lower().strip('.')

    if format == "csv":
        df.to_csv(path, index=kwargs.get('index', False), encoding='utf-8')
    elif format in ("excel", "xlsx", "xls"):
        _export_excel(df, path, **kwargs)
    elif format == "json":
        orient = kwargs.get('orient', 'records')
        df.to_json(path, orient=orient, indent=2, force_ascii=False)
    elif format == "latex":
        _export_latex(df, path, **kwargs)
    elif format == "parquet":
        df.to_parquet(path, index=kwargs.get('index', False))
    elif format in ("hdf5", "h5"):
        df.to_hdf(path, key=kwargs.get('key', 'data'), mode='w')
    elif format == "tsv":
        df.to_csv(path, index=kwargs.get('index', False), sep='\t', encoding='utf-8')
    elif format in ("markdown", "md"):
        _export_markdown(df, path, **kwargs)
    elif format == "html":
        df.to_html(path, index=kwargs.get('index', False))
    elif format == "rst":
        _export_rst(df, path, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}. Supported: csv, excel, json, latex, parquet, hdf5, tsv, markdown, html, rst")

    return str(path)


def _export_excel(df, path, **kwargs):
    """Export with formatting."""
    try:
        import openpyxl

        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            df.to_excel(writer, index=kwargs.get('index', False), sheet_name='Data')

            # Auto-adjust column widths
            worksheet = writer.sheets['Data']
            for i, col in enumerate(df.columns):
                max_len = max(
                    df[col].astype(str).str.len().max(),
                    len(str(col))
                )
                worksheet.column_dimensions[openpyxl.utils.get_column_letter(i + 1)].width = min(max_len + 2, 50)

    except ImportError:
        # Fallback without formatting
        df.to_excel(path, index=kwargs.get('index', False))


def _export_latex(df, path, **kwargs):
    """Export as LaTeX table."""
    caption = kwargs.get('caption', 'Data Table')
    label = kwargs.get('label', 'tab:data')

    latex_str = df.to_latex(
        index=kwargs.get('index', False),
        escape=kwargs.get('escape', True),
        caption=caption,
        label=label,
    )

    with open(path, 'w', encoding='utf-8') as f:
        f.write(latex_str)


def _export_markdown(df, path, **kwargs):
    """Export as Markdown table."""
    lines = []

    # Header
    columns = df.columns.tolist()
    lines.append("| " + " | ".join(str(c) for c in columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")

    # Rows
    for _, row in df.iterrows():
        values = [str(v).replace('|', '\\|') for v in row]
        lines.append("| " + " | ".join(values) + " |")

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def _export_rst(df, path, **kwargs):
    """Export as reStructuredText table."""
    columns = [str(c) for c in df.columns]
    col_widths = [len(c) for c in columns]

    for col in columns:
        max_val_len = df[col].astype(str).str.len().max()
        col_widths[columns.index(col)] = max(col_widths[columns.index(col)], max_val_len)

    # Build table
    separator = '+' + '+'.join('-' * (w + 2) for w in col_widths) + '+'
    header_sep = '+' + '+'.join('=' * (w + 2) for w in col_widths) + '+'

    lines = [separator]
    header = '|' + '|'.join(f' {c:<{col_widths[i]}} ' for i, c in enumerate(columns)) + '|'
    lines.append(header)
    lines.append(header_sep)

    for _, row in df.iterrows():
        values = [str(v) for v in row]
        line = '|' + '|'.join(f' {v:<{col_widths[i]}} ' for i, v in enumerate(values)) + '|'
        lines.append(line)
        lines.append(separator)

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def export_plot(
    figure_or_path,
    format: str,
    output_path: str,
    dpi: int = 300,
) -> str:
    """Export a plot to various formats.

    Args:
        figure_or_path: matplotlib figure, Plotly figure, or file path
        format: Output format (png, svg, pdf, eps, html, json)
        output_path: Output file path
        dpi: Resolution for raster formats

    Returns:
        Path to exported file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    format = format.lower().strip('.')

    # Check if it's a file path
    if isinstance(figure_or_path, (str, Path)):
        return _convert_image(figure_or_path, format, output_path)

    # Check if it's a matplotlib figure
    try:
        import matplotlib.figure
        if isinstance(figure_or_path, matplotlib.figure.Figure):
            if format == "png":
                figure_or_path.savefig(output_path, dpi=dpi, bbox_inches='tight')
            elif format == "svg":
                figure_or_path.savefig(output_path, format='svg', bbox_inches='tight')
            elif format == "pdf":
                figure_or_path.savefig(output_path, format='pdf', bbox_inches='tight')
            elif format == "eps":
                figure_or_path.savefig(output_path, format='eps', bbox_inches='tight')
            else:
                figure_or_path.savefig(output_path, dpi=dpi, bbox_inches='tight')
            return str(output_path)
    except ImportError:
        pass

    # Check if it's a Plotly figure
    try:
        import plotly.graph_objects as go
        if isinstance(figure_or_path, (go.Figure,)):
            if format == "html":
                figure_or_path.write_html(str(output_path))
            elif format == "png":
                figure_or_path.write_image(str(output_path), format='png', scale=dpi/72)
            elif format == "svg":
                figure_or_path.write_image(str(output_path), format='svg')
            elif format == "pdf":
                figure_or_path.write_image(str(output_path), format='pdf')
            elif format == "json":
                figure_or_path.write_json(str(output_path))
            else:
                figure_or_path.write_html(str(output_path))
            return str(output_path)
    except ImportError:
        pass

    raise ValueError(f"Unsupported figure type or format: {format}")


def _convert_image(input_path, format, output_path):
    """Convert an image file to another format."""
    import shutil
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if format in ("png", "jpg", "jpeg", "bmp", "tiff"):
        try:
            from PIL import Image
            img = Image.open(input_path)
            img.save(output_path)
        except ImportError:
            shutil.copy(input_path, output_path)
    else:
        shutil.copy(input_path, output_path)

    return str(output_path)


def export_reproducible_bundle(
    experiment_data: Dict[str, Any],
    output_path: str,
    data_files: Optional[List[str]] = None,
    model_files: Optional[List[str]] = None,
) -> str:
    """Export a quick reproducible bundle.

    Args:
        experiment_data: Experiment metadata
        output_path: Output ZIP path
        data_files: Data files to include
        model_files: Model files to include

    Returns:
        Path to created bundle
    """
    from confluencia_studio.core.bundle_exporter import ReproducibleBundle

    bundler = ReproducibleBundle()
    return bundler.create_bundle(
        experiment_id=experiment_data.get("id", "unknown"),
        output_path=output_path,
        include_data=bool(data_files),
        include_models=bool(model_files),
    )


def get_supported_formats() -> Dict[str, List[str]]:
    """Get supported export formats for each type."""
    return {
        "dataframe": ["csv", "excel", "json", "latex", "parquet", "hdf5", "tsv", "markdown", "html", "rst"],
        "plot": ["png", "svg", "pdf", "eps", "html"],
    }
