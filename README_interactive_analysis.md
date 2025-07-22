# Interactive Observed Loss Distribution Analysis

This interactive version of the observed loss distribution analysis provides enhanced visualization capabilities using Plotly, allowing for dynamic exploration of loss data with hover information, zoom, pan, and selection features.

## Features

### Interactive Visualizations
- **Bar Charts**: Color-coded yearly losses with hover details
- **Density Plots**: Interactive histograms with reference lines
- **Box Plots**: Distribution analysis with outlier identification
- **Scatter Plots**: Trend analysis with trend lines
- **Dashboard**: Multi-panel summary view
- **Comparison Plots**: Observed vs simulated losses overlay

### Interactive Capabilities
- **Hover Information**: Detailed tooltips on data points
- **Zoom & Pan**: Navigate through large datasets
- **Selection Tools**: Highlight specific data ranges
- **Export Options**: Save as PNG images or HTML files
- **Responsive Design**: Works on different screen sizes

## Installation

### Prerequisites
- Python 3.7+
- Required packages (see `requirements_interactive.txt`)

### Setup
```bash
# Install dependencies
pip install -r requirements_interactive.txt

# Or install individually
pip install plotly pandas numpy kaleido
```

## Usage

### Command Line
```bash
# Run the interactive analysis
python scripts/plot_observed_loss_distribution_interactive.py
```

### Jupyter Notebook
```python
# Import and use in notebook
from scripts.plot_observed_loss_distribution_interactive import *

# Run analysis
main()
```

### Custom Analysis
```python
# Load data
df = load_observed_losses(file_path)
stats = calculate_loss_statistics(df)

# Create custom interactive plots
plot_paths = create_interactive_loss_analysis(df, stats, run_path, port_name)
```

## Output Files

The script generates several interactive HTML files:

1. **`observed_losses_by_year_interactive.html`**
   - Bar chart of losses by year
   - Color-coded by loss severity
   - Reference lines for mean and P99

2. **`observed_loss_density_interactive.html`**
   - Histogram of loss distribution
   - Reference lines for key statistics
   - Interactive binning

3. **`observed_loss_boxplot_interactive.html`**
   - Box plot with outliers
   - Distribution summary
   - Interactive outlier identification

4. **`observed_loss_trend_interactive.html`**
   - Scatter plot with trend line
   - Color-coded by loss amount
   - Time series analysis

5. **`observed_loss_dashboard_interactive.html`**
   - Multi-panel dashboard
   - Summary statistics table
   - Percentile analysis

6. **`observed_vs_all_aep_methods_interactive.html`** (if AEP data available)
   - Comparison with simulated losses
   - Overlay histograms
   - Method comparison

## Interactive Features

### Navigation
- **Zoom**: Use mouse wheel or zoom tools
- **Pan**: Click and drag to move around
- **Reset**: Click home icon to reset view
- **Selection**: Use selection tools to highlight data

### Information
- **Hover**: Move mouse over data points for details
- **Click**: Select data points for detailed information
- **Legend**: Click legend items to show/hide traces

### Export
- **PNG**: Click camera icon to save as image
- **HTML**: Files are already in HTML format for sharing
- **Data**: Hover information shows exact values

## Configuration

The script uses the same configuration as the original analysis:

```python
# In config.py
config = {
    'RUN_PATH': 'run_g1',
    'reference_port': 'COLAN',
    'results_output_dir': 'results/cv_results/run_g1'
}
```

## Comparison with Static Version

| Feature | Static (Matplotlib) | Interactive (Plotly) |
|---------|-------------------|---------------------|
| File Format | PNG/PDF | HTML |
| Interactivity | None | Full |
| Hover Info | None | Detailed |
| Zoom/Pan | None | Yes |
| Export | Static images | Interactive HTML |
| File Size | Small | Larger |
| Sharing | Images only | Full interactive |

## Tips for Best Results

1. **Browser Compatibility**: Use modern browsers (Chrome, Firefox, Safari, Edge)
2. **File Sharing**: HTML files can be shared via email or file sharing
3. **Embedding**: HTML files can be embedded in web applications
4. **Performance**: Large datasets may take time to load initially
5. **Mobile**: Interactive plots work on mobile devices

## Troubleshooting

### Common Issues

**Plotly not installed:**
```bash
pip install plotly
```

**HTML files not opening:**
- Ensure you have a modern web browser
- Check file permissions
- Try opening with different browser

**Large file sizes:**
- Reduce number of data points
- Use sampling for large datasets
- Consider static plots for very large datasets

**Performance issues:**
- Close other browser tabs
- Use smaller datasets for testing
- Consider using static plots for presentations

## Examples

### Basic Usage
```python
# Simple interactive bar chart
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Bar(x=df['year'], y=df['observed_loss']))
fig.show()
```

### Advanced Usage
```python
# Custom interactive dashboard
from plotly.subplots import make_subplots

fig = make_subplots(rows=2, cols=2)
# Add traces...
fig.show()
```

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the example notebook
3. Check Plotly documentation
4. Verify your data format

## License

This interactive analysis tool follows the same license as the main project. 