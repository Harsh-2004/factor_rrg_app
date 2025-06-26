import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from scipy.interpolate import make_interp_spline
import time
from collections import Counter, defaultdict
import math

# Quadrant mapping
QUADRANT_MAP = {
    (1, 1): "Q1 (Leading)",
    (1, -1): "Q2 (Weakening)",
    (-1, -1): "Q3 (Lagging)",
    (-1, 1): "Q4 (Improving)"
}

# App configuration
st.set_page_config(layout="wide", page_title="Advanced RRG Analyzer")

# Sidebar controls
st.sidebar.header("RRG Configuration")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"], 
                                       help="Upload your sector data file with dates and price data")

# Default file path
default_file = "rrg_data updated13-25.xlsx"

# Load data
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_excel(file_path, parse_dates=["DATES"], index_col="DATES")
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    try:
        df = load_data(default_file)
    except:
        df = None

if df is None:
    st.error("Please upload a valid Excel file with the required format")
    st.stop()

# Available sectors (excluding benchmarks)
all_columns = df.columns.tolist()
benchmark_options = [col for col in all_columns]
sector_options = [col for col in all_columns]

# Benchmark selection
benchmark = st.sidebar.selectbox(
    "Select Benchmark Index",
    options=benchmark_options,
    index=benchmark_options.index("NIFTY Index") if "NIFTY Index" in benchmark_options else 0
)

# Sector selection
selected_sectors = st.sidebar.multiselect(
    "Select Sectors to Analyze",
    options=sector_options,
    default=["Auto"] if "Auto" in sector_options else sector_options[:1]
)

if not selected_sectors:
    st.warning("Please select at least one sector")
    st.stop()

# Parameters
st.sidebar.subheader("Analysis Parameters")
col1, col2 = st.sidebar.columns(2)

with col1:
    rolling_window = st.slider("Rolling Window", 10, 200, 36, 
                             help="Window size for rolling Z-score calculation")
    slice_window = st.slider("Slice Window", 1, 20, 5, 
                           help="Interval between points on the RRG path")
    tail_length = st.slider("Tail Length", 6, 50, 10, 
                           help="Number of historical points to show as tail")

with col2:
    short_period = st.slider("Momentum Period", 1, 100, 5, 
                            help="Period for momentum calculation")
    curve_smoothness = st.slider("Curve Smoothness", 0, 100, 30, 
                               help="Smoothness of the RRG path curves")
    past_days = st.slider("Animation Window (Days)", 30, 500, 120, 
                         help="Number of past days to animate through")

# Animation control
animate = st.sidebar.checkbox("Show Animation", value=False)
if animate:
    animation_speed = st.slider("Animation Speed (ms)", 100, 2000, 500, 
                              help="Speed of the animation in milliseconds")

def calculate_rrg(df, sectors, benchmark, rolling_window, short_period, slice_window):
    """Calculate RRG components with improved normalization"""
    # Calculate RS Ratio
    rs_ratios = df[sectors].div(df[benchmark], axis=0)
    rs_ratios = rs_ratios.iloc[::slice_window]
    
    # Smooth RS ratios
    rs_ratios_smoothed = rs_ratios.rolling(window=max(5, short_period)).mean()
    
    # Calculate momentum
    rs_momentum = rs_ratios_smoothed.diff(short_period) / rs_ratios_smoothed.shift(short_period) * 100
    
    # Normalize using shorter rolling window
    normalization_window = min(rolling_window, 30)
    
    rs_ratios_normalized = rs_ratios_smoothed.rolling(window=normalization_window).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if len(x.dropna()) > 1 and x.std() > 0 else 0
    )
    
    rs_momentum_normalized = rs_momentum.rolling(window=normalization_window).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if len(x.dropna()) > 1 and x.std() > 0 else 0
    )
    
    return rs_ratios_normalized, rs_momentum_normalized

rs_ratios_normalized, rs_momentum_normalized = calculate_rrg(
    df, selected_sectors, benchmark, rolling_window, short_period, slice_window
)

# Advanced analytics functions
def get_quadrant_number(x, y):
    """Get quadrant number (1-4) based on position"""
    if x >= 0 and y >= 0:
        return 1  # Leading
    elif x < 0 and y >= 0:
        return 2  # Weakening
    elif x < 0 and y < 0:
        return 3  # Lagging
    else:
        return 4  # Improving

def analyze_directional_patterns(rs_ratios_norm, rs_momentum_norm, asset_type="Sector"):
    """Analyze directional patterns for a set of assets"""
    
    results = {}
    
    # Expected patterns
    expected_counterclockwise = [(1,4), (4,3), (3,2), (2,1)]  # Sectors expected
    expected_clockwise = [(1,2), (2,3), (3,4), (4,1)]         # Traditional RRG
    
    for asset in rs_ratios_norm.columns:
        rs_data = rs_ratios_norm[asset]
        mom_data = rs_momentum_norm[asset]
        
        # Get valid data points
        valid_mask = rs_data.notna() & mom_data.notna()
        if valid_mask.sum() < 30:  # Need sufficient data
            continue
            
        rs_values = rs_data[valid_mask].values
        mom_values = mom_data[valid_mask].values
        dates = rs_data[valid_mask].index
        
        # Analyze this asset's pattern
        asset_analysis = analyze_single_asset_pattern(rs_values, mom_values, dates, asset)
        
        # Pattern reliability scoring
        asset_analysis['pattern_scores'] = score_pattern_reliability(
            asset_analysis['transitions'], expected_counterclockwise, expected_clockwise
        )
        
        results[asset] = asset_analysis
    
    # Aggregate results
    aggregated = aggregate_pattern_analysis(results, asset_type)
    
    return {
        'individual_assets': results,
        'aggregated': aggregated
    }

def analyze_single_asset_pattern(rs_values, mom_values, dates, asset_name):
    """Detailed analysis of single asset's RRG pattern"""
    
    # Classify quadrants
    quadrants = []
    for i in range(len(rs_values)):
        quadrants.append(get_quadrant_number(rs_values[i], mom_values[i]))
    
    # Analyze transitions
    transitions = []
    transition_dates = []
    for i in range(1, len(quadrants)):
        if quadrants[i] != quadrants[i-1]:
            transitions.append((quadrants[i-1], quadrants[i]))
            transition_dates.append(dates[i])
    
    # Analyze sequences (consecutive transitions)
    sequences = analyze_transition_sequences(transitions)
    
    # Time analysis
    time_analysis = analyze_transition_timing(transitions, transition_dates)
    
    # Directional consistency
    directional_analysis = analyze_directional_consistency(transitions)
    
    return {
        'asset_name': asset_name,
        'total_points': len(quadrants),
        'quadrant_distribution': dict(Counter(quadrants)),
        'transitions': transitions,
        'transition_count': len(transitions),
        'sequences': sequences,
        'time_analysis': time_analysis,
        'directional_analysis': directional_analysis
    }

def analyze_transition_sequences(transitions):
    """Analyze sequences of transitions to identify patterns"""
    
    sequences = {
        'two_step': [],
        'three_step': [],
        'four_step': [],
        'longer': []
    }
    
    # Look for consecutive sequences
    for i in range(len(transitions)):
        # Two-step sequences
        if i < len(transitions) - 1:
            two_step = (transitions[i], transitions[i+1])
            sequences['two_step'].append(two_step)
        
        # Three-step sequences  
        if i < len(transitions) - 2:
            three_step = (transitions[i], transitions[i+1], transitions[i+2])
            sequences['three_step'].append(three_step)
            
        # Four-step sequences (full cycle)
        if i < len(transitions) - 3:
            four_step = (transitions[i], transitions[i+1], transitions[i+2], transitions[i+3])
            sequences['four_step'].append(four_step)
    
    return sequences

def analyze_transition_timing(transitions, dates):
    """Analyze timing patterns of transitions"""
    
    if len(dates) < 2:
        return {}
    
    # Calculate time between transitions
    time_diffs = []
    for i in range(1, len(dates)):
        diff = (dates[i] - dates[i-1]).days
        time_diffs.append(diff)
    
    return {
        'avg_time_between_transitions': np.mean(time_diffs) if time_diffs else 0,
        'median_time_between_transitions': np.median(time_diffs) if time_diffs else 0,
        'std_time_between_transitions': np.std(time_diffs) if time_diffs else 0,
        'min_time_between_transitions': np.min(time_diffs) if time_diffs else 0,
        'max_time_between_transitions': np.max(time_diffs) if time_diffs else 0
    }

def analyze_directional_consistency(transitions):
    """Analyze directional consistency of transitions"""
    
    if not transitions:
        return {}
    
    # Define transition types
    counterclockwise_transitions = [(1,4), (4,3), (3,2), (2,1)]
    clockwise_transitions = [(1,2), (2,3), (3,4), (4,1)]
    
    # Count each type
    ccw_count = sum(1 for t in transitions if t in counterclockwise_transitions)
    cw_count = sum(1 for t in transitions if t in clockwise_transitions)
    
    # Adjacent quadrant transitions (any direction)
    adjacent_transitions = counterclockwise_transitions + clockwise_transitions
    adjacent_count = sum(1 for t in transitions if t in adjacent_transitions)
    
    # Diagonal transitions 
    diagonal_transitions = [(1,3), (3,1), (2,4), (4,2)]
    diagonal_count = sum(1 for t in transitions if t in diagonal_transitions)
    
    total_transitions = len(transitions)
    
    return {
        'counterclockwise_count': ccw_count,
        'clockwise_count': cw_count,
        'adjacent_count': adjacent_count,
        'diagonal_count': diagonal_count,
        'total_transitions': total_transitions,
        'ccw_percentage': ccw_count / total_transitions * 100 if total_transitions > 0 else 0,
        'cw_percentage': cw_count / total_transitions * 100 if total_transitions > 0 else 0,
        'adjacent_percentage': adjacent_count / total_transitions * 100 if total_transitions > 0 else 0,
        'diagonal_percentage': diagonal_count / total_transitions * 100 if total_transitions > 0 else 0,
        'dominant_direction': 'counterclockwise' if ccw_count > cw_count else 'clockwise' if cw_count > ccw_count else 'mixed'
    }

def score_pattern_reliability(transitions, expected_ccw, expected_cw):
    """Score how well transitions match expected patterns"""
    
    if not transitions:
        return {}
    
    # Count matches with expected patterns
    ccw_matches = sum(1 for t in transitions if t in expected_ccw)
    cw_matches = sum(1 for t in transitions if t in expected_cw)
    total_transitions = len(transitions)
    
    # Calculate reliability scores
    ccw_reliability = ccw_matches / total_transitions
    cw_reliability = cw_matches / total_transitions
    
    # Pattern consistency (what percentage follows ANY consistent pattern)
    pattern_consistency = max(ccw_reliability, cw_reliability)
    
    # Directional clarity (how clearly one direction dominates)
    directional_clarity = abs(ccw_reliability - cw_reliability)
    
    return {
        'counterclockwise_reliability': ccw_reliability,
        'clockwise_reliability': cw_reliability,
        'pattern_consistency': pattern_consistency,
        'directional_clarity': directional_clarity,
        'preferred_direction': 'counterclockwise' if ccw_reliability > cw_reliability else 'clockwise',
        'confidence_in_direction': max(ccw_reliability, cw_reliability)
    }

def aggregate_pattern_analysis(individual_results, asset_type):
    """Aggregate individual asset results"""
    
    if not individual_results:
        return {}
    
    # Collect all directional analysis data
    all_ccw_percentages = []
    all_cw_percentages = []
    all_pattern_consistencies = []
    all_directional_clarities = []
    
    dominant_directions = []
    
    for asset, results in individual_results.items():
        dir_analysis = results['directional_analysis']
        pattern_scores = results['pattern_scores']
        
        all_ccw_percentages.append(dir_analysis['ccw_percentage'])
        all_cw_percentages.append(dir_analysis['cw_percentage'])
        all_pattern_consistencies.append(pattern_scores['pattern_consistency'])
        all_directional_clarities.append(pattern_scores['directional_clarity'])
        
        dominant_directions.append(dir_analysis['dominant_direction'])
    
    # Calculate aggregated statistics
    aggregated = {
        'asset_type': asset_type,
        'total_assets_analyzed': len(individual_results),
        'average_ccw_percentage': np.mean(all_ccw_percentages),
        'average_cw_percentage': np.mean(all_cw_percentages),
        'average_pattern_consistency': np.mean(all_pattern_consistencies),
        'average_directional_clarity': np.mean(all_directional_clarities),
        'dominant_direction_distribution': dict(Counter(dominant_directions)),
        'assets_with_strong_ccw': sum(1 for pct in all_ccw_percentages if pct > 60),
        'assets_with_strong_cw': sum(1 for pct in all_cw_percentages if pct > 60),
        'assets_with_high_consistency': sum(1 for cons in all_pattern_consistencies if cons > 0.7)
    }
    
    # Determine overall pattern reliability
    if aggregated['average_ccw_percentage'] > 60:
        aggregated['overall_pattern'] = 'Strongly Counterclockwise (1→4→3→2)'
        aggregated['pattern_reliability'] = 'HIGH'
    elif aggregated['average_cw_percentage'] > 60:
        aggregated['overall_pattern'] = 'Strongly Clockwise (1→2→3→4)'
        aggregated['pattern_reliability'] = 'HIGH'
    elif aggregated['average_pattern_consistency'] > 0.5:
        aggregated['overall_pattern'] = 'Mixed but Consistent'
        aggregated['pattern_reliability'] = 'MEDIUM'
    else:
        aggregated['overall_pattern'] = 'Inconsistent/Random'
        aggregated['pattern_reliability'] = 'LOW'
    
    return aggregated

# Create the base RRG plot with quadrants (static elements)
def create_base_plot():
    fig = go.Figure()
    
    # Add quadrant lines and background colors
    fig.add_shape(type="line", x0=0, y0=-3, x1=0, y1=3, line=dict(color="black", width=1))
    fig.add_shape(type="line", x0=-3, y0=0, x1=3, y1=0, line=dict(color="black", width=1))
    
    # Add colored quadrant backgrounds
    fig.add_shape(type="rect", x0=0, y0=0, x1=3, y1=3, fillcolor="lightgreen", opacity=0.3, layer="below")
    fig.add_shape(type="rect", x0=0, y0=0, x1=3, y1=-3, fillcolor="yellow", opacity=0.3, layer="below")
    fig.add_shape(type="rect", x0=0, y0=0, x1=-3, y1=-3, fillcolor="lightcoral", opacity=0.3, layer="below")
    fig.add_shape(type="rect", x0=0, y0=0, x1=-3, y1=3, fillcolor="lightblue", opacity=0.3, layer="below")
    
    # Add quadrant labels
    fig.add_annotation(x=1.5, y=1.5, text="Leading<br>(Strong & Improving)", 
                      showarrow=False, font=dict(size=10, color="darkgreen"))
    fig.add_annotation(x=1.5, y=-1.5, text="Weakening<br>(Strong & Deteriorating)", 
                      showarrow=False, font=dict(size=10, color="orange"))
    fig.add_annotation(x=-1.5, y=-1.5, text="Lagging<br>(Weak & Deteriorating)", 
                      showarrow=False, font=dict(size=10, color="darkred"))
    fig.add_annotation(x=-1.5, y=1.5, text="Improving<br>(Weak & Improving)", 
                      showarrow=False, font=dict(size=10, color="blue"))
    
    # Set layout
    fig.update_layout(
        xaxis_title="Relative Strength Ratio (Z-score)",
        yaxis_title="Relative Strength Momentum (Z-score)",
        showlegend=True,
        width=900,
        height=700,
        xaxis=dict(range=[-3, 3], zeroline=True),
        yaxis=dict(range=[-3, 3], zeroline=True),
        margin=dict(l=40, r=40, t=80, b=40)
    )
    
    return fig

# Get snake tail data for a specific end position
def get_snake_tail_data(rs_data, momentum_data, end_idx, tail_length):
    """Get tail data for snake-like movement"""
    start_idx = max(0, end_idx - tail_length + 1)
    
    tail_rs = rs_data.iloc[start_idx:end_idx + 1]
    tail_momentum = momentum_data.iloc[start_idx:end_idx + 1]
    
    return tail_rs, tail_momentum

# Create traces for current frame with snake movement
def create_snake_traces(rs_data, momentum_data, sectors, current_idx, tail_length, curve_smoothness):
    """Create traces for snake-like animation"""
    traces = []
    colors = px.colors.qualitative.Plotly[:len(sectors)]
    
    for i, (sector, color) in enumerate(zip(sectors, colors)):
        # Get snake tail data
        tail_rs, tail_momentum = get_snake_tail_data(
            rs_data[sector], momentum_data[sector], current_idx, tail_length
        )
        
        if len(tail_rs) < 1:
            continue
            
        # Create smooth curve if enough points and smoothness is enabled
        if curve_smoothness > 0 and len(tail_rs) > 3:
            x = tail_rs.values
            y = tail_momentum.values
            
            try:
                # Create spline interpolation
                t = np.linspace(0, 1, len(x))
                t_new = np.linspace(0, 1, max(curve_smoothness, len(x)))
                
                spl_x = make_interp_spline(t, x, k=min(3, len(x)-1))
                x_smooth = spl_x(t_new)
                
                spl_y = make_interp_spline(t, y, k=min(3, len(y)-1))
                y_smooth = spl_y(t_new)
                
                # Add smooth tail trace
                traces.append(
                    go.Scatter(
                        x=x_smooth,
                        y=y_smooth,
                        mode="lines",
                        line=dict(color=color, width=3, dash="dash"),
                        showlegend=False,
                        name=f"{sector}_tail",
                        opacity=0.7
                    )
                )
            except:
                # Fallback to straight lines if spline fails
                traces.append(
                    go.Scatter(
                        x=tail_rs,
                        y=tail_momentum,
                        mode="lines",
                        line=dict(color=color, width=3, dash="dash"),
                        showlegend=False,
                        name=f"{sector}_tail",
                        opacity=0.7
                    )
                )
        else:
            # Plot straight tail
            traces.append(
                go.Scatter(
                    x=tail_rs,
                    y=tail_momentum,
                    mode="lines",
                    line=dict(color=color, width=3, dash="dash"),
                    showlegend=False,
                    name=f"{sector}_tail",
                    opacity=0.7
                )
            )
        
        # Current position (head of the snake)
        current_rs = rs_data[sector].iloc[current_idx]
        current_momentum = momentum_data[sector].iloc[current_idx]
        
        traces.append(
            go.Scatter(
                x=[current_rs],
                y=[current_momentum],
                mode="markers+text",
                name=sector,
                marker=dict(size=15, color=color, symbol="circle", 
                           line=dict(width=2, color="white")),
                text=[sector],
                textposition="top center",
                textfont=dict(size=12, color="black")
            )
        )
    
    return traces

# Create animation frames
def create_animation_frames(rs_data, momentum_data, sectors, tail_length, curve_smoothness):
    """Create all animation frames at once"""
    frames = []
    total_points = len(rs_data)
    
    # Start animation from tail_length to ensure we have enough data for the tail
    start_idx = max(tail_length - 1, 0)
    
    for current_idx in range(start_idx, total_points):
        frame_traces = create_snake_traces(
            rs_data, momentum_data, sectors, current_idx, tail_length, curve_smoothness
        )
        
        current_date = rs_data.index[current_idx].strftime('%Y-%m-%d')
        
        frame = go.Frame(
            data=frame_traces,
            name=str(current_idx),
            layout=go.Layout(
                title=f"Relative Rotation Graph vs. {benchmark} ({current_date})"
            )
        )
        frames.append(frame)
    
    return frames

# Create time series plot (combined)
def create_time_series_plots(rs_data, momentum_data, sectors, current_idx, tail_length):
    """Create combined time series plot for RS ratio and momentum over tail period"""
    
    # Get data for the tail period
    start_idx = max(0, current_idx - tail_length + 1)
    tail_dates = rs_data.index[start_idx:current_idx + 1]
    
    colors = px.colors.qualitative.Plotly[:len(sectors)]
    
    # Create single plot
    fig = go.Figure()
    
    # Add RS ratio and momentum traces for each sector
    for i, (sector, color) in enumerate(zip(sectors, colors)):
        tail_rs = rs_data[sector].iloc[start_idx:current_idx + 1]
        tail_momentum = momentum_data[sector].iloc[start_idx:current_idx + 1]
        
        # Add RS ratio trace (solid line)
        fig.add_trace(
            go.Scatter(
                x=tail_dates,
                y=tail_rs,
                mode='lines+markers',
                name=f'{sector} RS',
                line=dict(color=color, width=3),
                marker=dict(size=6, symbol='circle'),
                showlegend=True
            )
        )
        
        # Add momentum trace (dashed line)
        fig.add_trace(
            go.Scatter(
                x=tail_dates,
                y=tail_momentum,
                mode='lines+markers',
                name=f'{sector} Mom',
                line=dict(color=color, width=3, dash='dash'),
                marker=dict(size=6, symbol='diamond'),
                showlegend=True
            )
        )
    
    # Add horizontal reference line at zero
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.7, line_width=1)
    
    # Add positive/negative zones for context
    fig.add_hrect(y0=0, y1=3, fillcolor="lightgreen", opacity=0.1, layer="below")
    fig.add_hrect(y0=-3, y1=0, fillcolor="lightcoral", opacity=0.1, layer="below")
    
    # Update layout
    fig.update_layout(
        height=450,
        title=f"RS Ratio & Momentum Time Series - Tail Length: {tail_length} periods",
        xaxis_title="Date",
        yaxis_title="Z-score",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        yaxis=dict(range=[-3.5, 3.5]),
        hovermode='x unified'
    )
    
    return fig

# Main app
st.title("Advanced Relative Rotation Graph (RRG) Analysis")
st.markdown(f"**Benchmark:** {benchmark}")

# Create plot containers
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Relative Rotation Graph")
    rrg_placeholder = st.empty()

with col2:
    st.subheader("Time Series Analysis")
    ts_placeholder = st.empty()

# Run directional pattern analytics
st.subheader("Directional Pattern Analytics")
analytics_placeholder = st.empty()

# Perform analytics on the normalized data
analytics_results = analyze_directional_patterns(
    rs_ratios_normalized, 
    rs_momentum_normalized, 
    "Sectors"
)

# Display analytics results
if analytics_results['individual_assets']:
    # Display aggregated results
    agg = analytics_results['aggregated']
    st.markdown(f"### Aggregated Analysis ({agg['asset_type']})")
    st.markdown(f"**Total Assets Analyzed:** {agg['total_assets_analyzed']}")
    st.markdown(f"**Overall Pattern:** {agg['overall_pattern']} ({agg['pattern_reliability']} reliability)")
    
    # Create metrics columns
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Counterclockwise %", f"{agg['average_ccw_percentage']:.1f}%")
    col2.metric("Avg Clockwise %", f"{agg['average_cw_percentage']:.1f}%")
    col3.metric("Pattern Consistency", f"{agg['average_pattern_consistency']:.2f}")
    
    col1.metric("Assets with Strong CCW", agg['assets_with_strong_ccw'])
    col2.metric("Assets with Strong CW", agg['assets_with_strong_cw'])
    col3.metric("High Consistency Assets", agg['assets_with_high_consistency'])
    
    # Display dominant direction distribution
    st.markdown("**Dominant Direction Distribution:**")
    st.write(agg['dominant_direction_distribution'])
    
    # Display individual asset analytics
    st.markdown("### Individual Sector Analytics")
    for sector, analysis in analytics_results['individual_assets'].items():
        with st.expander(f"Analytics for {sector}"):
            st.markdown(f"**Total Points:** {analysis['total_points']}")
            st.markdown(f"**Total Transitions:** {analysis['transition_count']}")
            
            # Quadrant distribution
            st.markdown("**Quadrant Distribution:**")
            quad_data = pd.DataFrame({
                'Quadrant': [f"Q{quad} ({['Leading','Weakening','Lagging','Improving'][quad-1]})" 
                             for quad in analysis['quadrant_distribution'].keys()],
                'Count': analysis['quadrant_distribution'].values(),
                'Percentage': [count/analysis['total_points']*100 
                             for count in analysis['quadrant_distribution'].values()]
            })
            st.dataframe(quad_data)
            
            # Directional analysis
            dir_analysis = analysis['directional_analysis']
            st.markdown("**Directional Analysis:**")
            st.markdown(f"- Counterclockwise: {dir_analysis['ccw_percentage']:.1f}%")
            st.markdown(f"- Clockwise: {dir_analysis['cw_percentage']:.1f}%")
            st.markdown(f"- Adjacent: {dir_analysis['adjacent_percentage']:.1f}%")
            st.markdown(f"- Diagonal: {dir_analysis['diagonal_percentage']:.1f}%")
            st.markdown(f"- Dominant Direction: {dir_analysis['dominant_direction']}")
            
            # Pattern scores
            pattern = analysis['pattern_scores']
            st.markdown("**Pattern Reliability Scores:**")
            st.markdown(f"- Counterclockwise Reliability: {pattern['counterclockwise_reliability']:.2f}")
            st.markdown(f"- Clockwise Reliability: {pattern['clockwise_reliability']:.2f}")
            st.markdown(f"- Pattern Consistency: {pattern['pattern_consistency']:.2f}")
            st.markdown(f"- Directional Clarity: {pattern['directional_clarity']:.2f}")
            st.markdown(f"- Preferred Direction: {pattern['preferred_direction']}")
            st.markdown(f"- Confidence in Direction: {pattern['confidence_in_direction']:.2f}")
            
            # Time analysis
            time_analysis = analysis['time_analysis']
            st.markdown("**Transition Timing:**")
            st.markdown(f"- Avg Time Between Transitions: {time_analysis['avg_time_between_transitions']:.1f} days")
            st.markdown(f"- Median Time: {time_analysis['median_time_between_transitions']:.1f} days")
            st.markdown(f"- Min Time: {time_analysis['min_time_between_transitions']} days")
            st.markdown(f"- Max Time: {time_analysis['max_time_between_transitions']} days")
else:
    st.warning("Insufficient data for directional pattern analytics")

# Animation and display logic remains the same as before
# ... [The rest of the animation and display logic remains unchanged] ...
# Animation and display logic remains the same as before
# ... [The rest of the animation and display logic remains unchanged from your original code] ...
if animate:
    st.sidebar.info("Animation shows snake-like movement of sectors through time")
    
    # Create base figure
    animated_fig = create_base_plot()
    
    # Create animation frames
    with st.spinner("Preparing animation frames..."):
        frames = create_animation_frames(
            rs_ratios_normalized,
            rs_momentum_normalized,
            selected_sectors,
            tail_length,
            curve_smoothness
        )
    
    # Add frames to figure
    animated_fig.frames = frames
    
    # Add initial frame data and create initial time series
    if frames:
        for trace in frames[0].data:
            animated_fig.add_trace(trace)
        
        # Get initial index for time series
        initial_idx = max(tail_length - 1, 0)
        initial_ts_fig = create_time_series_plots(
            rs_ratios_normalized, rs_momentum_normalized, 
            selected_sectors, initial_idx, tail_length
        )
        
        animated_fig.update_layout(
            title=frames[0].layout.title.text,
            updatemenus=[{
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [None, {
                            "frame": {"duration": animation_speed, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 50}
                        }]
                    },
                    {
                        "label": "⏸ Pause",
                        "method": "animate",
                        "args": [[None], {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }]
                    }
                ]
            }],
            sliders=[{
                "steps": [
                    {
                        "args": [[frame.name], {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }],
                        "label": rs_ratios_normalized.index[int(frame.name)].strftime('%Y-%m-%d'),
                        "method": "animate"
                    }
                    for frame in frames
                ],
                "active": 0,
                "currentvalue": {"prefix": "Date: "},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "xanchor": "left",
                "yanchor": "top"
            }]
        )
    
    # Display the plots
    rrg_placeholder.plotly_chart(animated_fig, use_container_width=True, key="animated_plot")
    ts_placeholder.plotly_chart(initial_ts_fig, use_container_width=True, key="initial_ts")
    
    # Add manual time series update for slider interaction
    st.sidebar.info("💡 **Tip**: Use the slider below the RRG to see time series for different periods")
    
    # Create a selectbox for manual time series viewing during animation
    if st.sidebar.checkbox("Manual Time Series Control", help="Control time series independently"):
        manual_date_idx = st.sidebar.slider(
            "Select Date for Time Series", 
            min_value=max(tail_length - 1, 0),
            max_value=len(rs_ratios_normalized) - 1,
            value=len(rs_ratios_normalized) - 1,
            format="%d"
        )
        
        manual_date = rs_ratios_normalized.index[manual_date_idx].strftime('%Y-%m-%d')
        st.sidebar.write(f"**Selected Date:** {manual_date}")
        
        # Update time series plot
        manual_ts_fig = create_time_series_plots(
            rs_ratios_normalized, rs_momentum_normalized,
            selected_sectors, manual_date_idx, tail_length
        )
        ts_placeholder.plotly_chart(manual_ts_fig, use_container_width=True, key="manual_ts")
    
    st.sidebar.success("Animation ready! Use the play button to start.")
    
else:
    # Show static plot when animation is off
    static_fig = create_base_plot()
    
    # Add static traces for the final state
    final_idx = len(rs_ratios_normalized) - 1
    final_traces = create_snake_traces(
        rs_ratios_normalized,
        rs_momentum_normalized,
        selected_sectors,
        final_idx,
        tail_length,
        curve_smoothness
    )
    
    for trace in final_traces:
        static_fig.add_trace(trace)
    
    static_date = rs_ratios_normalized.index[-1].strftime('%Y-%m-%d')
    static_fig.update_layout(title=f"Relative Rotation Graph vs. {benchmark} ({static_date})")
    
    # Create time series for the final period
    final_ts_fig = create_time_series_plots(
        rs_ratios_normalized, rs_momentum_normalized,
        selected_sectors, final_idx, tail_length
    )
    
    # Display both plots
    rrg_placeholder.plotly_chart(static_fig, use_container_width=True, key="static_plot")
    ts_placeholder.plotly_chart(final_ts_fig, use_container_width=True, key="static_ts")
    
    # Add interactive date selection for static mode
    st.sidebar.subheader("Time Series Control")
    selected_date_idx = st.sidebar.slider(
        "Select Date for Analysis", 
        min_value=max(tail_length - 1, 0),
        max_value=len(rs_ratios_normalized) - 1,
        value=len(rs_ratios_normalized) - 1,
        format="%d"
    )
    
    selected_date = rs_ratios_normalized.index[selected_date_idx].strftime('%Y-%m-%d')
    st.sidebar.write(f"**Selected Date:** {selected_date}")
    
    # Update both plots based on selected date
    if selected_date_idx != final_idx:
        # Update RRG plot
        updated_fig = create_base_plot()
        updated_traces = create_snake_traces(
            rs_ratios_normalized,
            rs_momentum_normalized,
            selected_sectors,
            selected_date_idx,
            tail_length,
            curve_smoothness
        )
        
        for trace in updated_traces:
            updated_fig.add_trace(trace)
        
        updated_fig.update_layout(title=f"Relative Rotation Graph vs. {benchmark} ({selected_date})")
        rrg_placeholder.plotly_chart(updated_fig, use_container_width=True, key="updated_plot")
        
        # Update time series plot
        updated_ts_fig = create_time_series_plots(
            rs_ratios_normalized, rs_momentum_normalized,
            selected_sectors, selected_date_idx, tail_length
        )
        ts_placeholder.plotly_chart(updated_ts_fig, use_container_width=True, key="updated_ts")

# Data download
st.sidebar.markdown("---")
st.sidebar.subheader("Export Data")
if st.sidebar.button("Download RRG Data as CSV"):
    combined_df = pd.DataFrame({
        'Date': rs_ratios_normalized.index,
        **{f"{sector}_RS": rs_ratios_normalized[sector] for sector in selected_sectors},
        **{f"{sector}_Momentum": rs_momentum_normalized[sector] for sector in selected_sectors}
    })
    csv = combined_df.to_csv(index=False)
    st.sidebar.download_button(
        label="Download CSV",
        data=csv,
        file_name="rrg_analysis.csv",
        mime="text/csv"
    )

# Display raw data
if st.checkbox("Show Raw Data"):
    st.subheader("Normalized Relative Strength Ratios")
    st.dataframe(rs_ratios_normalized.tail(20))
    
    st.subheader("Normalized Momentum Values")
    st.dataframe(rs_momentum_normalized.tail(20))