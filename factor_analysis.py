import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz

# Set page config
st.set_page_config(
    page_title="Portfolio Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    .metric-card-red {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        color: #333;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0;
    }
    .metric-label {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 0.9rem;
    }
    div[data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #e6e6e6;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading function
@st.cache_data
def load_and_process_data():
    try:
        low_vol_30 = pd.read_csv('nifty100_low_volatility_30.csv')
        momentum_15 = pd.read_csv('nifty200_momentum_15.csv')
        all_momentum = pd.read_csv('nifty200_all_momentum_scores.csv')
        all_volatility = pd.read_csv('nifty100_all_volatility_scores.csv')
        
        existing_index_file = pd.ExcelFile('existing_index.xlsx')
        existing_low_vol_df = pd.read_excel(existing_index_file, sheet_name='low vol')
        existing_momentum_df = pd.read_excel(existing_index_file, sheet_name='mom')
        
        mktcap_df = pd.read_excel('factor_data.xlsx', sheet_name='maktcap')
        name_to_id_mapping = dict(zip(mktcap_df['Name'], mktcap_df['ID']))
        
        manual_mappings = {
            'The Indian Hotels Company Ltd.': 'IH IS Equity',
            'Indian Hotels Company Ltd.': 'IH IS Equity',
            'Coforge Ltd.': 'COFORGE IS Equity',
            'Coforge Limited': 'COFORGE IS Equity',
            'HDFC Bank Ltd.': 'HDFCB IS Equity',
            'Titan Company Ltd.': 'TTAN IS Equity',
        }
        name_to_id_mapping.update(manual_mappings)
        
        return (low_vol_30, momentum_15, all_momentum, all_volatility, 
                existing_low_vol_df, existing_momentum_df, name_to_id_mapping)
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

def clean_company_name(name):
    name = str(name)
    name = (name.replace(' Ltd.', '').replace(' Limited', '').replace(' Ltd', '').replace(' LTD', '')
            .replace('.', '').replace('&', 'AND').replace(',', '').replace("'", '').strip())
    if name.upper().startswith('THE '):
        name = name[4:]
    return name.upper()

def convert_names_to_ids(company_names, name_to_id_mapping):
    ids = []
    unmapped_names = []
    manual_mappings = {
        'The Indian Hotels Company Ltd.': 'IH IS Equity',
        'Indian Hotels Company Ltd.': 'IH IS Equity',
        'Coforge Ltd.': 'COFORGE IS Equity',
        'Coforge Limited': 'COFORGE IS Equity',
        'HDFC Bank Ltd.': 'HDFCB IS Equity',
        'Titan Company Ltd.': 'TTAN IS Equity',
    }
    
    for name in company_names:
        if name in ['Tri-Party Repo (TREPS)', 'Cash & Cash Equivalent', 'Net Current Asset']:
            ids.append('N/A')
            continue
            
        matched = False
        matched_id = None
        clean_name = clean_company_name(name)
        
        if name in manual_mappings:
            matched_id = manual_mappings[name]
            matched = True
        elif clean_name in manual_mappings:
            matched_id = manual_mappings[clean_name]
            matched = True
        elif name in name_to_id_mapping:
            matched_id = name_to_id_mapping[name]
            matched = True
        elif clean_name in name_to_id_mapping:
            matched_id = name_to_id_mapping[clean_name]
            matched = True
        else:
            best_match = None
            best_score = 0
            
            for mapped_name, mapped_id in name_to_id_mapping.items():
                clean_mapped = clean_company_name(mapped_name)
                score = fuzz.ratio(clean_name, clean_mapped)
                
                if score > best_score and score > 80:
                    best_score = score
                    best_match = (mapped_name, mapped_id)
            
            if best_match:
                matched_id = best_match[1]
                matched = True
        
        if matched and matched_id:
            ids.append(matched_id)
        else:
            ids.append('UNMAPPED')
            unmapped_names.append(name)
    
    return ids

def calculate_weighted_overlap_data(new_portfolio, existing_index_df):
    existing_index_df = existing_index_df[existing_index_df['ID'] != 'UNMAPPED']
    merged = pd.merge(new_portfolio, existing_index_df, 
                     left_on='Unnamed: 0', right_on='ID', how='inner')
    
    if not merged.empty:
        new_weights = merged['Weight']
        existing_weights = merged['Holding Percentage']
        merged['weight_overlap'] = np.minimum(new_weights, existing_weights)
        total_weight_overlap = merged['weight_overlap'].sum()
        
        overlap_count = len(merged)
        total_new = len(new_portfolio)
        
        merged['New_Weight'] = new_weights
        merged['Existing_Weight'] = existing_weights
        merged['Overlap_Weight'] = merged['weight_overlap']
        
        display_columns = ['Unnamed: 0', 'Company Name', 'New_Weight', 'Existing_Weight', 'Overlap_Weight']
        merged_display = merged[display_columns].sort_values('Overlap_Weight', ascending=False)
        
        return {
            'overlap_count': overlap_count,
            'total_new': total_new,
            'weight_overlap': total_weight_overlap,
            'merged_data': merged_display
        }
    return {'overlap_count': 0, 'total_new': len(new_portfolio), 'weight_overlap': 0, 'merged_data': pd.DataFrame()}

def calculate_quartile_data(index_df, factor_scores_df, factor_name):
    factor_scores = factor_scores_df.set_index('Unnamed: 0')[factor_scores_df.columns[1]]
    quartiles = pd.qcut(factor_scores, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    index_quartiles = index_df['ID'].map(quartiles).value_counts().reindex(['Q1', 'Q2', 'Q3', 'Q4'], fill_value=0)
    
    total = len(index_df)
    quartile_pct = (index_quartiles / total * 100).round(1)
    
    index_with_scores = pd.merge(index_df, factor_scores.rename('factor_score'), 
                                left_on='ID', right_index=True, how='left')
    
    # Add quartile column
    index_with_scores['quartile'] = index_with_scores['ID'].map(quartiles)
    
    # Sort for full list
    if factor_name == 'Low Volatility':
        all_stocks_sorted = index_with_scores.sort_values(by=['quartile', 'factor_score'], ascending=[True, True])
    else:  # Momentum
        all_stocks_sorted = index_with_scores.sort_values(by=['quartile', 'factor_score'], ascending=[True, False])
    
    all_stocks_with_quartile = all_stocks_sorted[['ID', 'Company Name', 'factor_score', 'quartile']]
    
    return {
        'quartile_pct': quartile_pct,
        'top_stocks': index_with_scores.sort_values('factor_score', ascending=(factor_name == 'Low Volatility')).head(5),
        'bottom_stocks': index_with_scores.sort_values('factor_score', ascending=(factor_name != 'Low Volatility')).head(5),
        'all_scores': index_with_scores['factor_score'].dropna(),
        'universe_scores': factor_scores,
        'all_stocks_with_quartile': all_stocks_with_quartile
    }

def calculate_missing_stocks_analysis(new_portfolio, existing_index_df, factor_scores_df, factor_name):
    factor_scores = factor_scores_df.set_index('Unnamed: 0')[factor_scores_df.columns[1]]
    quartiles = pd.qcut(factor_scores, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    
    new_portfolio_ids = set(new_portfolio['Unnamed: 0'].tolist())
    existing_index_ids = set(existing_index_df['ID'].tolist())
    
    missing_ids = existing_index_ids - new_portfolio_ids
    
    if missing_ids:
        missing_stocks_df = existing_index_df[existing_index_df['ID'].isin(missing_ids)].copy()
        missing_stocks_df['factor_score'] = missing_stocks_df['ID'].map(factor_scores)
        missing_stocks_df['quartile'] = missing_stocks_df['ID'].map(quartiles)
        
        missing_quartile_dist = missing_stocks_df['quartile'].value_counts().reindex(
            ['Q1', 'Q2', 'Q3', 'Q4'], fill_value=0
        )
        missing_quartile_pct = (missing_quartile_dist / len(missing_stocks_df) * 100).round(1)
        
        missing_stocks_sorted = missing_stocks_df.sort_values(
            'factor_score', 
            ascending=(factor_name == 'Low Volatility')
        )
        
        return {
            'missing_count': len(missing_stocks_df),
            'total_existing': len(existing_index_df),
            'missing_quartile_dist': missing_quartile_dist,
            'missing_quartile_pct': missing_quartile_pct,
            'missing_stocks_detailed': missing_stocks_sorted,
            'worst_misses': missing_stocks_sorted,
            'factor_name': factor_name
        }
    
    return {
        'missing_count': 0,
        'total_existing': len(existing_index_df),
        'missing_quartile_dist': pd.Series([0, 0, 0, 0], index=['Q1', 'Q2', 'Q3', 'Q4']),
        'missing_quartile_pct': pd.Series([0, 0, 0, 0], index=['Q1', 'Q2', 'Q3', 'Q4']),
        'missing_stocks_detailed': pd.DataFrame(),
        'worst_misses': pd.DataFrame(),
        'factor_name': factor_name
    }

# Header
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; text-align: center; margin-bottom: 2rem; border-radius: 10px;">
    <h1 style="margin: 0; font-size: 2.5rem; font-weight: 300;">Portfolio Analysis Dashboard</h1>
    <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">Interactive analysis of Low Volatility and Momentum portfolios</p>
</div>
""", unsafe_allow_html=True)

# Load data
try:
    with st.spinner("Loading data..."):
        (low_vol_30, momentum_15, all_momentum, all_volatility, 
         existing_low_vol_df, existing_momentum_df, name_to_id_mapping) = load_and_process_data()
        
        # Convert names to IDs
        existing_low_vol_df['ID'] = convert_names_to_ids(existing_low_vol_df['Company Name'].tolist(), name_to_id_mapping)
        existing_momentum_df['ID'] = convert_names_to_ids(existing_momentum_df['Company Name'].tolist(), name_to_id_mapping)
        
        # Filter out unmapped stocks
        existing_low_vol_df = existing_low_vol_df[existing_low_vol_df['ID'] != 'UNMAPPED']
        existing_momentum_df = existing_momentum_df[existing_momentum_df['ID'] != 'UNMAPPED']
        
        # Calculate all data
        low_vol_overlap_data = calculate_weighted_overlap_data(low_vol_30, existing_low_vol_df)
        momentum_overlap_data = calculate_weighted_overlap_data(momentum_15, existing_momentum_df)
        
        low_vol_quartile_data = calculate_quartile_data(existing_low_vol_df, all_volatility, 'Low Volatility')
        momentum_quartile_data = calculate_quartile_data(existing_momentum_df, all_momentum, 'Momentum')
        
        low_vol_missing_data = calculate_missing_stocks_analysis(low_vol_30, existing_low_vol_df, all_volatility, 'Low Volatility')
        momentum_missing_data = calculate_missing_stocks_analysis(momentum_15, existing_momentum_df, all_momentum, 'Momentum')

except Exception as e:
    st.error(f"Error processing data: {e}")
    st.stop()

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Portfolio Overlap Analysis", "Quartile Distribution", "Factor Score Analysis", "Missing Stocks Analysis"])

with tab1:
    st.subheader("Portfolio Overlap Analysis")
    
    # Low Volatility Section
    st.markdown("### Low Volatility Portfolio Overlap")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Overlapping Stocks",
            value=f"{low_vol_overlap_data['overlap_count']}"
        )
    
    with col2:
        st.metric(
            label="Stock Overlap %",
            value=f"{low_vol_overlap_data['overlap_count']/low_vol_overlap_data['total_new']:.1%}"
        )
    
    with col3:
        st.metric(
            label="Weight Overlap",
            value=f"{low_vol_overlap_data['weight_overlap']:.1f}%"
        )
    
    st.markdown("#### Top Overlapping Stocks by Weight")
    if not low_vol_overlap_data['merged_data'].empty:
        display_data = low_vol_overlap_data['merged_data'].head(10).copy()
        display_data.columns = ['Stock ID', 'Company Name', 'New Weight (%)', 'Existing Weight (%)', 'Overlap Weight (%)']
        st.dataframe(display_data, use_container_width=True)
    else:
        st.info("No overlapping stocks found.")
    
    st.markdown("---")
    
    # Momentum Section
    st.markdown("### Momentum Portfolio Overlap")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Overlapping Stocks",
            value=f"{momentum_overlap_data['overlap_count']}"
        )
    
    with col2:
        st.metric(
            label="Stock Overlap %",
            value=f"{momentum_overlap_data['overlap_count']/momentum_overlap_data['total_new']:.1%}"
        )
    
    with col3:
        st.metric(
            label="Weight Overlap",
            value=f"{momentum_overlap_data['weight_overlap']:.1f}%"
        )
    
    st.markdown("#### Top Overlapping Stocks by Weight")
    if not momentum_overlap_data['merged_data'].empty:
        display_data = momentum_overlap_data['merged_data'].head(10).copy()
        display_data.columns = ['Stock ID', 'Company Name', 'New Weight (%)', 'Existing Weight (%)', 'Overlap Weight (%)']
        st.dataframe(display_data, use_container_width=True)
    else:
        st.info("No overlapping stocks found.")

with tab2:
    st.subheader("Quartile Distribution")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        low_vol_quartile_fig = px.bar(
            x=['Q1 (Top)', 'Q2', 'Q3', 'Q4 (Bottom)'],
            y=low_vol_quartile_data['quartile_pct'].values,
            title="Low Volatility Index - Quartile Distribution",
            labels={'x': 'Quartile', 'y': 'Percentage (%)'},
            color=low_vol_quartile_data['quartile_pct'].values,
            color_continuous_scale='RdYlBu_r'
        )
        low_vol_quartile_fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(low_vol_quartile_fig, use_container_width=True)
    
    with col2:
        momentum_quartile_fig = px.bar(
            x=['Q1 (Top)', 'Q2', 'Q3', 'Q4 (Bottom)'],
            y=momentum_quartile_data['quartile_pct'].values,
            title="Momentum Index - Quartile Distribution",
            labels={'x': 'Quartile', 'y': 'Percentage (%)'},
            color=momentum_quartile_data['quartile_pct'].values,
            color_continuous_scale='RdYlBu_r'
        )
        momentum_quartile_fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(momentum_quartile_fig, use_container_width=True)
    
    # Top stocks tables
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Low Vol - Top 5 Stocks by Factor Score")
        top_low_vol = low_vol_quartile_data['top_stocks'][['Company Name', 'factor_score']].copy()
        top_low_vol.columns = ['Company Name', 'Factor Score']
        st.dataframe(top_low_vol, use_container_width=True)
    
    with col2:
        st.markdown("#### Momentum - Top 5 Stocks by Factor Score")
        top_momentum = momentum_quartile_data['top_stocks'][['Company Name', 'factor_score']].copy()
        top_momentum.columns = ['Company Name', 'Factor Score']
        st.dataframe(top_momentum, use_container_width=True)
    
    # Complete lists
    st.markdown("### Complete List of Stocks by Quartile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Low Volatility")
        low_vol_complete = low_vol_quartile_data['all_stocks_with_quartile'].copy()
        low_vol_complete.columns = ['Stock ID', 'Company Name', 'Factor Score', 'Quartile']
        st.dataframe(low_vol_complete, use_container_width=True, height=400)
    
    with col2:
        st.markdown("#### Momentum")
        momentum_complete = momentum_quartile_data['all_stocks_with_quartile'].copy()
        momentum_complete.columns = ['Stock ID', 'Company Name', 'Factor Score', 'Quartile']
        st.dataframe(momentum_complete, use_container_width=True, height=400)

with tab3:
    st.subheader("Factor Score Analysis")
    
    # Low Volatility Distribution
    low_vol_dist_fig = go.Figure()
    low_vol_dist_fig.add_trace(go.Histogram(
        x=low_vol_quartile_data['universe_scores'],
        name='Full Universe',
        opacity=0.5,
        nbinsx=30
    ))
    low_vol_dist_fig.add_trace(go.Histogram(
        x=low_vol_quartile_data['all_scores'],
        name='Index Constituents',
        opacity=0.8,
        nbinsx=15
    ))
    low_vol_dist_fig.add_vline(
        x=low_vol_quartile_data['universe_scores'].quantile(0.75),
        line_dash="dash",
        line_color="blue",
        annotation_text="Q1 Boundary"
    )
    low_vol_dist_fig.add_vline(
        x=low_vol_quartile_data['universe_scores'].quantile(0.25),
        line_dash="dash",
        line_color="red",
        annotation_text="Q3 Boundary"
    )
    low_vol_dist_fig.update_layout(
        title="Low Volatility Factor Score Distribution",
        xaxis_title="Factor Score",
        yaxis_title="Number of Stocks",
        height=500
    )
    st.plotly_chart(low_vol_dist_fig, use_container_width=True)
    
    # Momentum Distribution
    momentum_dist_fig = go.Figure()
    momentum_dist_fig.add_trace(go.Histogram(
        x=momentum_quartile_data['universe_scores'],
        name='Full Universe',
        opacity=0.5,
        nbinsx=30
    ))
    momentum_dist_fig.add_trace(go.Histogram(
        x=momentum_quartile_data['all_scores'],
        name='Index Constituents',
        opacity=0.8,
        nbinsx=15
    ))
    momentum_dist_fig.add_vline(
        x=momentum_quartile_data['universe_scores'].quantile(0.75),
        line_dash="dash",
        line_color="blue",
        annotation_text="Q1 Boundary"
    )
    momentum_dist_fig.add_vline(
        x=momentum_quartile_data['universe_scores'].quantile(0.25),
        line_dash="dash",
        line_color="red",
        annotation_text="Q3 Boundary"
    )
    momentum_dist_fig.update_layout(
        title="Momentum Factor Score Distribution",
        xaxis_title="Factor Score",
        yaxis_title="Number of Stocks",
        height=500
    )
    st.plotly_chart(momentum_dist_fig, use_container_width=True)

with tab4:
    st.subheader("Missing Stocks Analysis")
    
    # Summary metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Low Volatility - Missing Stocks Summary")
        sub_col1, sub_col2, sub_col3 = st.columns(3)
        
        with sub_col1:
            st.metric(
                label="Stocks Dropped",
                value=f"{low_vol_missing_data['missing_count']}"
            )
        
        with sub_col2:
            st.metric(
                label="Turnover Rate",
                value=f"{low_vol_missing_data['missing_count']/low_vol_missing_data['total_existing']:.1%}"
            )
        
        with sub_col3:
            st.metric(
                label="Q1 Stocks Lost",
                value=f"{low_vol_missing_data['missing_quartile_dist']['Q1']}"
            )
    
    with col2:
        st.markdown("### Momentum - Missing Stocks Summary")
        sub_col1, sub_col2, sub_col3 = st.columns(3)
        
        with sub_col1:
            st.metric(
                label="Stocks Dropped",
                value=f"{momentum_missing_data['missing_count']}"
            )
        
        with sub_col2:
            st.metric(
                label="Turnover Rate",
                value=f"{momentum_missing_data['missing_count']/momentum_missing_data['total_existing']:.1%}"
            )
        
        with sub_col3:
            st.metric(
                label="Q1 Stocks Lost",
                value=f"{momentum_missing_data['missing_quartile_dist']['Q1']}"
            )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        low_vol_missing_fig = px.bar(
            x=['Q1 (Top)', 'Q2', 'Q3', 'Q4 (Bottom)'],
            y=low_vol_missing_data['missing_quartile_pct'].values,
            title="Low Volatility - Missing Stocks Quartile Distribution",
            labels={'x': 'Quartile', 'y': 'Percentage of Missing Stocks (%)'},
            color=low_vol_missing_data['missing_quartile_pct'].values,
            color_continuous_scale='Reds'
        )
        low_vol_missing_fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(low_vol_missing_fig, use_container_width=True)
    
    with col2:
        momentum_missing_fig = px.bar(
            x=['Q1 (Top)', 'Q2', 'Q3', 'Q4 (Bottom)'],
            y=momentum_missing_data['missing_quartile_pct'].values,
            title="Momentum - Missing Stocks Quartile Distribution",
            labels={'x': 'Quartile', 'y': 'Percentage of Missing Stocks (%)'},
            color=momentum_missing_data['missing_quartile_pct'].values,
            color_continuous_scale='Reds'
        )
        momentum_missing_fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(momentum_missing_fig, use_container_width=True)
    
    # Detailed tables
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Low Vol - Top 10 Best Performing Stocks That Were Dropped")
        if not low_vol_missing_data['worst_misses'].empty:
            low_vol_dropped = low_vol_missing_data['worst_misses'][
                ['ID', 'Company Name', 'Holding Percentage', 'factor_score', 'quartile']
            ].head(10).copy()
            low_vol_dropped.columns = ['Stock ID', 'Company Name', 'Previous Weight (%)', 'Factor Score', 'Quartile']
            st.dataframe(low_vol_dropped, use_container_width=True)
        else:
            st.info("No missing stocks data available.")
    
    with col2:
        st.markdown("#### Momentum - Top 10 Best Performing Stocks That Were Dropped")
        if not momentum_missing_data['worst_misses'].empty:
            momentum_dropped = momentum_missing_data['worst_misses'][
                ['ID', 'Company Name', 'Holding Percentage', 'factor_score', 'quartile']
            ].head(10).copy()
            momentum_dropped.columns = ['Stock ID', 'Company Name', 'Previous Weight (%)', 'Factor Score', 'Quartile']
            st.dataframe(momentum_dropped, use_container_width=True)
        else:
            st.info("No missing stocks data available.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 1rem;">
        Portfolio Analysis Dashboard | Built with Streamlit
    </div>
    """,
    unsafe_allow_html=True
)