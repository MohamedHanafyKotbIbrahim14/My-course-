import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Course Analysis Tool Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize default font settings if not in session state
if 'font_size' not in st.session_state:
    st.session_state.font_size = 16
if 'header_size' not in st.session_state:
    st.session_state.header_size = 24
if 'make_bold' not in st.session_state:
    st.session_state.make_bold = True
if 'plot_size_multiplier' not in st.session_state:
    st.session_state.plot_size_multiplier = 1.2

# Custom CSS for better styling with dynamic font sizes
def get_dynamic_css(font_size, header_size, make_bold):
    font_weight = "bold" if make_bold else "normal"
    return f"""
<style>
    /* Dynamic font settings */
    .stMarkdown, .stText, p, span, label {{
        font-size: {font_size}px !important;
        font-weight: {font_weight} !important;
    }}
    
    h1 {{
        font-size: {header_size + 8}px !important;
        font-weight: bold !important;
    }}
    
    h2 {{
        font-size: {header_size + 4}px !important;
        font-weight: bold !important;
    }}
    
    h3 {{
        font-size: {header_size}px !important;
        font-weight: bold !important;
    }}
    
    h4 {{
        font-size: {header_size - 2}px !important;
        font-weight: bold !important;
    }}
    
    /* Dataframe text */
    .dataframe {{
        font-size: {font_size}px !important;
    }}
    
    .main-header {{
        font-size: {header_size + 12}px !important;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }}
    
    .sub-header {{
        text-align: center;
        color: #666;
        font-size: {font_size + 2}px !important;
        margin-bottom: 2rem;
        font-weight: {font_weight} !important;
    }}
    
    /* Special styling for Analysis Scope radio buttons */
    .analysis-scope-container {{
        background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        animation: pulse 2s ease-in-out infinite;
    }}
    
    @keyframes pulse {{
        0% {{
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }}
        50% {{
            box-shadow: 0 8px 25px rgba(255,107,107,0.4);
        }}
        100% {{
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }}
    }}
    
    .analysis-scope-title {{
        font-size: {header_size}px !important;
        font-weight: bold !important;
        color: #2d3436 !important;
        text-align: center;
        margin-bottom: 15px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }}
    
    /* Radio button styling */
    .stRadio > div {{
        font-size: {font_size + 2}px !important;
        font-weight: bold !important;
    }}
    
    .stRadio > div[role="radiogroup"] > label {{
        background-color: rgba(255, 255, 255, 0.9);
        padding: 12px 20px !important;
        border-radius: 10px;
        margin: 5px !important;
        transition: all 0.3s ease;
        font-size: {font_size + 2}px !important;
        font-weight: bold !important;
    }}
    
    .stRadio > div[role="radiogroup"] > label:hover {{
        background-color: #667eea;
        color: white;
        transform: scale(1.05);
    }}
    
    /* Metric styling */
    .metric-card {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        font-size: {font_size}px !important;
        font-weight: {font_weight} !important;
    }}
    
    /* Tab labels */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {{
        font-size: {font_size + 2}px !important;
        font-weight: bold !important;
    }}
    
    /* Buttons */
    .stButton > button {{
        font-size: {font_size}px !important;
        font-weight: bold !important;
        padding: 10px 20px;
    }}
    
    /* Selectbox and other inputs */
    .stSelectbox label, .stTextInput label, .stNumberInput label {{
        font-size: {font_size}px !important;
        font-weight: {font_weight} !important;
    }}
    
    .stSelectbox > div > div {{
        font-size: {font_size}px !important;
    }}
    
    /* Info boxes */
    .stAlert > div {{
        font-size: {font_size}px !important;
        font-weight: {font_weight} !important;
    }}
    
    /* Highlight box */
    .highlight-box {{
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }}
</style>
"""

# Apply dynamic CSS with current settings
if 'font_size' in st.session_state:
    st.markdown(get_dynamic_css(
        st.session_state.font_size,
        st.session_state.header_size,
        st.session_state.make_bold
    ), unsafe_allow_html=True)

# Helper function to style matplotlib plots with font settings
def style_matplotlib_plot(ax, title=None, xlabel=None, ylabel=None):
    """Apply font settings to matplotlib plot"""
    font_size = st.session_state.get('font_size', 16)
    header_size = st.session_state.get('header_size', 24)
    weight = 'bold' if st.session_state.get('make_bold', True) else 'normal'
    
    # Title
    if title:
        ax.set_title(title, fontsize=header_size, fontweight='bold')
    
    # Axis labels
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=font_size + 2, fontweight=weight)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=font_size + 2, fontweight=weight)
    
    # Tick labels
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    for label in ax.get_xticklabels():
        label.set_fontweight(weight)
    for label in ax.get_yticklabels():
        label.set_fontweight(weight)
    
    # Legend if present
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_fontsize(font_size)
            text.set_fontweight(weight)

# Helper function to get plotly font settings
def get_plotly_font_settings():
    """Get font settings for plotly plots"""
    font_size = st.session_state.get('font_size', 16)
    header_size = st.session_state.get('header_size', 24)
    weight = 'bold' if st.session_state.get('make_bold', True) else 'normal'
    
    return {
        'title_font': {'size': header_size, 'family': 'Arial, sans-serif'},
        'font': {'size': font_size, 'family': 'Arial, sans-serif'},
        'xaxis': {
            'title_font': {'size': font_size + 2},
            'tickfont': {'size': font_size}
        },
        'yaxis': {
            'title_font': {'size': font_size + 2},
            'tickfont': {'size': font_size}
        }
    }

# Header
st.markdown('<h1 class="main-header">🎓 Course Analysis Tool Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Analytics & Comparison Platform for Educational Data</p>', unsafe_allow_html=True)

# Initialize session state
if 'files_uploaded' not in st.session_state:
    st.session_state.files_uploaded = False
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# Sidebar for file upload and settings
with st.sidebar:
    st.markdown("## 📁 File Upload Center")
    st.markdown("---")
    
    # File upload section
    uploaded_files = st.file_uploader(
        "Upload CSV Files",
        type=['csv', 'CSV'],
        accept_multiple_files=True,
        help="Upload at least 2 CSV files for comparison",
        key="file_uploader"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
        st.session_state.files_uploaded = True
        
        # Display uploaded files
        st.markdown("### 📋 Uploaded Files:")
        for i, file in enumerate(uploaded_files, 1):
            st.markdown(f"{i}. 📄 **{file.name}**")
            file_size = file.size / 1024  # Convert to KB
            st.caption(f"   Size: {file_size:.2f} KB")
    
    st.markdown("---")
    
    # Font and Display Settings
    st.markdown("## 🔤 Display Settings")
    
    st.session_state.font_size = st.slider(
        "📏 Base Font Size",
        min_value=12,
        max_value=24,
        value=st.session_state.font_size,
        step=1,
        help="Adjust the base font size for all text"
    )
    
    st.session_state.header_size = st.slider(
        "📐 Header Font Size",
        min_value=18,
        max_value=36,
        value=st.session_state.header_size,
        step=2,
        help="Adjust the header font size"
    )
    
    st.session_state.make_bold = st.checkbox(
        "🔤 **Bold Text**",
        value=st.session_state.make_bold,
        help="Make all text bold for better visibility"
    )
    
    st.session_state.plot_size_multiplier = st.slider(
        "📊 Plot Size Multiplier",
        min_value=0.8,
        max_value=2.0,
        value=st.session_state.plot_size_multiplier,
        step=0.1,
        help="Adjust the size of all plots and charts"
    )
    
    st.markdown("---")
    
    # Chart type preference
    chart_type = st.selectbox(
        "📈 Chart Style",
        ["Interactive (Plotly)", "Static (Matplotlib)"],
        help="Choose between interactive or static charts"
    )
    
    # Export options
    st.markdown("### 💾 Export Options")
    export_format = st.radio(
        "Export Format",
        ["CSV", "Excel", "HTML Report"],
        horizontal=True
    )
    
    st.markdown("---")
    
    # Info section
    st.markdown("### ℹ️ About")
    st.info("""
    **Course Analysis Tool Pro**
    Version 2.0
    
    Features:
    • Multi-file comparison
    • Advanced statistics
    • Interactive visualizations
    • Grade distribution analysis
    • Export capabilities
    """)

# Use default color theme
theme_colors = {
    "primary": "#667eea",
    "secondary": "#764ba2", 
    "grades": ['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336'],
    "histogram": "#667eea"
}

# Helper functions
def process_dataframe(df):
    """Process and clean the dataframe"""
    # Skip first row if it's header info
    if 'Student ID' in df.columns and len(df) > 0 and str(df.iloc[0, 0]).startswith('ACTL'):
        df = df.iloc[1:].reset_index(drop=True)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # COLUMN D (index 3): ALWAYS contains Final Mark + Grade
    if len(df.columns) > 3:
        final_col = df.iloc[:, 3]  # Column D
        df['Final_Mark'] = final_col.astype(str).str.extract(r'(\d+)').astype(float)
        df['Grade'] = final_col.astype(str).str.extract(r'([A-Z]{2,3})')[0]
    
    # Clean Student ID
    df['Student ID'] = df.iloc[:, 0].astype(str).str.strip()
    
    return df

def get_grade_distribution(df):
    """Calculate grade distribution percentages"""
    grade_counts = df['Grade'].value_counts()
    total = len(df[df['Grade'].notna()])
    
    grade_order = ['HD', 'DN', 'CR', 'PS', 'FL']
    distribution = {}
    
    for grade in grade_order:
        count = grade_counts.get(grade, 0)
        percentage = (count / total * 100) if total > 0 else 0
        distribution[grade] = {'count': count, 'percentage': percentage}
    
    return distribution

def get_assessment_columns(df):
    """Get all assessment columns: Column D + Column E onwards"""
    if len(df.columns) <= 3:
        return []
    
    # Get the actual Column D header name
    col_d_name = df.columns[3]  # Column D actual name
    
    # Start with Column D
    assessment_cols = [col_d_name]
    
    # Add columns from index 4 onwards (Column E+)
    for i in range(4, len(df.columns)):
        col_name = df.columns[i]
        if col_name not in ['Final_Mark', 'Grade', 'Student ID']:
            assessment_cols.append(col_name)
    
    return assessment_cols

def create_download_link(df, filename, file_format='csv'):
    """Create a download link for dataframe"""
    if file_format == 'csv':
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        return f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">📥 Download {filename}.csv</a>'
    elif file_format == 'excel':
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
        excel_data = output.getvalue()
        b64 = base64.b64encode(excel_data).decode()
        return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx">📥 Download {filename}.xlsx</a>'

def create_grade_pie_chart(distribution, title, use_plotly=True, colors=None):
    """Create a pie chart for grade distribution"""
    grades = []
    percentages = []
    if colors is None:
        colors = ['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336']  # Default HD to FL colors
    
    for grade in ['HD', 'DN', 'CR', 'PS', 'FL']:
        if grade in distribution:
            grades.append(f"{grade} ({distribution[grade]['count']})")
            percentages.append(distribution[grade]['percentage'])
    
    if use_plotly:
        fig = px.pie(
            values=percentages,
            names=grades,
            title=title,
            color_discrete_sequence=colors,
            hole=0.3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        # Apply font settings
        font_settings = get_plotly_font_settings()
        fig.update_layout(**font_settings)
        return fig
    else:
        plot_mult = st.session_state.get('plot_size_multiplier', 1.2)
        fig, ax = plt.subplots(figsize=(8 * plot_mult, 6 * plot_mult))
        wedges, texts, autotexts = ax.pie(percentages, labels=grades, colors=colors, 
                                          autopct='%1.1f%%', startangle=90)
        
        # Apply font settings to pie chart
        font_size = st.session_state.get('font_size', 16)
        weight = 'bold' if st.session_state.get('make_bold', True) else 'normal'
        
        # Style the labels
        for text in texts:
            text.set_fontsize(font_size)
            text.set_fontweight(weight)
        
        # Style the percentages
        for autotext in autotexts:
            autotext.set_fontsize(font_size)
            autotext.set_fontweight(weight)
            autotext.set_color('white')
        
        ax.set_title(title, fontsize=st.session_state.get('header_size', 24), fontweight='bold')
        return fig

# Main content
if not uploaded_files or len(uploaded_files) < 2:
    # Welcome screen when no files are uploaded
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="highlight-box">
        <h3 style="text-align: center;">🚀 Getting Started</h3>
        <p style="text-align: center;">Upload at least 2 CSV files using the sidebar to begin your analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Features showcase
    st.markdown("## 🌟 Key Features")
    
    feature_cols = st.columns(4)
    
    with feature_cols[0]:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
        <h1>📊</h1>
        <h4>Advanced Analytics</h4>
        <p>Comprehensive statistical analysis with visual insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feature_cols[1]:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
        <h1>🔄</h1>
        <h4>Multi-File Comparison</h4>
        <p>Compare multiple courses side by side</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feature_cols[2]:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
        <h1>📈</h1>
        <h4>Interactive Visualizations</h4>
        <p>Dynamic charts with Plotly integration</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feature_cols[3]:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
        <h1>💾</h1>
        <h4>Export Reports</h4>
        <p>Download results in multiple formats</p>
        </div>
        """, unsafe_allow_html=True)
    
    
else:
    # Process uploaded files
    dataframes = []
    file_names = []
    
    for file in uploaded_files:
        df = pd.read_csv(file)
        df = process_dataframe(df)
        dataframes.append(df)
        file_names.append(file.name)
    
    # Create tabs for different views
    tab1, tab2, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🔄 Comparison", 
        "🎯 Individual Performance",
        "📥 Export Results"
    ])
    
    with tab1:
        st.markdown("## 📊 Files Overview")
        
        # Create metrics for each file
        cols = st.columns(min(len(dataframes), 4))
        
        for idx, (df, name) in enumerate(zip(dataframes, file_names)):
            with cols[idx % 4]:
                st.markdown(f"### 📄 {name}")
                st.metric("Total Students", len(df))
                
                if 'Final_Mark' in df.columns:
                    avg_mark = df['Final_Mark'].mean()
                    st.metric("Average Mark", f"{avg_mark:.1f}")
                
                if 'Grade' in df.columns:
                    dist = get_grade_distribution(df)
                    hd_dn_pct = dist.get('HD', {}).get('percentage', 0) + dist.get('DN', {}).get('percentage', 0)
                    st.metric("HD+DN %", f"{hd_dn_pct:.1f}%")
        
        st.markdown("---")
        
        # Grade distribution comparison
        st.markdown("## 🎯 Grade Distribution Comparison")
        
        use_plotly = (chart_type == "Interactive (Plotly)")
        
        if use_plotly:
            fig = make_subplots(
                rows=1, cols=len(dataframes),
                subplot_titles=file_names,
                specs=[[{'type': 'pie'}] * len(dataframes)]
            )
            
            for idx, (df, name) in enumerate(zip(dataframes, file_names)):
                if 'Grade' in df.columns:
                    dist = get_grade_distribution(df)
                    grades = []
                    percentages = []
                    
                    for grade in ['HD', 'DN', 'CR', 'PS', 'FL']:
                        if grade in dist:
                            grades.append(f"{grade} ({dist[grade]['count']})")
                            percentages.append(dist[grade]['percentage'])
                    
                    fig.add_trace(
                        go.Pie(
                            labels=grades,
                            values=percentages,
                            hole=0.3,
                            marker_colors=theme_colors['grades'][:len(grades)],
                            textposition='inside',
                            textinfo='percent+label'
                        ),
                        row=1, col=idx+1
                    )
            
            # Apply font settings
            font_settings = get_plotly_font_settings()
            fig.update_layout(
                height=int(400 * st.session_state.get('plot_size_multiplier', 1.2)), 
                showlegend=False,
                **font_settings
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            cols = st.columns(len(dataframes))
            for idx, (df, name) in enumerate(zip(dataframes, file_names)):
                with cols[idx]:
                    if 'Grade' in df.columns:
                        dist = get_grade_distribution(df)
                        fig = create_grade_pie_chart(dist, name, use_plotly=False, colors=theme_colors['grades'])
                        st.pyplot(fig)
    
    with tab2:
        st.markdown("## 🔄 File Comparison")
        
        if len(dataframes) < 2:
            st.warning("Need at least 2 files for comparison")
        else:
            # File selection
            col1, col2 = st.columns(2)
            
            with col1:
                file1_idx = st.selectbox(
                    "Select First File",
                    range(len(file_names)),
                    format_func=lambda x: file_names[x],
                    key="file1_select"
                )
            
            with col2:
                file2_idx = st.selectbox(
                    "Select Second File",
                    range(len(file_names)),
                    format_func=lambda x: file_names[x],
                    key="file2_select"
                )
            
            if file1_idx == file2_idx:
                st.error("⚠️ Please select two different files!")
            else:
                df1 = dataframes[file1_idx]
                df2 = dataframes[file2_idx]
                name1 = file_names[file1_idx]
                name2 = file_names[file2_idx]
                
                st.success(f"✅ Comparing: **{name1}** vs **{name2}**")
                
                # Get assessment columns
                cols1 = get_assessment_columns(df1)
                cols2 = get_assessment_columns(df2)
                
                if cols1 and cols2:
                    st.markdown("---")
                    st.markdown("### 📊 Select Columns to Compare")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        selected_col1 = st.selectbox(
                            f"Column from {name1}",
                            cols1,
                            key="comp_col1"
                        )
                    
                    with col2:
                        selected_col2 = st.selectbox(
                            f"Column from {name2}",
                            cols2,
                            key="comp_col2"
                        )
                    
                    # Prepare data for comparison
                    col_d_name1 = df1.columns[3] if len(df1.columns) > 3 else None
                    col_d_name2 = df2.columns[3] if len(df2.columns) > 3 else None
                    
                    if selected_col1 == col_d_name1:
                        col1_data = 'Final_Mark'
                    else:
                        col1_data = selected_col1
                        df1[selected_col1] = pd.to_numeric(df1[selected_col1], errors='coerce')
                    
                    if selected_col2 == col_d_name2:
                        col2_data = 'Final_Mark'
                    else:
                        col2_data = selected_col2
                        df2[selected_col2] = pd.to_numeric(df2[selected_col2], errors='coerce')
                    
                    # Find common students
                    common_ids = set(df1['Student ID']) & set(df2['Student ID'])
                    
                    # Metrics
                    st.markdown("---")
                    st.markdown("### 📊 Student Overview")
                    
                    met_cols = st.columns(4)
                    met_cols[0].metric("Students in File 1", len(df1))
                    met_cols[1].metric("Students in File 2", len(df2))
                    met_cols[2].metric("Common Students", len(common_ids))
                    met_cols[3].metric("Unique Total", len(set(df1['Student ID']) | set(df2['Student ID'])))
                    
                    # Filter option
                    st.markdown("---")
                    
                    # Create prominent Analysis Scope section
                    st.markdown("""
                    <div class="analysis-scope-container">
                        <h3 class="analysis-scope-title">⚡ IMPORTANT: Select Analysis Scope ⚡</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add emphasis box
                    with st.container():
                        col1, col2, col3 = st.columns([1, 3, 1])
                        with col2:
                            st.warning("⚠️ **This selection affects ALL analysis below!** Choose carefully:")
                            filter_option = st.radio(
                                "",
                                ["🌍 **ALL Students** - Analyze everyone in each file", 
                                 "🎯 **ONLY Common Students** - Analyze students in BOTH files"],
                                horizontal=False,
                                key="filter_radio"
                            )
                    
                    # Extract the actual option value
                    if "ALL Students" in filter_option:
                        show_common_only = False
                        st.success("✅ Analyzing **ALL STUDENTS** from each file independently")
                    else:
                        show_common_only = True
                        st.error("🎯 Analyzing **ONLY STUDENTS** present in BOTH files")
                    
                    # Apply filtering based on selection
                    if show_common_only:
                        # Filter for common students only
                        df1_filtered = df1[df1['Student ID'].isin(common_ids)]
                        df2_filtered = df2[df2['Student ID'].isin(common_ids)]
                        st.info(f"🎯 Analyzing **{len(common_ids)}** common students")
                    else:
                        # Use all students
                        df1_filtered = df1.copy()
                        df2_filtered = df2.copy()
                        st.info(f"🌍 Analyzing ALL students - File 1: **{len(df1)}** | File 2: **{len(df2)}**")
                    
                    # Statistics section (works for both ALL and COMMON)
                    st.markdown("### 📊 Statistical Comparison")
                    
                    stat_col1, stat_col2 = st.columns(2)
                    
                    with stat_col1:
                        st.markdown(f"#### {name1} - {selected_col1}")
                        data1 = df1_filtered[col1_data].dropna()
                        if len(data1) > 0:
                            st.metric("Mean", f"{data1.mean():.1f}")
                            st.metric("Std Dev", f"{data1.std():.1f}")
                            st.metric("Median", f"{data1.median():.1f}")
                    
                    with stat_col2:
                        st.markdown(f"#### {name2} - {selected_col2}")
                        data2 = df2_filtered[col2_data].dropna()
                        if len(data2) > 0:
                            st.metric("Mean", f"{data2.mean():.1f}")
                            st.metric("Std Dev", f"{data2.std():.1f}")
                            st.metric("Median", f"{data2.median():.1f}")
                    
                    # Scatter plot - ONLY for common students
                    if show_common_only and len(common_ids) > 0:
                        st.markdown("### 📈 Correlation Analysis")
                        
                        # Prepare data for scatter plot
                        plot_df1 = df1_filtered[['Student ID', col1_data]].copy()
                        plot_df1.columns = ['Student ID', 'Metric1']
                        
                        plot_df2 = df2_filtered[['Student ID', col2_data]].copy()
                        plot_df2.columns = ['Student ID', 'Metric2']
                        
                        plot_df = plot_df1.merge(plot_df2, on='Student ID', how='inner')
                        plot_df = plot_df.dropna()
                        
                        if len(plot_df) > 0:
                            use_plotly = (chart_type == "Interactive (Plotly)")
                            
                            if use_plotly:
                                # Create scatter plot without trendline (to avoid statsmodels dependency)
                                fig = px.scatter(
                                    plot_df, 
                                    x='Metric1', 
                                    y='Metric2',
                                    hover_data=['Student ID'],
                                    title=f'{selected_col1} vs {selected_col2}',
                                    labels={'Metric1': f'{selected_col1} ({name1})', 
                                           'Metric2': f'{selected_col2} ({name2})'},
                                    color_discrete_sequence=[theme_colors['primary']]
                                )
                                
                                # Add diagonal line
                                min_val = min(plot_df['Metric1'].min(), plot_df['Metric2'].min())
                                max_val = max(plot_df['Metric1'].max(), plot_df['Metric2'].max())
                                fig.add_trace(
                                    go.Scatter(
                                        x=[min_val, max_val],
                                        y=[min_val, max_val],
                                        mode='lines',
                                        line=dict(dash='dash', color='gray'),
                                        name='Equal Performance',
                                        showlegend=True
                                    )
                                )
                                
                                # Apply dynamic height and font settings
                                font_settings = get_plotly_font_settings()
                                fig.update_layout(
                                    height=int(500 * st.session_state.plot_size_multiplier),
                                    **font_settings
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                plot_mult = st.session_state.get('plot_size_multiplier', 1.2)
                                fig, ax = plt.subplots(figsize=(10 * plot_mult, 6 * plot_mult))
                                ax.scatter(plot_df['Metric1'], plot_df['Metric2'], 
                                          c=theme_colors['primary'], alpha=0.6, s=50)
                                
                                min_val = min(plot_df['Metric1'].min(), plot_df['Metric2'].min())
                                max_val = max(plot_df['Metric1'].max(), plot_df['Metric2'].max())
                                ax.plot([min_val, max_val], [min_val, max_val], 
                                       'k--', alpha=0.3, label='Equal Performance')
                                
                                style_matplotlib_plot(ax,
                                                    title=f'{selected_col1} vs {selected_col2}',
                                                    xlabel=f'{selected_col1} ({name1})',
                                                    ylabel=f'{selected_col2} ({name2})')
                                ax.legend()
                                ax.grid(True, alpha=0.3)
                                
                                st.pyplot(fig)
                            
                            # Correlation coefficient
                            if len(plot_df) > 1:
                                correlation = plot_df['Metric1'].corr(plot_df['Metric2'])
                                st.info(f"📊 Correlation Coefficient: **{correlation:.3f}**")
                            
                            # Detailed comparison table
                            st.markdown("### 📋 Detailed Comparison Table")
                            
                            comparison_df = plot_df.copy()
                            comparison_df['Difference'] = comparison_df['Metric2'] - comparison_df['Metric1']
                            comparison_df = comparison_df.sort_values('Difference', ascending=False)
                            
                            # Add performance indicators
                            comparison_df['Performance'] = comparison_df['Difference'].apply(
                                lambda x: '🟢 Better' if x > 5 else ('🔴 Worse' if x < -5 else '🟡 Similar')
                            )
                            
                            # Rename columns for display
                            comparison_df.columns = [
                                'Student ID',
                                f'{selected_col1} ({name1})',
                                f'{selected_col2} ({name2})',
                                'Difference',
                                'Performance'
                            ]
                            
                            st.dataframe(
                                comparison_df.style.background_gradient(subset=['Difference']),
                                use_container_width=True,
                                height=400
                            )
                            
                            # Summary statistics
                            st.markdown("### 📊 Performance Summary")
                            
                            stat_cols = st.columns(3)
                            improved = (comparison_df['Difference'] > 5).sum()
                            declined = (comparison_df['Difference'] < -5).sum()
                            stable = ((comparison_df['Difference'] >= -5) & (comparison_df['Difference'] <= 5)).sum()
                            
                            stat_cols[0].metric("📈 Improved", improved, f"{improved/len(comparison_df)*100:.1f}%")
                            stat_cols[1].metric("📉 Declined", declined, f"{declined/len(comparison_df)*100:.1f}%")
                            stat_cols[2].metric("➡️ Stable", stable, f"{stable/len(comparison_df)*100:.1f}%")
                        else:
                            st.warning("No data available for scatter plot with the selected columns")
                    
                    # Distribution plots for ALL students with detailed analysis
                    elif not show_common_only:
                        st.markdown("### 📊 Comprehensive Analysis - ALL Students")
                        
                        use_plotly = (chart_type == "Interactive (Plotly)")
                        
                        # 1. HISTOGRAMS SIDE BY SIDE
                        st.markdown("#### 📈 Distribution Comparison")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**{name1} - {selected_col1}**")
                            data1 = df1_filtered[col1_data].dropna()
                            if len(data1) > 0:
                                if use_plotly:
                                    fig = px.histogram(
                                        x=data1,
                                        nbins=20,
                                        title=f'Distribution (n={len(data1)})',
                                        color_discrete_sequence=[theme_colors['histogram']]
                                    )
                                    fig.add_vline(x=data1.mean(), line_dash="dash", line_color="red",
                                                 annotation_text=f"Mean: {data1.mean():.1f}")
                                    # Apply font settings
                                    font_settings = get_plotly_font_settings()
                                    fig.update_layout(
                                        height=int(350 * st.session_state.plot_size_multiplier),
                                        **font_settings
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    plot_mult = st.session_state.get('plot_size_multiplier', 1.2)
                                    fig, ax = plt.subplots(figsize=(8 * plot_mult, 5 * plot_mult))
                                    ax.hist(data1, bins=20, color=theme_colors['histogram'], alpha=0.7, edgecolor='black')
                                    ax.axvline(data1.mean(), color='red', linestyle='--', 
                                             label=f'Mean: {data1.mean():.1f}')
                                    style_matplotlib_plot(ax, 
                                                        title=f'Distribution (n={len(data1)})',
                                                        xlabel=selected_col1,
                                                        ylabel='Frequency')
                                    ax.legend()
                                    ax.grid(True, alpha=0.3)
                                    st.pyplot(fig)
                        
                        with col2:
                            st.markdown(f"**{name2} - {selected_col2}**")
                            data2 = df2_filtered[col2_data].dropna()
                            if len(data2) > 0:
                                if use_plotly:
                                    fig = px.histogram(
                                        x=data2,
                                        nbins=20,
                                        title=f'Distribution (n={len(data2)})',
                                        color_discrete_sequence=[theme_colors['secondary']]
                                    )
                                    fig.add_vline(x=data2.mean(), line_dash="dash", line_color="red",
                                                 annotation_text=f"Mean: {data2.mean():.1f}")
                                    # Apply font settings
                                    font_settings = get_plotly_font_settings()
                                    fig.update_layout(
                                        height=int(350 * st.session_state.plot_size_multiplier),
                                        **font_settings
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    plot_mult = st.session_state.get('plot_size_multiplier', 1.2)
                                    fig, ax = plt.subplots(figsize=(8 * plot_mult, 5 * plot_mult))
                                    ax.hist(data2, bins=20, color=theme_colors['secondary'], alpha=0.7, edgecolor='black')
                                    ax.axvline(data2.mean(), color='red', linestyle='--', 
                                             label=f'Mean: {data2.mean():.1f}')
                                    style_matplotlib_plot(ax, 
                                                        title=f'Distribution (n={len(data2)})',
                                                        xlabel=selected_col2,
                                                        ylabel='Frequency')
                                    ax.legend()
                                    ax.grid(True, alpha=0.3)
                                    st.pyplot(fig)
                        
                        st.markdown("---")
                        
                        # 2. SUMMARY STATISTICS TABLES
                        st.markdown("#### 📊 Statistical Summary & Grade Distribution")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**{name1}**")
                            
                            # Statistics for Final Mark
                            if 'Final_Mark' in df1_filtered.columns:
                                stats1 = df1_filtered['Final_Mark'].describe()
                                
                                # Create a nice summary table
                                summary_data1 = {
                                    'Metric': ['Students', 'Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                                    'Value': [
                                        f"{stats1['count']:.0f}",
                                        f"{stats1['mean']:.2f}",
                                        f"{stats1['50%']:.2f}",
                                        f"{stats1['std']:.2f}",
                                        f"{stats1['min']:.2f}",
                                        f"{stats1['max']:.2f}"
                                    ]
                                }
                                
                                summary_df1 = pd.DataFrame(summary_data1)
                                st.dataframe(summary_df1, use_container_width=True, hide_index=True)
                                
                                # Grade distribution
                                if 'Grade' in df1_filtered.columns:
                                    st.markdown("**Grade Distribution:**")
                                    dist1 = get_grade_distribution(df1_filtered)
                                    
                                    # Grade breakdown table
                                    grade_data1 = []
                                    for grade in ['HD', 'DN', 'CR', 'PS', 'FL']:
                                        if grade in dist1:
                                            grade_data1.append({
                                                'Grade': grade,
                                                'Count': dist1[grade]['count'],
                                                'Percentage': f"{dist1[grade]['percentage']:.1f}%"
                                            })
                                    
                                    if grade_data1:
                                        grade_df1 = pd.DataFrame(grade_data1)
                                        st.dataframe(grade_df1, use_container_width=True, hide_index=True)
                        
                        with col2:
                            st.markdown(f"**{name2}**")
                            
                            # Statistics for Final Mark
                            if 'Final_Mark' in df2_filtered.columns:
                                stats2 = df2_filtered['Final_Mark'].describe()
                                
                                # Create a nice summary table
                                summary_data2 = {
                                    'Metric': ['Students', 'Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                                    'Value': [
                                        f"{stats2['count']:.0f}",
                                        f"{stats2['mean']:.2f}",
                                        f"{stats2['50%']:.2f}",
                                        f"{stats2['std']:.2f}",
                                        f"{stats2['min']:.2f}",
                                        f"{stats2['max']:.2f}"
                                    ]
                                }
                                
                                summary_df2 = pd.DataFrame(summary_data2)
                                st.dataframe(summary_df2, use_container_width=True, hide_index=True)
                                
                                # Grade distribution
                                if 'Grade' in df2_filtered.columns:
                                    st.markdown("**Grade Distribution:**")
                                    dist2 = get_grade_distribution(df2_filtered)
                                    
                                    # Grade breakdown table
                                    grade_data2 = []
                                    for grade in ['HD', 'DN', 'CR', 'PS', 'FL']:
                                        if grade in dist2:
                                            grade_data2.append({
                                                'Grade': grade,
                                                'Count': dist2[grade]['count'],
                                                'Percentage': f"{dist2[grade]['percentage']:.1f}%"
                                            })
                                    
                                    if grade_data2:
                                        grade_df2 = pd.DataFrame(grade_data2)
                                        st.dataframe(grade_df2, use_container_width=True, hide_index=True)
                        
                        st.markdown("---")
                        
                        # 3. CORRELATION MATRICES (Half/Triangular)
                        st.markdown("#### 🔗 Assessment Correlation Analysis")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**{name1} - Correlation Matrix**")
                            
                            # Get assessment columns for correlation
                            assessment_cols1 = get_assessment_columns(df1_filtered)
                            if len(assessment_cols1) > 1:
                                # Prepare numeric columns
                                numeric_cols1 = []
                                for col in assessment_cols1:  # Process all columns
                                    if col == df1_filtered.columns[3]:  # Column D
                                        if 'Final_Mark' in df1_filtered.columns:
                                            numeric_cols1.append('Final_Mark')
                                    else:
                                        df1_filtered[col] = pd.to_numeric(df1_filtered[col], errors='coerce')
                                        if df1_filtered[col].notna().sum() > 10:
                                            numeric_cols1.append(col)
                                
                                if len(numeric_cols1) > 1:
                                    corr_matrix1 = df1_filtered[numeric_cols1].corr()
                                    
                                    # Create mask for upper triangle
                                    mask = np.triu(np.ones_like(corr_matrix1, dtype=bool))
                                    
                                    if use_plotly:
                                        # Create triangular heatmap with plotly
                                        corr_masked = corr_matrix1.where(~mask)
                                        fig = px.imshow(
                                            corr_masked,
                                            color_continuous_scale='RdBu',
                                            zmin=-1, zmax=1,
                                            aspect='auto',
                                            text_auto='.2f'
                                        )
                                        # Apply font settings
                                        font_settings = get_plotly_font_settings()
                                        fig.update_layout(
                                            height=int(400 * st.session_state.plot_size_multiplier), 
                                            title="Lower Triangle Only",
                                            **font_settings
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        plot_mult = st.session_state.get('plot_size_multiplier', 1.2)
                                        fig, ax = plt.subplots(figsize=(8 * plot_mult, 6 * plot_mult))
                                        sns.heatmap(corr_matrix1, mask=mask, annot=True, fmt='.2f',
                                                  cmap='coolwarm', center=0, vmin=-1, vmax=1,
                                                  square=True, ax=ax, cbar_kws={"shrink": 0.8})
                                        
                                        # Apply font settings
                                        style_matplotlib_plot(ax, title='Assessment Correlations (Lower Triangle)')
                                        
                                        # Also style colorbar
                                        cbar = ax.collections[0].colorbar
                                        if cbar:
                                            cbar.ax.tick_params(labelsize=st.session_state.get('font_size', 16))
                                        
                                        st.pyplot(fig)
                                else:
                                    st.info("Not enough numeric columns for correlation analysis")
                            else:
                                st.info("Not enough assessment columns for correlation analysis")
                        
                        with col2:
                            st.markdown(f"**{name2} - Correlation Matrix**")
                            
                            # Get assessment columns for correlation
                            assessment_cols2 = get_assessment_columns(df2_filtered)
                            if len(assessment_cols2) > 1:
                                # Prepare numeric columns
                                numeric_cols2 = []
                                for col in assessment_cols2:  # Process all columns
                                    if col == df2_filtered.columns[3]:  # Column D
                                        if 'Final_Mark' in df2_filtered.columns:
                                            numeric_cols2.append('Final_Mark')
                                    else:
                                        df2_filtered[col] = pd.to_numeric(df2_filtered[col], errors='coerce')
                                        if df2_filtered[col].notna().sum() > 10:
                                            numeric_cols2.append(col)
                                
                                if len(numeric_cols2) > 1:
                                    corr_matrix2 = df2_filtered[numeric_cols2].corr()
                                    
                                    # Create mask for upper triangle
                                    mask = np.triu(np.ones_like(corr_matrix2, dtype=bool))
                                    
                                    if use_plotly:
                                        # Create triangular heatmap with plotly
                                        corr_masked = corr_matrix2.where(~mask)
                                        fig = px.imshow(
                                            corr_masked,
                                            color_continuous_scale='RdBu',
                                            zmin=-1, zmax=1,
                                            aspect='auto',
                                            text_auto='.2f'
                                        )
                                        # Apply font settings
                                        font_settings = get_plotly_font_settings()
                                        fig.update_layout(
                                            height=int(400 * st.session_state.plot_size_multiplier), 
                                            title="Lower Triangle Only",
                                            **font_settings
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        plot_mult = st.session_state.get('plot_size_multiplier', 1.2)
                                        fig, ax = plt.subplots(figsize=(8 * plot_mult, 6 * plot_mult))
                                        sns.heatmap(corr_matrix2, mask=mask, annot=True, fmt='.2f',
                                                  cmap='coolwarm', center=0, vmin=-1, vmax=1,
                                                  square=True, ax=ax, cbar_kws={"shrink": 0.8})
                                        
                                        # Apply font settings
                                        style_matplotlib_plot(ax, title='Assessment Correlations (Lower Triangle)')
                                        
                                        # Also style colorbar
                                        cbar = ax.collections[0].colorbar
                                        if cbar:
                                            cbar.ax.tick_params(labelsize=st.session_state.get('font_size', 16))
                                        
                                        st.pyplot(fig)
                                else:
                                    st.info("Not enough numeric columns for correlation analysis")
                            else:
                                st.info("Not enough assessment columns for correlation analysis")
    
    with tab4:
        st.markdown("## 🎯 Individual Student Performance")
        
        # Select file
        file_idx = st.selectbox(
            "Select File",
            range(len(file_names)),
            format_func=lambda x: file_names[x],
            key="individual_file"
        )
        
        df_individual = dataframes[file_idx]
        
        # Student search
        student_search = st.text_input(
            "🔍 Search Student ID",
            placeholder="Enter Student ID...",
            key="student_search"
        )
        
        if student_search:
            student_data = df_individual[df_individual['Student ID'] == student_search]
            
            if not student_data.empty:
                st.success(f"✅ Found student: {student_search}")
                st.markdown("### 📋 Student Details")
                
                if 'Final_Mark' in student_data.columns:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Final Mark", f"{student_data['Final_Mark'].iloc[0]:.0f}")
                    with col2:
                        if 'Grade' in student_data.columns:
                            st.metric("Grade", student_data['Grade'].iloc[0])
            else:
                st.error(f"❌ Student ID '{student_search}' not found")
        else:
            st.info("Enter a Student ID to view individual performance")
    
    with tab5:
        st.markdown("## 📥 Export Results")
        st.info("Export functionality - select data to export and download in your preferred format.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>Course Analysis Tool Pro v2.1 | Built with Streamlit</p>
    <p>© 2024 - Advanced Educational Analytics Platform</p>
</div>
""", unsafe_allow_html=True)
