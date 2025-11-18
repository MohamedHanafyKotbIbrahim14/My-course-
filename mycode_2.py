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

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
    }
    .highlight-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

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
    
    # Settings section
    st.markdown("## ⚙️ Settings")
    
    # Color theme selector
    color_theme = st.selectbox(
        "🎨 Color Theme",
        ["Default", "Dark", "Colorful", "Professional"],
        help="Choose visualization color theme"
    )
    
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

def create_grade_pie_chart(distribution, title, use_plotly=True):
    """Create a pie chart for grade distribution"""
    grades = []
    percentages = []
    colors = ['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336']  # HD to FL colors
    
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
        return fig
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(percentages, labels=grades, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title(title)
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
    
    # Sample data format
    st.markdown("---")
    st.markdown("## 📋 Expected Data Format")
    
    sample_data = pd.DataFrame({
        'Student ID': ['S001', 'S002', 'S003'],
        'Name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
        'Assignment 1': [85, 92, 78],
        'Final Mark/Grade': ['85 HD', '92 HD', '78 DN'],
        'Assignment 2': [88, 90, 82]
    })
    
    st.dataframe(sample_data, use_container_width=True)
    st.caption("Your CSV should have Student ID in Column A and Final Mark/Grade in Column D")
    
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🔄 Comparison", 
        "📈 Detailed Analysis",
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
            
            colors = ['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336']
            
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
                            marker_colors=colors[:len(grades)],
                            textposition='inside',
                            textinfo='percent+label'
                        ),
                        row=1, col=idx+1
                    )
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            cols = st.columns(len(dataframes))
            for idx, (df, name) in enumerate(zip(dataframes, file_names)):
                with cols[idx]:
                    if 'Grade' in df.columns:
                        dist = get_grade_distribution(df)
                        fig = create_grade_pie_chart(dist, name, use_plotly=False)
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
                    filter_option = st.radio(
                        "🔍 Analysis Scope",
                        ["🌍 ALL Students", "🎯 ONLY Common Students"],
                        horizontal=True,
                        key="filter_radio"
                    )
                    
                    show_common_only = (filter_option == "🎯 ONLY Common Students")
                    
                    if show_common_only and len(common_ids) > 0:
                        # Filter for common students
                        df1_filtered = df1[df1['Student ID'].isin(common_ids)]
                        df2_filtered = df2[df2['Student ID'].isin(common_ids)]
                        
                        # Scatter plot
                        st.markdown("### 📈 Correlation Analysis")
                        
                        plot_df1 = df1_filtered[['Student ID', col1_data]].copy()
                        plot_df1.columns = ['Student ID', 'Metric1']
                        
                        plot_df2 = df2_filtered[['Student ID', col2_data]].copy()
                        plot_df2.columns = ['Student ID', 'Metric2']
                        
                        plot_df = plot_df1.merge(plot_df2, on='Student ID', how='inner')
                        plot_df = plot_df.dropna()
                        
                        if len(plot_df) > 0:
                            if use_plotly:
                                fig = px.scatter(
                                    plot_df, 
                                    x='Metric1', 
                                    y='Metric2',
                                    hover_data=['Student ID'],
                                    title=f'{selected_col1} vs {selected_col2}',
                                    labels={'Metric1': f'{selected_col1} ({name1})', 
                                           'Metric2': f'{selected_col2} ({name2})'},
                                    trendline="ols",
                                    color_discrete_sequence=['#667eea']
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
                                
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.scatter(plot_df['Metric1'], plot_df['Metric2'], 
                                          c='#667eea', alpha=0.6, s=50)
                                
                                min_val = min(plot_df['Metric1'].min(), plot_df['Metric2'].min())
                                max_val = max(plot_df['Metric1'].max(), plot_df['Metric2'].max())
                                ax.plot([min_val, max_val], [min_val, max_val], 
                                       'k--', alpha=0.3, label='Equal Performance')
                                
                                ax.set_xlabel(f'{selected_col1} ({name1})')
                                ax.set_ylabel(f'{selected_col2} ({name2})')
                                ax.set_title(f'{selected_col1} vs {selected_col2}')
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
                        comparison_df['% Change'] = (comparison_df['Difference'] / comparison_df['Metric1'] * 100).round(1)
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
                            '% Change',
                            'Performance'
                        ]
                        
                        st.dataframe(
                            comparison_df.style.background_gradient(subset=['Difference', '% Change']),
                            use_container_width=True,
                            height=400
                        )
                        
                        # Summary statistics
                        st.markdown("### 📊 Summary Statistics")
                        
                        stat_cols = st.columns(3)
                        improved = (comparison_df['Difference'] > 5).sum()
                        declined = (comparison_df['Difference'] < -5).sum()
                        stable = ((comparison_df['Difference'] >= -5) & (comparison_df['Difference'] <= 5)).sum()
                        
                        stat_cols[0].metric("📈 Improved", improved, f"{improved/len(comparison_df)*100:.1f}%")
                        stat_cols[1].metric("📉 Declined", declined, f"{declined/len(comparison_df)*100:.1f}%")
                        stat_cols[2].metric("➡️ Stable", stable, f"{stable/len(comparison_df)*100:.1f}%")
    
    with tab3:
        st.markdown("## 📈 Detailed Analysis")
        
        # Select file for detailed analysis
        selected_file_idx = st.selectbox(
            "Select File for Detailed Analysis",
            range(len(file_names)),
            format_func=lambda x: file_names[x],
            key="detailed_file"
        )
        
        df_detail = dataframes[selected_file_idx]
        file_detail_name = file_names[selected_file_idx]
        
        st.markdown(f"### Analyzing: **{file_detail_name}**")
        
        # Statistical summary
        st.markdown("### 📊 Statistical Summary")
        
        if 'Final_Mark' in df_detail.columns:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                stats = df_detail['Final_Mark'].describe()
                stats_df = pd.DataFrame({
                    'Statistic': ['Count', 'Mean', 'Std Dev', 'Min', '25%', 'Median', '75%', 'Max'],
                    'Value': [
                        f"{stats['count']:.0f}",
                        f"{stats['mean']:.2f}",
                        f"{stats['std']:.2f}",
                        f"{stats['min']:.2f}",
                        f"{stats['25%']:.2f}",
                        f"{stats['50%']:.2f}",
                        f"{stats['75%']:.2f}",
                        f"{stats['max']:.2f}"
                    ]
                })
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
            
            with col2:
                # Distribution plot
                if use_plotly:
                    fig = px.histogram(
                        df_detail,
                        x='Final_Mark',
                        nbins=20,
                        title='Final Mark Distribution',
                        labels={'Final_Mark': 'Final Mark', 'count': 'Number of Students'},
                        color_discrete_sequence=['#667eea']
                    )
                    fig.add_vline(
                        x=df_detail['Final_Mark'].mean(),
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Mean: {df_detail['Final_Mark'].mean():.1f}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(df_detail['Final_Mark'].dropna(), bins=20, 
                           color='#667eea', alpha=0.7, edgecolor='black')
                    ax.axvline(df_detail['Final_Mark'].mean(), color='red', 
                             linestyle='--', linewidth=2, 
                             label=f'Mean: {df_detail["Final_Mark"].mean():.1f}')
                    ax.set_xlabel('Final Mark')
                    ax.set_ylabel('Number of Students')
                    ax.set_title('Final Mark Distribution')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
        
        # Assessment columns analysis
        assessment_cols = get_assessment_columns(df_detail)
        if len(assessment_cols) > 1:
            st.markdown("### 📊 Assessment Components Analysis")
            
            # Create correlation matrix
            numeric_cols = []
            for col in assessment_cols:
                if col == df_detail.columns[3]:  # Column D
                    numeric_cols.append('Final_Mark')
                else:
                    df_detail[col] = pd.to_numeric(df_detail[col], errors='coerce')
                    numeric_cols.append(col)
            
            # Filter out columns with too many NaN values
            valid_cols = []
            for col in numeric_cols:
                if df_detail[col].notna().sum() > 10:  # At least 10 valid values
                    valid_cols.append(col)
            
            if len(valid_cols) > 1:
                corr_matrix = df_detail[valid_cols].corr()
                
                # Correlation heatmap
                if use_plotly:
                    fig = px.imshow(
                        corr_matrix,
                        title="Assessment Correlation Matrix",
                        color_continuous_scale='RdBu',
                        aspect='auto',
                        text_auto='.2f'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr_matrix, annot=True, fmt='.2f', 
                              cmap='coolwarm', center=0, ax=ax)
                    ax.set_title('Assessment Correlation Matrix')
                    st.pyplot(fig)
    
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
                
                # Display student information
                st.markdown("### 📋 Student Details")
                
                # Get all assessments for this student
                assessment_cols = get_assessment_columns(df_individual)
                
                if assessment_cols:
                    # Create performance summary
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown("#### 📊 Performance Summary")
                        
                        if 'Final_Mark' in student_data.columns:
                            final_mark = student_data['Final_Mark'].iloc[0]
                            grade = student_data['Grade'].iloc[0] if 'Grade' in student_data.columns else 'N/A'
                            
                            st.metric("Final Mark", f"{final_mark:.0f}")
                            st.metric("Grade", grade)
                            
                            # Compare to class average
                            class_avg = df_individual['Final_Mark'].mean()
                            diff = final_mark - class_avg
                            st.metric("vs Class Average", f"{diff:+.1f}", 
                                     delta_color="normal" if diff >= 0 else "inverse")
                    
                    with col2:
                        st.markdown("#### 📈 Assessment Breakdown")
                        
                        # Create assessment scores chart
                        scores = []
                        labels = []
                        
                        for col in assessment_cols:
                            if col == df_individual.columns[3]:  # Column D
                                value = student_data['Final_Mark'].iloc[0]
                            else:
                                value = pd.to_numeric(student_data[col].iloc[0], errors='coerce')
                            
                            if pd.notna(value):
                                scores.append(value)
                                labels.append(col[:20])  # Truncate long names
                        
                        if scores:
                            if use_plotly:
                                fig = go.Figure(data=[
                                    go.Bar(x=labels, y=scores, marker_color='#667eea')
                                ])
                                fig.update_layout(
                                    title="Assessment Scores",
                                    xaxis_title="Assessment",
                                    yaxis_title="Score",
                                    showlegend=False
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.bar(labels, scores, color='#667eea')
                                ax.set_xlabel('Assessment')
                                ax.set_ylabel('Score')
                                ax.set_title('Assessment Scores')
                                plt.xticks(rotation=45, ha='right')
                                st.pyplot(fig)
                
                # Percentile ranking
                if 'Final_Mark' in student_data.columns:
                    st.markdown("### 📊 Class Ranking")
                    
                    final_mark = student_data['Final_Mark'].iloc[0]
                    percentile = (df_individual['Final_Mark'] <= final_mark).mean() * 100
                    
                    st.progress(percentile / 100)
                    st.markdown(f"**Percentile:** {percentile:.1f}% (Better than {percentile:.0f}% of the class)")
            else:
                st.error(f"❌ Student ID '{student_search}' not found")
        else:
            st.info("Enter a Student ID to view individual performance")
    
    with tab5:
        st.markdown("## 📥 Export Results")
        
        # Export options
        st.markdown("### Select Data to Export")
        
        export_options = st.multiselect(
            "Choose data to include in export",
            ["Raw Data", "Statistical Summary", "Grade Distribution", "Comparison Results"],
            default=["Statistical Summary", "Grade Distribution"]
        )
        
        # File format selection (already in sidebar)
        st.info(f"Export format selected: **{export_format}**")
        
        if st.button("🚀 Generate Export", type="primary"):
            with st.spinner("Generating export..."):
                # Create export data
                export_data = {}
                
                for idx, (df, name) in enumerate(zip(dataframes, file_names)):
                    if "Raw Data" in export_options:
                        export_data[f"{name}_raw"] = df
                    
                    if "Statistical Summary" in export_options and 'Final_Mark' in df.columns:
                        stats = df['Final_Mark'].describe()
                        export_data[f"{name}_stats"] = pd.DataFrame(stats).T
                    
                    if "Grade Distribution" in export_options and 'Grade' in df.columns:
                        dist = get_grade_distribution(df)
                        grade_df = pd.DataFrame([
                            {'Grade': grade, 'Count': data['count'], 'Percentage': f"{data['percentage']:.1f}%"}
                            for grade, data in dist.items()
                        ])
                        export_data[f"{name}_grades"] = grade_df
                
                # Create download based on format
                if export_format == "CSV":
                    # Create a zip file with multiple CSVs
                    st.markdown("### 📦 Download CSV Files")
                    for key, df in export_data.items():
                        csv = df.to_csv(index=False)
                        st.download_button(
                            f"📥 Download {key}.csv",
                            csv,
                            f"{key}.csv",
                            "text/csv",
                            key=f"download_{key}"
                        )
                
                elif export_format == "Excel":
                    # Create Excel file with multiple sheets
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        for key, df in export_data.items():
                            # Clean sheet name (Excel has limitations)
                            sheet_name = key[:31]  # Excel sheet name limit
                            df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    excel_data = output.getvalue()
                    
                    st.download_button(
                        "📥 Download Excel Report",
                        excel_data,
                        f"course_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                elif export_format == "HTML Report":
                    # Create HTML report
                    html_report = f"""
                    <html>
                    <head>
                        <title>Course Analysis Report</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 40px; }}
                            h1 {{ color: #667eea; }}
                            h2 {{ color: #764ba2; }}
                            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                            th {{ background-color: #667eea; color: white; }}
                            .metric {{ background: #f0f2f6; padding: 10px; border-radius: 5px; margin: 10px 0; }}
                        </style>
                    </head>
                    <body>
                        <h1>Course Analysis Report</h1>
                        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    """
                    
                    for key, df in export_data.items():
                        html_report += f"<h2>{key}</h2>"
                        html_report += df.to_html(index=False)
                    
                    html_report += """
                    </body>
                    </html>
                    """
                    
                    st.download_button(
                        "📥 Download HTML Report",
                        html_report,
                        f"course_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        "text/html"
                    )
                
                st.success("✅ Export generated successfully!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>Course Analysis Tool Pro v2.0 | Built with Streamlit</p>
    <p>© 2024 - Advanced Educational Analytics Platform</p>
</div>
""", unsafe_allow_html=True)
