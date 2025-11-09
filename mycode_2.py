import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Course Analysis Tool", layout="wide")

st.title("📊 Advanced Course Analysis & Comparison Tool")

# Sidebar
st.sidebar.header("📁 Upload CSV Files")

st.sidebar.markdown("""
**Instructions:**
1. Upload 2 CSV files using the file uploaders below
2. Select which assessment columns to compare (after upload)
3. Choose analysis filter (ALL vs BOTH)
""")

# File uploaders
uploaded_file1 = st.sidebar.file_uploader("Upload File 1 (CSV)", type=['csv'], key="uploader1")
uploaded_file2 = st.sidebar.file_uploader("Upload File 2 (CSV)", type=['csv'], key="uploader2")

st.sidebar.markdown("---")

def process_dataframe(df):
    """Process and clean the dataframe"""
    # Skip first row if it's header info
    if 'Student ID' in df.columns and len(df) > 0 and str(df.iloc[0, 0]).startswith('ACTL'):
        df = df.iloc[1:].reset_index(drop=True)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # COLUMN D (index 3): ALWAYS contains Final Mark + Grade
    final_col = df.iloc[:, 3]  # Column D
    df['Final_Mark'] = final_col.astype(str).str.extract('(\d+)').astype(float)
    df['Grade'] = final_col.astype(str).str.extract('([A-Z]{2,3})')[0]
    
    # Clean Student ID (should be in first columns)
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
    # Get the actual Column D header name (don't rename it)
    col_d_name = df.columns[3]  # Column D actual name
    
    # Start with Column D
    assessment_cols = [col_d_name]
    
    # Add columns from index 4 onwards (Column E+)
    for i in range(4, len(df.columns)):
        col_name = df.columns[i]
        if col_name not in ['Final_Mark', 'Grade', 'Student ID']:
            assessment_cols.append(col_name)
    
    return assessment_cols

# Main content
if not uploaded_file1 or not uploaded_file2:
    st.info("👈 **Please upload 2 CSV files** in the sidebar to begin analysis")
    st.markdown("""
    ### How to use this tool:
    1. Click on **"Browse files"** in the sidebar
    2. Select your first CSV file
    3. Select your second CSV file
    4. Choose metrics to compare
    5. View comprehensive analysis and visualizations
    """)
    st.stop()

# Get file names
file1 = uploaded_file1.name
file2 = uploaded_file2.name

st.sidebar.success(f"✅ Both files uploaded successfully!")

try:
    # Read and process both files
    df1_orig = pd.read_csv(uploaded_file1)
    df2_orig = pd.read_csv(uploaded_file2)
    
    df1_orig = process_dataframe(df1_orig)
    df2_orig = process_dataframe(df2_orig)
    
    # Get available assessment columns (from Column E onwards)
    assessment_cols_file1 = get_assessment_columns(df1_orig)
    assessment_cols_file2 = get_assessment_columns(df2_orig)
    
    # Show column selection in sidebar
    st.sidebar.success(f"✅ Both files uploaded successfully!")
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Select Columns to Compare")
    
    st.sidebar.markdown(f"**File 1:** {file1}")
    if len(assessment_cols_file1) > 0:
        selected_col1 = st.sidebar.selectbox(
            "Select column:",
            assessment_cols_file1,
            key="col1",
            help="Column D (final mark/grade) + all detected assessment columns"
        )
    else:
        st.sidebar.error("No assessment columns found in File 1!")
        st.stop()
    
    st.sidebar.markdown(f"**File 2:** {file2}")
    if len(assessment_cols_file2) > 0:
        selected_col2 = st.sidebar.selectbox(
            "Select column:",
            assessment_cols_file2,
            key="col2",
            help="Column D (final mark/grade) + all detected assessment columns"
        )
    else:
        st.sidebar.error("No assessment columns found in File 2!")
        st.stop()
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"""
    **Comparison Setup:**
    - X-axis: {selected_col1} (File 1)
    - Y-axis: {selected_col2} (File 2)
    """)
    
    # Get Column D names for both files
    col_d_name_file1 = df1_orig.columns[3]
    col_d_name_file2 = df2_orig.columns[3]
    
    # Map selected columns to actual column names
    # If Column D is selected, use the 'Final_Mark' column we created
    if selected_col1 == col_d_name_file1:
        col1_name = 'Final_Mark'
    else:
        col1_name = selected_col1
        # Convert to numeric if it's an assessment column
        df1_orig[selected_col1] = pd.to_numeric(df1_orig[selected_col1], errors='coerce')
    
    if selected_col2 == col_d_name_file2:
        col2_name = 'Final_Mark'
    else:
        col2_name = selected_col2
        # Convert to numeric if it's an assessment column
        df2_orig[selected_col2] = pd.to_numeric(df2_orig[selected_col2], errors='coerce')
    
    # Store display names for labels (use the actual column names)
    display_col1 = selected_col1
    display_col2 = selected_col2
    
    # Find common students
    common_ids = set(df1_orig['Student ID']) & set(df2_orig['Student ID'])
    
    st.header("📈 Analysis & Comparison")
    
    # --- STUDENT OVERVIEW ---
    st.subheader("👥 Student Overview")
    overview_col1, overview_col2, overview_col3 = st.columns(3)
    overview_col1.metric("📄 Students in File 1", len(df1_orig))
    overview_col2.metric("📄 Students in File 2", len(df2_orig))
    overview_col3.metric("🔗 Students in BOTH", len(common_ids))
    
    # --- DISTRIBUTION VISUALIZATION FOR ALL STUDENTS ---
    st.subheader("📊 Distribution Analysis - ALL Students")
    st.info("These distributions show ALL students from each file, regardless of the filter setting")
    
    # Create comprehensive distribution plots - 2x2 layout but use only 3 plots
    fig_dist, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # File 1 - Metric 1 Distribution
    data1_full = df1_orig[col1_name].dropna()
    data1_common = df1_orig[df1_orig['Student ID'].isin(common_ids)][col1_name].dropna()
    
    axes[0, 0].hist(data1_full, bins=20, color='lightblue', alpha=0.7, edgecolor='black', label='All Students')
    axes[0, 0].hist(data1_common, bins=20, color='red', alpha=0.6, edgecolor='black', label='Students in BOTH files')
    axes[0, 0].axvline(data1_full.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean (All): {data1_full.mean():.1f}')
    axes[0, 0].axvline(data1_common.mean(), color='darkred', linestyle='--', linewidth=2, label=f'Mean (Common): {data1_common.mean():.1f}')
    axes[0, 0].set_xlabel(display_col1, fontweight='bold')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'{file1} - {display_col1}\nTotal: {len(data1_full)} students | Common: {len(data1_common)} students', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # File 2 - Metric 2 Distribution
    data2_full = df2_orig[col2_name].dropna()
    data2_common = df2_orig[df2_orig['Student ID'].isin(common_ids)][col2_name].dropna()
    
    axes[0, 1].hist(data2_full, bins=20, color='lightcoral', alpha=0.7, edgecolor='black', label='All Students')
    axes[0, 1].hist(data2_common, bins=20, color='red', alpha=0.6, edgecolor='black', label='Students in BOTH files')
    axes[0, 1].axvline(data2_full.mean(), color='coral', linestyle='--', linewidth=2, label=f'Mean (All): {data2_full.mean():.1f}')
    axes[0, 1].axvline(data2_common.mean(), color='darkred', linestyle='--', linewidth=2, label=f'Mean (Common): {data2_common.mean():.1f}')
    axes[0, 1].set_xlabel(display_col2, fontweight='bold')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'{file2} - {display_col2}\nTotal: {len(data2_full)} students | Common: {len(data2_common)} students', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Box Plot Comparison - merge bottom two plots into one
    # Remove axes[1,1] and expand axes[1,0] to span both columns
    fig_dist.delaxes(axes[1, 1])
    axes[1, 0] = plt.subplot(2, 1, 2)  # Make bottom plot span full width
    
    box_data = [data1_full, data1_common, data2_full, data2_common]
    box_labels = [f'File 1\nAll\n(n={len(data1_full)})', 
                 f'File 1\nCommon\n(n={len(data1_common)})',
                 f'File 2\nAll\n(n={len(data2_full)})', 
                 f'File 2\nCommon\n(n={len(data2_common)})']
    box_colors = ['lightblue', 'red', 'lightcoral', 'darkred']
    
    bp = axes[1, 0].boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1, 0].set_ylabel('Score', fontweight='bold')
    axes[1, 0].set_title('Box Plot Comparison\n(Shows distribution quartiles)', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    st.pyplot(fig_dist)
    
    st.markdown("---")
    
    # --- FILTER SELECTION (Right after box plot) ---
    st.subheader("🔍 Analysis Filter")
    st.markdown("**⚠️ This affects ALL analysis below:**")
    filter_option = st.radio(
        "Analyze which students?",
        ["🌍 ALL Students", "🎯 ONLY Students in BOTH Files"],
        index=0,
        help="Changes statistics, plots, distributions, and tables",
        horizontal=True
    )
    show_only_common = (filter_option == "🎯 ONLY Students in BOTH Files")
    
    if show_only_common:
        st.error("🎯 ONLY COMMON - Shows scatter plot & detailed comparisons")
    else:
        st.success("🌍 ALL - Shows distributions & statistics only")
    
    st.markdown("---")
    
    # --- APPLY FILTER ---
    if show_only_common:
        df1 = df1_orig[df1_orig['Student ID'].isin(common_ids)].copy()
        df2 = df2_orig[df2_orig['Student ID'].isin(common_ids)].copy()
        st.success(f"🎯 Analyzing **COMMON** students only: **{len(common_ids)}** students")
    else:
        df1 = df1_orig.copy()
        df2 = df2_orig.copy()
        st.info(f"🌍 Analyzing **ALL** students | File 1: **{len(df1)}** | File 2: **{len(df2)}** | Common: **{len(common_ids)}**")
    
    st.markdown("---")
    
    # --- STATISTICAL SUMMARY ---
    filter_text = "COMMON" if show_only_common else "ALL"
    st.subheader(f"📊 Statistical Summary - {filter_text} Students")
    st.caption(f"Comparing: **{display_col1}** (File 1) vs **{display_col2}** (File 2)")
    if show_only_common:
        st.warning(f"⚠️ Statistics below are for COMMON students only ({len(common_ids)} students)")
    else:
        st.info(f"ℹ️ Statistics below are for ALL students in each file")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown(f"### 📄 {file1}")
        st.caption(f"Students analyzed: **{len(df1)}**")
        
        # Stats for selected column
        stats1_m1 = df1[col1_name].describe()
        st.markdown(f"**{display_col1}**")
        stats_df1_m1 = pd.DataFrame({
            'Statistic': ['Count', 'Mean', 'Std Dev', 'Min', '25%', 'Median', '75%', 'Max'],
            'Value': [
                f"{stats1_m1['count']:.0f}",
                f"{stats1_m1['mean']:.2f}",
                f"{stats1_m1['std']:.2f}",
                f"{stats1_m1['min']:.2f}",
                f"{stats1_m1['25%']:.2f}",
                f"{stats1_m1['50%']:.2f}",
                f"{stats1_m1['75%']:.2f}",
                f"{stats1_m1['max']:.2f}"
            ]
        })
        st.dataframe(stats_df1_m1, use_container_width=True, hide_index=True)
        
        # Grade distribution (always available since Column D is Final Mark)
        st.markdown("**📊 Grade Distribution (Final Mark)**")
        dist1 = get_grade_distribution(df1)
        grade_df1 = pd.DataFrame([
            {'Grade': grade, 'Count': data['count'], 'Percentage': f"{data['percentage']:.1f}%"}
            for grade, data in dist1.items()
        ])
        st.dataframe(grade_df1, use_container_width=True, hide_index=True)
    
    with col_right:
        st.markdown(f"### 📄 {file2}")
        st.caption(f"Students analyzed: **{len(df2)}**")
        
        # Stats for selected column
        stats2_m2 = df2[col2_name].describe()
        st.markdown(f"**{display_col2}**")
        stats_df2_m2 = pd.DataFrame({
            'Statistic': ['Count', 'Mean', 'Std Dev', 'Min', '25%', 'Median', '75%', 'Max'],
            'Value': [
                f"{stats2_m2['count']:.0f}",
                f"{stats2_m2['mean']:.2f}",
                f"{stats2_m2['std']:.2f}",
                f"{stats2_m2['min']:.2f}",
                f"{stats2_m2['25%']:.2f}",
                f"{stats2_m2['50%']:.2f}",
                f"{stats2_m2['75%']:.2f}",
                f"{stats2_m2['max']:.2f}"
            ]
        })
        st.dataframe(stats_df2_m2, use_container_width=True, hide_index=True)
        
        # Grade distribution (always available since Column D is Final Mark)
        st.markdown("**📊 Grade Distribution (Final Mark)**")
        dist2 = get_grade_distribution(df2)
        grade_df2 = pd.DataFrame([
            {'Grade': grade, 'Count': data['count'], 'Percentage': f"{data['percentage']:.1f}%"}
            for grade, data in dist2.items()
        ])
        st.dataframe(grade_df2, use_container_width=True, hide_index=True)
    
    # --- SCATTER PLOT (Only for COMMON students) ---
    if show_only_common:
        st.subheader(f"📈 Scatter Plot: {display_col1} (File 1) vs {display_col2} (File 2)")
        
        # Count students who CAN be plotted
        plot_df1_temp = df1_orig[['Student ID', col1_name]].dropna()
        plot_df2_temp = df2_orig[['Student ID', col2_name]].dropna()
        plottable_students = set(plot_df1_temp['Student ID']) & set(plot_df2_temp['Student ID'])
        
        # Show info about plottable students
        col_info1, col_info2, col_info3 = st.columns(3)
        col_info1.metric("Students in File 1", len(df1_orig))
        col_info2.metric("Students in File 2", len(df2_orig))
        col_info3.metric("Students in BOTH (Plottable)", len(plottable_students), 
                        help="Only these students can appear on scatter plot")
        
        # Explanation box
        if file1 != file2:
            if len(plottable_students) < max(len(df1_orig), len(df2_orig)):
                st.warning(f"""
                ⚠️ **Scatter Plot Limitation:** Only **{len(plottable_students)}** students can be plotted (those in BOTH files).
                
                Why? A scatter plot needs BOTH x and y coordinates:
                - **{len(df1_orig) - len(plottable_students)}** students ONLY in File 1 → Cannot plot (no Y coordinate)
                - **{len(df2_orig) - len(plottable_students)}** students ONLY in File 2 → Cannot plot (no X coordinate)
                
                💡 The filter (ALL vs ONLY) affects tables and histograms, but scatter plot always shows only students with both coordinates.
                """)
            else:
                st.success("✅ All students appear in both files, so all can be plotted!")
        
        # Prepare data for plotting
        plot_df1 = df1[['Student ID', col1_name]].copy()
        plot_df1.columns = ['Student ID', 'Metric1']
        
        plot_df2 = df2[['Student ID', col2_name]].copy()
        plot_df2.columns = ['Student ID', 'Metric2']
        
        plot_df = plot_df1.merge(plot_df2, on='Student ID', how='inner')  # Use inner to get only students in both
        plot_df = plot_df.dropna()  # Remove any rows with NaN values
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # All plotted students are in both files (mathematically required for scatter plot)
        if len(plot_df) > 0:
            ax.scatter(plot_df['Metric1'], plot_df['Metric2'], 
                      c='red', alpha=0.7, s=100, label=f'Students in BOTH files (n={len(plot_df)})')
        else:
            st.error("No students found in both files with valid data for both metrics!")
        
        # Add diagonal line
        min_val = 0
        max_val = max(plot_df['Metric1'].max(), plot_df['Metric2'].max()) if len(plot_df) > 0 else 100
        ax.plot([min_val, max_val], [min_val, max_val], 
               'k--', alpha=0.3, linewidth=1, label='Equal Performance Line')
        
        ax.set_xlabel(f'{display_col1} - {file1}', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{display_col2} - {file2}', fontsize=12, fontweight='bold')
        
        title = f'{display_col1} vs {display_col2} Comparison\n(All points = Students appearing in BOTH files)'
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    # --- DETAILED COMPARISON TABLE (Only for COMMON students) ---
    if show_only_common and len(common_ids) > 0:
        st.subheader("📋 Students in Both Files - Detailed Comparison")
        
        # Get common students data
        common_df1 = df1_orig[df1_orig['Student ID'].isin(common_ids)][['Student ID', col1_name]].copy()
        common_df2 = df2_orig[df2_orig['Student ID'].isin(common_ids)][['Student ID', col2_name]].copy()
        
        comparison_df = common_df1.merge(common_df2, on='Student ID')
        comparison_df.columns = ['Student ID', f'{display_col1} (File1)', f'{display_col2} (File2)']
        comparison_df['Difference'] = comparison_df[f'{display_col2} (File2)'] - comparison_df[f'{display_col1} (File1)']
        comparison_df['Change'] = comparison_df['Difference'].apply(
            lambda x: '📈 Higher' if x > 5 else ('📉 Lower' if x < -5 else '➡️ Similar')
        )
        comparison_df = comparison_df.sort_values('Difference', ascending=False).reset_index(drop=True)
        
        st.dataframe(comparison_df, use_container_width=True)
        
        # Summary stats for common students
        st.markdown("**Performance Comparison Summary:**")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Higher in File 2", f"{(comparison_df['Difference'] > 5).sum()} students")
        col_b.metric("Lower in File 2", f"{(comparison_df['Difference'] < -5).sum()} students")
        col_c.metric("Similar", f"{(abs(comparison_df['Difference']) <= 5).sum()} students")
        
        # Download option
        csv = comparison_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Comparison Table",
            csv,
            "comparison_table.csv",
            "text/csv"
        )
    
    # --- DISTRIBUTION COMPARISON - Filtered View (Only for COMMON students, at the END) ---
    if show_only_common:
        st.subheader("📊 Distribution Comparison - Filtered View")
        st.caption("Showing distributions for COMMON students only")
        
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Distribution for File 1
        data1 = df1[col1_name].dropna()
        ax1.hist(data1, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(data1.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {data1.mean():.1f}')
        ax1.set_xlabel(display_col1)
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'{file1}\n{display_col1}\n(n={len(data1)})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Distribution for File 2
        data2 = df2[col2_name].dropna()
        ax2.hist(data2, bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
        ax2.axvline(data2.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {data2.mean():.1f}')
        ax2.set_xlabel(display_col2)
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'{file2}\n{display_col2}\n(n={len(data2)})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig2)
    
except Exception as e:
    st.error(f"Error processing files: {str(e)}")
    st.exception(e)
