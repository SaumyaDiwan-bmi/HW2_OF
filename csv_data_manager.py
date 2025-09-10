#!/usr/bin/env python3
"""
Efficient CSV Data Access and Storage Manager
============================================

This module provides comprehensive tools for accessing, cleaning, and storing CSV data
efficiently using pandas and other optimized libraries.

Features:
- Fast CSV reading with pandas
- Data cleaning and validation
- Multiple storage formats (CSV, Pickle, Parquet, HDF5)
- Memory-efficient processing
- Data quality reporting
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    print("Note: Visualization libraries not available. Install matplotlib and seaborn for plotting features.")

class CSVDataManager:
    """
    A comprehensive class for efficient CSV data management.
    """
    
    def __init__(self, csv_path: str):
        """
        Initialize the CSV Data Manager.
        
        Args:
            csv_path (str): Path to the CSV file
        """
        self.csv_path = Path(csv_path)
        self.data = None
        self.cleaned_data = None
        self.data_info = {}
        self.preprocessing_log = []  # Track preprocessing steps
        
    def load_csv(self, 
                 skip_rows: int = 0, 
                 comment_char: str = '#',
                 **kwargs) -> pd.DataFrame:
        """
        Load CSV data efficiently with pandas.
        
        Args:
            skip_rows (int): Number of rows to skip from the beginning
            comment_char (str): Character that indicates comments
            **kwargs: Additional arguments for pd.read_csv()
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            # First, try to load with error handling
            read_params = {
                'skiprows': skip_rows,
                'comment': comment_char,
                'na_values': ['', 'NA', 'N/A', 'null', 'NULL'],
                'keep_default_na': True,
                'encoding': 'utf-8',
                'on_bad_lines': 'skip',  # Skip problematic lines
                **kwargs
            }
            
            self.data = pd.read_csv(self.csv_path, **read_params)
            self._generate_data_info()
            
            # Log preprocessing step
            self.preprocessing_log.append({
                'step': 'Load CSV',
                'input_rows': 'Unknown (raw file)',
                'output_rows': len(self.data),
                'removed_rows': 'N/A',
                'reason': f'Skipped {skip_rows} comment lines, handled comments with "{comment_char}"'
            })
            
            print(f"✅ Successfully loaded {len(self.data)} rows and {len(self.data.columns)} columns")
            return self.data
            
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            print("🔄 Attempting to fix data issues...")
            
            # Try to fix common CSV issues
            try:
                # Read the file line by line and fix common issues
                fixed_lines = []
                total_data_lines = 0
                malformed_lines_removed = 0
                double_comma_lines_fixed = 0
                
                with open(self.csv_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for i, line in enumerate(lines):
                        line_num = i + 1
                        
                        # Skip comment lines and header
                        if line.strip().startswith(comment_char) or line_num <= skip_rows:
                            continue
                        
                        # Skip empty lines
                        if not line.strip():
                            continue
                        
                        # Count this as a data line
                        total_data_lines += 1
                        
                        # Fix common issues
                        fixed_line = line.strip()
                        
                        # Fix double commas (e.g., "3,333,68" -> "3.333,68")
                        if fixed_line.count(',') > 1:
                            # Find the pattern of number,number,number and fix it
                            import re
                            # Pattern to match: digit,digit,digit (where middle is likely a decimal)
                            pattern = r'(\d+),(\d+),(\d+)'
                            match = re.search(pattern, fixed_line)
                            if match:
                                # Convert to decimal format
                                fixed_line = re.sub(pattern, r'\1.\2,\3', fixed_line)
                                double_comma_lines_fixed += 1
                        
                        # Only add lines that have exactly one comma (for 2 columns)
                        if fixed_line.count(',') == 1:
                            fixed_lines.append(fixed_line)
                        else:
                            malformed_lines_removed += 1
                            print(f"⚠️  Skipping malformed line {line_num}: {line.strip()}")
                
                # Create DataFrame from fixed lines
                if fixed_lines:
                    import io
                    csv_content = '\n'.join(fixed_lines)
                    self.data = pd.read_csv(io.StringIO(csv_content), 
                                          na_values=['', 'NA', 'N/A', 'null', 'NULL'],
                                          keep_default_na=True)
                    self._generate_data_info()
                    
                    # Log preprocessing step
                    self.preprocessing_log.append({
                        'step': 'Load CSV (Fixed)',
                        'input_rows': total_data_lines + skip_rows,  # Total lines processed
                        'output_rows': len(self.data),
                        'removed_rows': malformed_lines_removed,
                        'reason': f'Fixed {double_comma_lines_fixed} double comma lines, removed {malformed_lines_removed} malformed lines with >1 comma, skipped {skip_rows} comment lines'
                    })
                    
                    print(f"✅ Successfully loaded {len(self.data)} rows after fixing data issues")
                    return self.data
                else:
                    print("❌ No valid data lines found after fixing")
                    return None
                    
            except Exception as fix_error:
                print(f"❌ Failed to fix CSV issues: {fix_error}")
                return None
    
    def clean_data(self,
                   fix_typos: bool = True,
                   handle_missing: str = 'drop',
                   validate_logical: bool = True) -> pd.DataFrame:
        """
        Clean the loaded data by fixing common issues.

        Args:
            fix_typos (bool): Whether to remove alphanumeric entries in numeric data
            handle_missing (str): How to handle missing values ('drop', 'fill', 'keep')
            validate_logical (bool): Whether to validate logical constraints (e.g., eruption_time <= waiting_time)

        Returns:
            pd.DataFrame: Cleaned data
        """
        if self.data is None:
            print("❌ No data loaded. Call load_csv() first.")
            return None
            
        input_rows = len(self.data)
        self.cleaned_data = self.data.copy()
        
        # Remove alphanumeric entries in numeric columns
        alphanumeric_removed = 0
        if fix_typos:
            before_count = len(self.cleaned_data)
            for col in self.cleaned_data.columns:
                if self.cleaned_data[col].dtype == 'object':
                    # Remove rows with alphanumeric entries (letters mixed with numbers)
                    self.cleaned_data[col] = self.cleaned_data[col].astype(str)
                    # Remove rows that contain letters in numeric columns
                    alphanumeric_mask = self.cleaned_data[col].str.contains('[a-zA-Z]', regex=True)
                    if alphanumeric_mask.sum() > 0:
                        print(f"🔄 Removing {alphanumeric_mask.sum()} rows with alphanumeric entries in column '{col}'")
                        self.cleaned_data = self.cleaned_data[~alphanumeric_mask]

            alphanumeric_removed = before_count - len(self.cleaned_data)

        # Ensure numeric columns are properly typed after alphanumeric removal
        for col in self.cleaned_data.columns:
            if col in ['eruptions', 'waiting']:  # Known numeric columns
                try:
                    self.cleaned_data[col] = pd.to_numeric(self.cleaned_data[col], errors='coerce')
                except:
                    pass

        # Handle missing values
        missing_before = self.cleaned_data.isnull().sum().sum()
        if handle_missing == 'drop':
            self.cleaned_data = self.cleaned_data.dropna()
        elif handle_missing == 'fill':
            numeric_cols = self.cleaned_data.select_dtypes(include=[np.number]).columns
            self.cleaned_data[numeric_cols] = self.cleaned_data[numeric_cols].fillna(
                self.cleaned_data[numeric_cols].mean()
            )
        
        missing_after = self.cleaned_data.isnull().sum().sum()
        missing_removed = missing_before - missing_after
        
        # Validate logical constraints
        logical_removed = 0
        if validate_logical:
            before_count = len(self.cleaned_data)
            # Check if eruption time is longer than waiting time (which doesn't make sense)
            if 'eruptions' in self.cleaned_data.columns and 'waiting' in self.cleaned_data.columns:
                logical_violations = self.cleaned_data['eruptions'] > self.cleaned_data['waiting']
                logical_removed = logical_violations.sum()
                if logical_removed > 0:
                    self.cleaned_data = self.cleaned_data[~logical_violations]
                    print(f"🔄 Removed {logical_removed} rows with logical violations (eruption_time > waiting_time)")

        # Log cleaning step
        self.preprocessing_log.append({
            'step': 'Clean Data',
            'input_rows': input_rows,
            'output_rows': len(self.cleaned_data),
            'removed_rows': input_rows - len(self.cleaned_data),
            'reason': f'Removed {alphanumeric_removed} alphanumeric entries, removed {missing_removed} missing values, removed {logical_removed} logical violations'
        })

        print(f"✅ Data cleaned: {len(self.cleaned_data)} rows remaining")
        return self.cleaned_data
    
    def validate_data(self) -> Dict[str, Any]:
        """
        Validate data quality and return validation report.
        
        Returns:
            Dict[str, Any]: Validation report
        """
        if self.cleaned_data is None:
            print("❌ No cleaned data available. Call clean_data() first.")
            return {}
        
        report = {
            'total_rows': len(self.cleaned_data),
            'total_columns': len(self.cleaned_data.columns),
            'missing_values': self.cleaned_data.isnull().sum().to_dict(),
            'data_types': self.cleaned_data.dtypes.to_dict(),
            'numeric_columns': self.cleaned_data.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': self.cleaned_data.select_dtypes(include=['object']).columns.tolist(),
        }
        
        # Add statistics for numeric columns
        numeric_cols = self.cleaned_data.select_dtypes(include=[np.number])
        if not numeric_cols.empty:
            report['numeric_stats'] = numeric_cols.describe().to_dict()
        
        # Check for outliers (using IQR method)
        outlier_info = {}
        for col in numeric_cols.columns:
            Q1 = numeric_cols[col].quantile(0.25)
            Q3 = numeric_cols[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((numeric_cols[col] < lower_bound) | (numeric_cols[col] > upper_bound)).sum()
            outlier_info[col] = {
                'count': outliers,
                'percentage': (outliers / len(numeric_cols)) * 100
            }
        report['outliers'] = outlier_info
        
        self.data_info = report
        return report
    
    def save_data(self, 
                  output_path: str,
                  format: str = 'csv',
                  compression: Optional[str] = None) -> bool:
        """
        Save data in various efficient formats.
        
        Args:
            output_path (str): Output file path
            format (str): Output format ('csv', 'pickle', 'parquet', 'hdf5', 'feather')
            compression (str): Compression type ('gzip', 'bz2', 'xz', 'lz4', 'zstd')
            
        Returns:
            bool: Success status
        """
        data_to_save = self.cleaned_data if self.cleaned_data is not None else self.data
        
        if data_to_save is None:
            print("❌ No data to save. Load data first.")
            return False
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format.lower() == 'csv':
                data_to_save.to_csv(output_path, index=False, compression=compression)
            elif format.lower() == 'pickle':
                data_to_save.to_pickle(output_path)
            elif format.lower() == 'parquet':
                data_to_save.to_parquet(output_path, compression=compression)
            elif format.lower() == 'hdf5':
                data_to_save.to_hdf(output_path, key='data', mode='w', format='table')
            elif format.lower() == 'feather':
                data_to_save.to_feather(output_path, compression=compression)
            else:
                print(f"❌ Unsupported format: {format}")
                return False
            
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            print(f"✅ Data saved to {output_path} ({file_size:.2f} MB)")
            return True
            
        except Exception as e:
            print(f"❌ Error saving data: {e}")
            return False
    
    def load_saved_data(self, file_path: str, format: str = 'auto') -> pd.DataFrame:
        """
        Load previously saved data efficiently.
        
        Args:
            file_path (str): Path to saved file
            format (str): File format ('auto', 'csv', 'pickle', 'parquet', 'hdf5', 'feather')
            
        Returns:
            pd.DataFrame: Loaded data
        """
        file_path = Path(file_path)
        
        if format == 'auto':
            suffix = file_path.suffix.lower()
            if suffix == '.csv':
                format = 'csv'
            elif suffix == '.pkl':
                format = 'pickle'
            elif suffix == '.parquet':
                format = 'parquet'
            elif suffix in ['.h5', '.hdf5']:
                format = 'hdf5'
            elif suffix == '.feather':
                format = 'feather'
            else:
                print(f"❌ Cannot auto-detect format for {suffix}")
                return None
        
        try:
            if format.lower() == 'csv':
                return pd.read_csv(file_path)
            elif format.lower() == 'pickle':
                return pd.read_pickle(file_path)
            elif format.lower() == 'parquet':
                return pd.read_parquet(file_path)
            elif format.lower() == 'hdf5':
                return pd.read_hdf(file_path, key='data')
            elif format.lower() == 'feather':
                return pd.read_feather(file_path)
            else:
                print(f"❌ Unsupported format: {format}")
                return None
                
        except Exception as e:
            print(f"❌ Error loading saved data: {e}")
            return None
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get memory usage information for the loaded data.
        
        Returns:
            Dict[str, Any]: Memory usage statistics
        """
        if self.data is None:
            return {}
        
        memory_usage = self.data.memory_usage(deep=True)
        total_memory = memory_usage.sum() / (1024 * 1024)  # MB
        
        return {
            'total_memory_mb': total_memory,
            'per_column_mb': (memory_usage / (1024 * 1024)).to_dict(),
            'dtypes': self.data.dtypes.to_dict(),
            'shape': self.data.shape
        }
    
    def _generate_data_info(self):
        """Generate basic data information."""
        if self.data is not None:
            self.data_info = {
                'shape': self.data.shape,
                'columns': list(self.data.columns),
                'dtypes': self.data.dtypes.to_dict(),
                'memory_usage': self.get_memory_usage()
            }
    
    def create_periodicity_visualization(self,
                                       output_dir: str = "plots",
                                       figsize: Tuple[int, int] = (8, 0.8)) -> str:
        """
        Create a linear band visualization showing eruption cycle proportions.

        This visualization shows:
        1. Each sample as a horizontal band representing one eruption cycle
        2. Each band is split into eruption time (red) and wait time (blue) segments
        3. The proportions are scaled so that eruption + wait = 100% for each cycle
        4. Bands are positioned vertically based on their original magnitude (total time)
        5. This reveals the proportional relationship between eruption and wait times

        The key insight: Each cycle is normalized to 100%, showing that:
        - Eruption % and Wait % are inversely related within each cycle
        - The total magnitude varies between cycles
        - Patterns emerge in the proportion distributions

        Args:
            output_dir (str): Directory to save the plot
            figsize (Tuple[int, int]): Figure size

        Returns:
            str: Path to the saved visualization
        """
        if not HAS_VISUALIZATION:
            print("❌ Visualization libraries not available. Install matplotlib and seaborn.")
            return ""

        if self.cleaned_data is None:
            print("❌ No cleaned data available. Call clean_data() first.")
            return ""

        if 'eruptions' not in self.cleaned_data.columns or 'waiting' not in self.cleaned_data.columns:
            print("❌ Required columns 'eruptions' and 'waiting' not found.")
            return ""

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Create figure with standard projection and high DPI for crisp text
        fig, ax = plt.subplots(figsize=figsize, dpi=150)

        # Prepare data
        eruptions = self.cleaned_data['eruptions']
        waiting = self.cleaned_data['waiting']

        # Step 1: Calculate combined time and scale each sample to 100%
        combined_time = eruptions + waiting
        eruption_percentage = (eruptions / combined_time) * 100
        wait_percentage = (waiting / combined_time) * 100

        # Step 2: Sort by original magnitude (combined_time) for y-positioning
        sorted_indices = np.argsort(combined_time)
        sorted_combined = combined_time.iloc[sorted_indices]
        sorted_eruption_pct = eruption_percentage.iloc[sorted_indices]
        sorted_wait_pct = wait_percentage.iloc[sorted_indices]

        # Step 3: Create the linear band visualization
        band_height = 0.8  # Height of each band
        
        for i in range(len(sorted_combined)):
            # Y position as continuous index (no gaps)
            y_pos = i
            
            # Draw eruption time band (red, starts at 0)
            eruption_width = sorted_eruption_pct.iloc[i]
            ax.barh(y_pos, eruption_width, height=band_height, 
                   left=0, color='#e74c3c', alpha=0.8, 
                   label='Eruption Time' if i == 0 else "")
            
            # Draw wait time band (blue, starts after eruption)
            wait_width = sorted_wait_pct.iloc[i]
            ax.barh(y_pos, wait_width, height=band_height, 
                   left=eruption_width, color='#3498db', alpha=0.8,
                   label='Wait Time' if i == 0 else "")
            

        # Customize the plot
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, len(sorted_combined) - 0.5)
        
        ax.set_xlabel('Cycle Percentage (%)', fontsize=18, fontweight='bold')
        ax.set_ylabel('Magnitude of (eruption time + wait time) --->', fontsize=18, fontweight='bold')
        ax.set_title('Geyser Eruption Cycle Proportions',
                    fontsize=20, fontweight='bold', pad=15)

        # Add percentage markers on x-axis including average
        avg_eruption_pct = eruption_percentage.mean()
        tick_positions = [0, 25, 50, 75, 100]
        tick_labels = ['0', '25', '50', '75', '100']
        
        # Insert average value in appropriate position
        if avg_eruption_pct not in tick_positions:
            tick_positions.append(avg_eruption_pct)
            tick_labels.append(f'{avg_eruption_pct:.1f}')
            # Sort ticks and labels together
            sorted_pairs = sorted(zip(tick_positions, tick_labels), key=lambda x: x[0])
            tick_positions, tick_labels = zip(*sorted_pairs)
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=15)
        
        # Remove y-axis tick marks and labels
        ax.set_yticks([])
        ax.tick_params(axis='y', which='both', left=False, right=False)

        # Add minimal grid for better readability
        ax.grid(True, axis='x', alpha=0.3, linestyle='-')

        # Add vertical line for average eruption percentage
        ax.axvline(x=avg_eruption_pct, color='black', linestyle='--', linewidth=2, alpha=0.8,
                  label=f'Avg Eruption End: {avg_eruption_pct:.1f}%')

        # Add compact legend
        ax.legend(loc='upper right', framealpha=0.9, fontsize=15)


        plt.tight_layout()

        # Save the plot
        periodicity_path = output_dir / "periodicity_analysis.png"
        plt.savefig(periodicity_path, dpi=300)
        plt.close()

        print(f"✅ Periodicity visualization saved: {periodicity_path}")
        
        # Also create the circular ring version
        ring_path = self.create_ring_periodicity_visualization(output_dir, figsize)
        
        return str(periodicity_path)
    
    def create_ring_periodicity_visualization(self,
                                            output_dir: str = "plots",
                                            figsize: Tuple[int, int] = (6, 6)) -> str:
        """
        Create a circular ring version of the periodicity visualization.
        
        This wraps the linear bands around a ring where:
        - Each ring represents one eruption cycle
        - Ring radius corresponds to magnitude (inner = small, outer = large)
        - Angular position shows eruption % vs wait % around the circle
        - Colors remain red for eruption, blue for wait time
        """
        if not HAS_VISUALIZATION:
            print("❌ Visualization libraries not available. Install matplotlib and seaborn.")
            return ""

        if self.cleaned_data is None:
            print("❌ No cleaned data available. Call clean_data() first.")
            return ""

        if 'eruptions' not in self.cleaned_data.columns or 'waiting' not in self.cleaned_data.columns:
            print("❌ Required columns 'eruptions' and 'waiting' not found.")
            return ""

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Create figure with polar projection
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'), dpi=150)

        # Prepare data (same as linear version)
        eruptions = self.cleaned_data['eruptions']
        waiting = self.cleaned_data['waiting']
        combined_time = eruptions + waiting
        eruption_percentage = (eruptions / combined_time) * 100
        wait_percentage = (waiting / combined_time) * 100

        # Sort by magnitude for ring positioning (descending order - largest magnitude innermost)
        sorted_indices = np.argsort(combined_time)[::-1]  # Reverse for descending order
        sorted_combined = combined_time.iloc[sorted_indices]
        sorted_eruption_pct = eruption_percentage.iloc[sorted_indices]
        sorted_wait_pct = wait_percentage.iloc[sorted_indices]

        # Calculate ring parameters
        min_radius = 0.5  # Inner radius
        max_radius = 3.0  # Outer radius
        ring_width = (max_radius - min_radius) / len(sorted_combined)

        for i in range(len(sorted_combined)):
            # Ring position: inner rings = smaller magnitude, outer rings = larger magnitude
            inner_radius = min_radius + i * ring_width
            outer_radius = inner_radius + ring_width * 0.8  # Small gap between rings
            
            # Angular positions (convert percentages to radians)
            eruption_angle = np.radians(sorted_eruption_pct.iloc[i] * 360 / 100)
            
            # Create eruption arc (red) - from 0 to eruption_angle
            theta_eruption = np.linspace(0, eruption_angle, 50)
            r_inner_eruption = np.full_like(theta_eruption, inner_radius)
            r_outer_eruption = np.full_like(theta_eruption, outer_radius)
            
            ax.fill_between(theta_eruption, r_inner_eruption, r_outer_eruption,
                           color='#e74c3c', alpha=0.8)
            
            # Create wait arc (blue) - from eruption_angle to 2π
            theta_wait = np.linspace(eruption_angle, 2*np.pi, 50)
            r_inner_wait = np.full_like(theta_wait, inner_radius)
            r_outer_wait = np.full_like(theta_wait, outer_radius)
            
            ax.fill_between(theta_wait, r_inner_wait, r_outer_wait,
                           color='#3498db', alpha=0.8)

        # Add average eruption percentage as a radial line
        avg_eruption_pct = eruption_percentage.mean()
        avg_angle = np.radians(avg_eruption_pct * 360 / 100)
        ax.plot([avg_angle, avg_angle], [min_radius, max_radius], 
                color='black', linestyle='--', linewidth=3, alpha=0.9)
        
        # Add text labels directly on the rings for clarity (all horizontal)
        # Place "Eruption" label in the red section
        eruption_label_angle = avg_angle / 2  # Middle of eruption section
        eruption_label_radius = (min_radius + max_radius) / 2
        ax.text(eruption_label_angle, eruption_label_radius, 'ERUPTION', 
                rotation=0,  # Horizontal text
                ha='center', va='center', fontsize=16, fontweight='bold', 
                color='white')
        
        # Place "Wait" label in the blue section  
        wait_label_angle = avg_angle + (2*np.pi - avg_angle) / 2  # Middle of wait section
        wait_label_radius = (min_radius + max_radius) / 2
        ax.text(wait_label_angle, wait_label_radius, 'WAIT', 
                rotation=0,  # Horizontal text
                ha='center', va='center', fontsize=16, fontweight='bold', 
                color='white')
        
        # Add average eruption percentage label near the average line
        avg_label_radius = max_radius * 1.1
        ax.text(avg_angle, avg_label_radius, f'Avg: {avg_eruption_pct:.1f}%', 
                rotation=0,  # Horizontal text
                ha='center', va='center', fontsize=14, fontweight='bold', 
                color='black', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

        # Customize the polar plot
        ax.set_ylim(0, max_radius)
        ax.set_title('Geyser Eruption Cycle Ring Visualization',
                    fontsize=20, fontweight='bold', pad=20)

        # Set angular ticks (percentage labels)
        ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
        ax.set_xticklabels(['0%', '25%', '50%', '75%'], fontsize=15)
        
        # Remove radial tick labels (magnitude not as important in ring view)
        ax.set_rticks([])

        plt.tight_layout()

        # Save the ring plot
        ring_path = output_dir / "periodicity_ring_analysis.png"
        plt.savefig(ring_path, dpi=300)
        plt.close()

        print(f"✅ Ring periodicity visualization saved: {ring_path}")
        return str(ring_path)

    def create_visualizations(self,
                            output_dir: str = "plots",
                            figsize: Tuple[int, int] = (4, 2)) -> Dict[str, str]:
        """
        Create comprehensive visualizations of the data.
        
        Args:
            output_dir (str): Directory to save plots
            figsize (Tuple[int, int]): Figure size for plots
            
        Returns:
            Dict[str, str]: Dictionary mapping plot names to file paths
        """
        if not HAS_VISUALIZATION:
            print("❌ Visualization libraries not available. Install matplotlib and seaborn.")
            return {}
        
        if self.cleaned_data is None:
            print("❌ No cleaned data available. Call clean_data() first.")
            return {}
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        plot_paths = {}
        numeric_cols = self.cleaned_data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            print("❌ No numeric columns found for visualization.")
            return {}
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Distribution plots
        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(2, len(numeric_cols), figsize=figsize)
            if len(numeric_cols) == 1:
                axes = axes.reshape(-1, 1)
            
            for i, col in enumerate(numeric_cols):
                # Histogram
                axes[0, i].hist(self.cleaned_data[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
                axes[0, i].set_title(f'Distribution of {col}')
                axes[0, i].set_xlabel(col)
                axes[0, i].set_ylabel('Frequency')
                
                # Box plot
                axes[1, i].boxplot(self.cleaned_data[col].dropna())
                axes[1, i].set_title(f'Box Plot of {col}')
                axes[1, i].set_ylabel(col)
            
            plt.tight_layout()
            dist_path = output_dir / "distributions.png"
            plt.savefig(dist_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths['distributions'] = str(dist_path)
        
        # 2. Correlation heatmap
        if len(numeric_cols) > 1:
            plt.figure(figsize=figsize)
            correlation_matrix = self.cleaned_data[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.2f')
            plt.title('Correlation Matrix')
            corr_path = output_dir / "correlation_heatmap.png"
            plt.savefig(corr_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths['correlation'] = str(corr_path)
        
        # 3. Scatter plot matrix (if 2+ numeric columns)
        if len(numeric_cols) >= 2:
            if len(numeric_cols) == 2:
                plt.figure(figsize=figsize)
                plt.scatter(self.cleaned_data[numeric_cols[0]], 
                           self.cleaned_data[numeric_cols[1]], 
                           alpha=0.6, s=50)
                plt.xlabel(numeric_cols[0])
                plt.ylabel(numeric_cols[1])
                plt.title(f'{numeric_cols[0]} vs {numeric_cols[1]}')
                scatter_path = output_dir / "scatter_plot.png"
                plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
                plt.close()
                plot_paths['scatter'] = str(scatter_path)
            else:
                # Pair plot for multiple variables
                pair_plot = sns.pairplot(self.cleaned_data[numeric_cols], 
                                       diag_kind='hist', plot_kws={'alpha': 0.6})
                pair_path = output_dir / "pair_plot.png"
                pair_plot.savefig(pair_path, dpi=300, bbox_inches='tight')
                plt.close()
                plot_paths['pair_plot'] = str(pair_path)
        
        # 4. Missing data visualization
        if self.cleaned_data.isnull().any().any():
            plt.figure(figsize=figsize)
            sns.heatmap(self.cleaned_data.isnull(), cbar=True, yticklabels=False)
            plt.title('Missing Data Pattern')
            plt.tight_layout()
            missing_path = output_dir / "missing_data.png"
            plt.savefig(missing_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths['missing_data'] = str(missing_path)
        
        # 5. Time series plot (if applicable)
        if len(numeric_cols) >= 1:
            plt.figure(figsize=figsize)
            for col in numeric_cols:
                plt.plot(self.cleaned_data[col].values, label=col, alpha=0.7)
            plt.title('Data Trends')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            trend_path = output_dir / "trends.png"
            plt.savefig(trend_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths['trends'] = str(trend_path)

        # Create periodicity analysis
        periodicity_path = self.create_periodicity_visualization(output_dir=output_dir, figsize=figsize)
        if periodicity_path:
            plot_paths['periodicity'] = periodicity_path

        print(f"✅ Visualizations saved to {output_dir}")
        for name, path in plot_paths.items():
            print(f"  - {name}: {path}")

        return plot_paths
    
    def print_summary(self):
        """Print a comprehensive data summary."""
        if self.data is None:
            print("❌ No data loaded.")
            return
        
        print("\n" + "="*50)
        print("📊 DATA SUMMARY")
        print("="*50)
        print(f"Shape: {self.data.shape[0]} rows × {self.data.shape[1]} columns")
        print(f"Memory usage: {self.get_memory_usage()['total_memory_mb']:.2f} MB")
        print("\nColumns:")
        for col in self.data.columns:
            dtype = self.data[col].dtype
            null_count = self.data[col].isnull().sum()
            print(f"  - {col}: {dtype} ({null_count} nulls)")
        
        if not self.data.empty:
            print("\nFirst 5 rows:")
            print(self.data.head())
        
        if self.data_info:
            print(f"\nValidation report available: {len(self.data_info)} metrics")
    
    def show_preprocessing_pipeline(self):
        """
        Display a detailed preprocessing pipeline showing what data was removed at each step.
        """
        if not self.preprocessing_log:
            print("❌ No preprocessing steps recorded. Load and clean data first.")
            return
        
        print("\n" + "="*80)
        print("🔄 DATA PREPROCESSING PIPELINE")
        print("="*80)
        
        # Create a table-like display
        print(f"{'Step':<20} {'Input Rows':<12} {'Output Rows':<12} {'Removed':<10} {'Reason'}")
        print("-" * 80)
        
        for i, step in enumerate(self.preprocessing_log):
            step_name = step['step']
            input_rows = step['input_rows']
            output_rows = step['output_rows']
            removed = step['removed_rows']
            reason = step['reason']
            
            print(f"{step_name:<20} {str(input_rows):<12} {str(output_rows):<12} {str(removed):<10} {reason}")
        
        print("-" * 80)
        
        # Summary statistics
        if len(self.preprocessing_log) > 0:
            initial_rows = self.preprocessing_log[0]['output_rows']
            final_rows = self.preprocessing_log[-1]['output_rows']
            total_removed = initial_rows - final_rows
            
            print(f"\n📊 SUMMARY:")
            print(f"  Initial data rows: {initial_rows}")
            print(f"  Final data rows: {final_rows}")
            print(f"  Total rows removed: {total_removed}")
            print(f"  Data retention rate: {(final_rows/initial_rows)*100:.1f}%")
        
        # Data quality issues found
        print(f"\n🔍 DATA QUALITY ISSUES IDENTIFIED:")
        issues = []
        
        for step in self.preprocessing_log:
            if 'typos' in step['reason']:
                issues.append("• Typographical errors (l→1, O→0, etc.)")
            if 'missing values' in step['reason']:
                issues.append("• Missing values")
            if 'duplicates' in step['reason']:
                issues.append("• Duplicate rows")
            if 'malformed' in step['reason']:
                issues.append("• Malformed CSV lines (double commas)")
            if 'comment' in step['reason']:
                issues.append("• Comment lines")
        
        # Remove duplicates and display
        unique_issues = list(set(issues))
        for issue in unique_issues:
            print(f"  {issue}")
        
        if not unique_issues:
            print("  • No significant data quality issues found")
    
    def create_pipeline_mermaid(self) -> str:
        """
        Generate Mermaid diagram code for the preprocessing pipeline.
        
        Returns:
            str: Mermaid diagram code
        """
        if not self.preprocessing_log:
            return "No preprocessing steps recorded."
        
        mermaid = "graph TD\n"
        
        # Add nodes for each step
        for i, step in enumerate(self.preprocessing_log):
            step_name = step['step'].replace(' ', '_').replace('(', '').replace(')', '')
            input_rows = step['input_rows']
            output_rows = step['output_rows']
            removed = step['removed_rows']
            
            if i == 0:
                node_id = "A"
                mermaid += f"    {node_id}[\"{step_name}<br/>Input: {input_rows}<br/>Output: {output_rows}\"]\n"
            else:
                node_id = f"step{i}"
                prev_node = f"step{i-1}" if i > 1 else "A"
                mermaid += f"    {node_id}[\"{step_name}<br/>Input: {input_rows}<br/>Output: {output_rows}<br/>Removed: {removed}\"]\n"
                mermaid += f"    {prev_node} --> {node_id}\n"
        
        # Add styling
        mermaid += "\n    style A fill:#ffcccc\n"
        for i in range(1, len(self.preprocessing_log)):
            mermaid += f"    style step{i} fill:#ccffcc\n"
        
        return mermaid


def main():
    """
    Example usage of the CSVDataManager.
    """
    # Initialize the manager
    csv_path = "/Users/lauhityareddy/Repos/BMI500/HW2_OF/geyser.csv"
    manager = CSVDataManager(csv_path)
    
    # Load the data (skip comment lines)
    print("🔄 Loading CSV data...")
    data = manager.load_csv(skip_rows=30)  # Skip the comment lines
    
    if data is not None:
        # Print summary
        manager.print_summary()
        
        # Clean the data
        print("\n🔄 Cleaning data...")
        cleaned_data = manager.clean_data()
        
        # Validate data quality
        print("\n🔄 Validating data...")
        validation_report = manager.validate_data()
        
        # Show preprocessing pipeline
        print("\n🔄 PREPROCESSING PIPELINE ANALYSIS")
        manager.show_preprocessing_pipeline()
        
        # Print validation results
        print("\n📋 VALIDATION REPORT")
        print("-" * 30)
        print(f"Total rows: {validation_report.get('total_rows', 'N/A')}")
        print(f"Missing values: {validation_report.get('missing_values', {})}")
        
        if 'outliers' in validation_report:
            print("\nOutliers detected:")
            for col, info in validation_report['outliers'].items():
                print(f"  {col}: {info['count']} ({info['percentage']:.1f}%)")
        
        # Generate Mermaid diagram
        print("\n🎨 MERMAID DIAGRAM CODE:")
        print("-" * 30)
        mermaid_code = manager.create_pipeline_mermaid()
        print(mermaid_code)
        
        # Save in multiple formats for comparison
        print("\n💾 Saving data in multiple formats...")
        
        # Original CSV (cleaned)
        manager.save_data("geyser_cleaned.csv", format="csv")
        
        # Pickle (fastest for Python)
        manager.save_data("geyser_cleaned.pkl", format="pickle")
        
        # Parquet (efficient for analytics)
        manager.save_data("geyser_cleaned.parquet", format="parquet")
        
        # Feather (fast I/O)
        manager.save_data("geyser_cleaned.feather", format="feather")
        
        print("\n✅ Data processing complete!")
        print("\nRecommended usage:")
        print("- Use .pkl for Python-only workflows (fastest)")
        print("- Use .parquet for analytics and cross-platform compatibility")
        print("- Use .feather for fast I/O operations")
        print("- Use .csv for human readability and compatibility")


if __name__ == "__main__":
    main()
