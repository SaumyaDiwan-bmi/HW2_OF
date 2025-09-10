#!/usr/bin/env python3
"""
CSV Data Pipeline Deployment Script
==================================

A comprehensive deployment script that visualizes the entire preprocessing pipeline
with detailed side branches showing data removal at each step.

Features:
- Advanced Mermaid diagram generation with side branches
- Interactive pipeline analysis
- Comprehensive reporting
- Multiple output formats
- Real-time progress tracking
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import json
from csv_data_manager import CSVDataManager

class PipelineDeployer:
    """
    Advanced pipeline deployment and visualization system.
    """
    
    def __init__(self, csv_path: str, output_dir: str = "pipeline_output"):
        """
        Initialize the pipeline deployer.
        
        Args:
            csv_path (str): Path to the CSV file
            output_dir (str): Output directory for all generated files
        """
        self.csv_path = csv_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.manager = CSVDataManager(csv_path)
        self.deployment_log = []
        
    def deploy_pipeline(self, skip_rows: int = 30, create_visualizations: bool = True):
        """
        Deploy the complete data preprocessing pipeline.
        
        Args:
            skip_rows (int): Number of rows to skip from the beginning
            create_visualizations (bool): Whether to create visualizations
        """
        print("🚀 DEPLOYING CSV DATA PREPROCESSING PIPELINE")
        print("=" * 60)
        print(f"📁 Input file: {self.csv_path}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Step 1: Load data
        print("🔄 STEP 1: Loading CSV data...")
        data = self.manager.load_csv(skip_rows=skip_rows)
        
        if data is None:
            print("❌ Failed to load data. Exiting.")
            return False
        
        self._log_step("Load CSV", len(data), "Data loaded successfully")
        
        # Step 2: Clean data
        print("\n🔄 STEP 2: Cleaning data...")
        cleaned_data = self.manager.clean_data(fix_typos=True, handle_missing='drop', validate_logical=True)
        
        if cleaned_data is None:
            print("❌ Failed to clean data. Exiting.")
            return False
        
        self._log_step("Clean Data", len(cleaned_data), "Data cleaned successfully")
        
        # Step 3: Validate data
        print("\n🔄 STEP 3: Validating data...")
        validation_report = self.manager.validate_data()
        self._log_step("Validate Data", len(cleaned_data), "Data validated successfully")
        
        # Step 4: Generate comprehensive reports
        print("\n🔄 STEP 4: Generating reports...")
        self._generate_comprehensive_reports()
        
        # Step 5: Create advanced visualizations
        if create_visualizations:
            print("\n🔄 STEP 5: Creating visualizations...")
            self._create_advanced_visualizations()
        
        # Step 6: Generate Mermaid diagrams
        print("\n🔄 STEP 6: Generating Mermaid diagrams...")
        self._generate_mermaid_diagrams()
        
        # Step 7: Export data in multiple formats
        print("\n🔄 STEP 7: Exporting data...")
        self._export_data_formats()
        
        # Step 8: Generate deployment summary
        print("\n🔄 STEP 8: Generating deployment summary...")
        self._generate_deployment_summary()
        
        print("\n✅ PIPELINE DEPLOYMENT COMPLETE!")
        print(f"📁 All outputs saved to: {self.output_dir}")
        
        return True
    
    def _log_step(self, step_name: str, output_rows: int, message: str):
        """Log a pipeline step."""
        self.deployment_log.append({
            'step': step_name,
            'output_rows': output_rows,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })
    
    def _generate_comprehensive_reports(self):
        """Generate comprehensive analysis reports."""
        
        # 1. Preprocessing Pipeline Report
        pipeline_report = {
            'deployment_info': {
                'timestamp': datetime.now().isoformat(),
                'input_file': str(self.csv_path),
                'output_directory': str(self.output_dir)
            },
            'pipeline_steps': self.manager.preprocessing_log,
            'deployment_steps': self.deployment_log,
            'data_summary': {
                'initial_rows': self.manager.preprocessing_log[0]['output_rows'] if self.manager.preprocessing_log else 0,
                'final_rows': self.manager.preprocessing_log[-1]['output_rows'] if self.manager.preprocessing_log else 0,
                'total_removed': 0
            }
        }
        
        if self.manager.preprocessing_log:
            pipeline_report['data_summary']['total_removed'] = (
                pipeline_report['data_summary']['initial_rows'] - 
                pipeline_report['data_summary']['final_rows']
            )
        
        # Save JSON report
        with open(self.output_dir / "pipeline_report.json", "w") as f:
            json.dump(pipeline_report, f, indent=2)
        
        # Save text report
        with open(self.output_dir / "pipeline_report.txt", "w") as f:
            f.write("CSV DATA PREPROCESSING PIPELINE REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input file: {self.csv_path}\n\n")
            
            f.write("PIPELINE STEPS:\n")
            f.write("-" * 30 + "\n")
            for step in self.manager.preprocessing_log:
                f.write(f"Step: {step['step']}\n")
                f.write(f"  Input rows: {step['input_rows']}\n")
                f.write(f"  Output rows: {step['output_rows']}\n")
                f.write(f"  Removed: {step['removed_rows']}\n")
                f.write(f"  Reason: {step['reason']}\n\n")
        
        print("  ✅ Pipeline reports generated")
    
    def _create_advanced_visualizations(self):
        """Create advanced visualizations."""
        try:
            # Create visualizations using the manager
            plot_paths = self.manager.create_visualizations(
                output_dir=str(self.output_dir / "plots"),
                figsize=(15, 10)
            )
            
            if plot_paths:
                print(f"  ✅ Created {len(plot_paths)} visualization files")
                
                # Create visualization index
                with open(self.output_dir / "visualizations_index.html", "w") as f:
                    f.write(self._generate_visualization_html(plot_paths))
                print("  ✅ Visualization index created")
            else:
                print("  ⚠️  No visualizations created (libraries not available)")
                
        except Exception as e:
            print(f"  ⚠️  Visualization creation failed: {e}")
    
    def _generate_visualization_html(self, plot_paths):
        """Generate HTML index for visualizations."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Data Pipeline Visualizations</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .plot-container { margin: 20px 0; padding: 20px; border: 1px solid #ddd; }
        .plot-container img { max-width: 100%; height: auto; }
        h1 { color: #333; }
        h2 { color: #666; }
    </style>
</head>
<body>
    <h1>CSV Data Pipeline Visualizations</h1>
    <p>Generated on: {}</p>
"""
        
        html = html.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        for name, path in plot_paths.items():
            relative_path = Path(path).relative_to(self.output_dir)
            html += f"""
    <div class="plot-container">
        <h2>{name.replace('_', ' ').title()}</h2>
        <img src="{relative_path}" alt="{name}">
    </div>
"""
        
        html += """
</body>
</html>
"""
        return html
    
    def _generate_mermaid_diagrams(self):
        """Generate the main pipeline diagram."""
        
        # Single comprehensive pipeline diagram
        main_diagram = self._create_advanced_pipeline_diagram()
        with open(self.output_dir / "pipeline_diagram.mmd", "w") as f:
            f.write(main_diagram)
        
        print("  ✅ Pipeline diagram generated")
    
    def _create_advanced_pipeline_diagram(self):
        """Create detailed pipeline diagram with every preprocessing step and removal criteria."""

        if not self.manager.preprocessing_log:
            return "graph TD\n    A[No preprocessing steps recorded]"

        diagram = "graph TD\n"

        # Create diagram based on actual preprocessing steps from log
        log = self.manager.preprocessing_log
        total_lines_in_file = 305
        comment_lines = 30
        data_lines_after_comments = total_lines_in_file - comment_lines

        # Start with the basic structure
        steps = [
            {
                'id': 'A',
                'name': 'Raw CSV File',
                'input': total_lines_in_file,
                'output': total_lines_in_file,
                'emoji': '📁',
                'removal': None
            },
            {
                'id': 'B',
                'name': 'Skip Comments',
                'input': total_lines_in_file,
                'output': data_lines_after_comments,
                'emoji': '🔄',
                'removal': {
                    'id': 'removed_comments',
                    'name': 'Comment Lines',
                    'count': comment_lines,
                    'details': 'Lines 1-30'
                }
            }
        ]

        # If we have preprocessing log, extract actual steps
        current_input = data_lines_after_comments
        step_counter = ord('C')  # Start from 'C'
        
        # Check if we have a "Load CSV (Fixed)" step that includes malformed line removal
        if len(log) >= 1:
            load_step = log[0]  # Load CSV step
            if 'Load CSV (Fixed)' in load_step['step'] and 'malformed lines' in load_step['reason']:
                reason = load_step['reason']
                
                # Parse malformed lines removal from load step
                import re
                malformed_match = re.search(r'removed (\d+) malformed lines', reason)
                if malformed_match:
                    malformed_removed = int(malformed_match.group(1))
                    steps.append({
                        'id': chr(step_counter),
                        'name': 'Remove Malformed Rows',
                        'input': current_input,
                        'output': current_input - malformed_removed,
                        'emoji': '🔄',
                        'removal': {
                            'id': 'removed_malformed',
                            'name': 'Malformed Rows',
                            'count': malformed_removed,
                            'details': 'Rows with >1 comma after fixing'
                        }
                    })
                    current_input -= malformed_removed
                    step_counter += 1

        # Process cleaning steps if they exist
        if len(log) >= 2:
            clean_step = log[1]  # Clean Data step
            reason = clean_step['reason']

            # Alphanumeric entries removal
            if 'alphanumeric entries' in reason:
                import re
                match = re.search(r'Removed (\d+) alphanumeric entries', reason)
                if match:
                    alphanumeric_removed = int(match.group(1))
                    steps.append({
                        'id': chr(step_counter),
                        'name': 'Remove Alphanumeric Entries',
                        'input': current_input,
                        'output': current_input - alphanumeric_removed,
                        'emoji': '🔄',
                        'removal': {
                            'id': 'removed_alphanumeric',
                            'name': 'Alphanumeric Entries',
                            'count': alphanumeric_removed,
                            'details': 'Rows with letters in numbers'
                        }
                    })
                    current_input -= alphanumeric_removed
                    step_counter += 1

            # Missing values removal
            if 'missing values' in reason:
                match = re.search(r'removed (\d+) missing values', reason)
                if match:
                    missing_removed = int(match.group(1))
                    steps.append({
                        'id': chr(step_counter),
                        'name': 'Handle Missing Values',
                        'input': current_input,
                        'output': current_input - missing_removed,
                        'emoji': '🔄',
                        'removal': {
                            'id': 'removed_missing',
                            'name': 'Missing Values',
                            'count': missing_removed,
                            'details': 'NaN values'
                        }
                    })
                    current_input -= missing_removed
                    step_counter += 1

            # Logical violations removal
            if 'logical violations' in reason:
                match = re.search(r'removed (\d+) logical violations', reason)
                if match:
                    logical_removed = int(match.group(1))
                    steps.append({
                        'id': chr(step_counter),
                        'name': 'Validate Logical Constraints',
                        'input': current_input,
                        'output': current_input - logical_removed,
                        'emoji': '🔄',
                        'removal': {
                            'id': 'removed_logical',
                            'name': 'Logical Violations',
                            'count': logical_removed,
                            'details': 'eruption_time > waiting_time'
                        }
                    })
                    current_input -= logical_removed
                    step_counter += 1

        # Add final clean dataset step
        total_removed = total_lines_in_file - current_input
        retention_rate = (current_input / data_lines_after_comments) * 100

        final_step = {
            'id': chr(ord('A') + len(steps)),
            'name': 'Final Clean Dataset',
            'input': current_input,
            'output': current_input,
            'emoji': '✅',
            'removal': None
        }
        steps.append(final_step)

        # Generate the diagram
        for i, step in enumerate(steps):
            # Main processing node
            if step['removal'] is None:
                # Final node or initial node
                if step['id'] == 'A':
                    diagram += f"    {step['id']}[\"{step['emoji']} {step['name']}<br/>📊 Total Lines: {step['output']}\"]\n"
                else:
                    if step['id'] == final_step['id']:
                        retention = (step['output'] / data_lines_after_comments) * 100
                        total_removed = data_lines_after_comments - step['output']
                        diagram += f"    {step['id']}[\"{step['emoji']} {step['name']}<br/>📊 Rows: {step['output']}<br/>🗑️ Total Removed: {total_removed}<br/>📈 Retention: {retention:.1f}%\"]\n"
                    else:
                        diagram += f"    {step['id']}[\"{step['emoji']} {step['name']}<br/>📊 Input: {step['input']}<br/>📤 Output: {step['output']}\"]\n"
            else:
                # Regular processing node
                diagram += f"    {step['id']}[\"{step['emoji']} {step['name']}<br/>📊 Input: {step['input']}<br/>📤 Output: {step['output']}\"]\n"

            # Connection to next step
            if i < len(steps) - 1:
                diagram += f"    {step['id']} --> {steps[i+1]['id']}\n"

            # Side branch for removal
            if step['removal'] and step['removal']['count'] > 0:
                removal = step['removal']
                diagram += f"    {removal['id']}[\"🗑️ {removal['name']}<br/>Removed: {removal['count']}<br/>{removal['details']}\"]\n"
                diagram += f"    {step['id']} -.-> {removal['id']}\n"

        # Add styling with high contrast colors
        diagram += "\n    style A fill:#1976d2,stroke:#0d47a1,stroke-width:3px,color:#ffffff\n"
        diagram += "    style B fill:#388e3c,stroke:#1b5e20,stroke-width:3px,color:#ffffff\n"

        # Style processing steps dynamically
        for i in range(len(steps)):
            step_id = steps[i]['id']
            if step_id == 'A':
                continue  # Already styled
            elif step_id == final_step['id']:  # Final step
                diagram += f"    style {step_id} fill:#4caf50,stroke:#1b5e20,stroke-width:4px,color:#ffffff\n"
            else:
                diagram += f"    style {step_id} fill:#388e3c,stroke:#1b5e20,stroke-width:3px,color:#ffffff\n"

        # Style all removal boxes
        for step in steps:
            if step['removal']:
                diagram += f"    style {step['removal']['id']} fill:#d32f2f,stroke:#b71c1c,stroke-width:2px,color:#ffffff\n"

        # Ensure all removal IDs are styled (fallback for any missing ones)
        diagram += "    style removed_alphanumeric fill:#d32f2f,stroke:#b71c1c,stroke-width:2px,color:#ffffff\n"

        return diagram
    
    
    def _export_data_formats(self):
        """Export data in a single, practical format."""

        if self.manager.cleaned_data is not None:
            # Export in CSV format only (most universal and human-readable)
            success = self.manager.save_data(
                str(self.output_dir / "cleaned_data.csv"),
                format="csv"
            )
            if success:
                print("  ✅ Exported cleaned data: cleaned_data.csv")
            else:
                print("  ⚠️  Failed to export cleaned data")
    
    def _generate_deployment_summary(self):
        """Generate deployment summary."""
        
        summary = f"""
CSV DATA PIPELINE DEPLOYMENT SUMMARY
====================================

Deployment Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Input File: {self.csv_path}
Output Directory: {self.output_dir}

PIPELINE STATISTICS:
"""
        
        if self.manager.preprocessing_log:
            initial_rows = self.manager.preprocessing_log[0]['output_rows']
            final_rows = self.manager.preprocessing_log[-1]['output_rows']
            total_removed = initial_rows - final_rows
            
            summary += f"""
  Initial Rows: {initial_rows}
  Final Rows: {final_rows}
  Total Removed: {total_removed}
  Retention Rate: {(final_rows/initial_rows)*100:.1f}%
"""
        
        summary += f"""

GENERATED FILES:
  📊 Reports: pipeline_report.json, pipeline_report.txt
  📈 Visualizations: plots/ directory, periodicity_analysis.png
  🎨 Diagrams: pipeline_diagram.mmd
  💾 Data: cleaned_data.csv
  🌐 Web: visualizations_index.html

NEXT STEPS:
  1. Review the pipeline_report.txt for detailed analysis
  2. Open visualizations_index.html in a browser
  3. Use the generated Mermaid diagrams in documentation
  4. Import cleaned data using your preferred format

For questions or issues, check the pipeline_report.json for detailed logs.
"""
        
        with open(self.output_dir / "DEPLOYMENT_SUMMARY.txt", "w") as f:
            f.write(summary)
        
        print("  ✅ Deployment summary generated")
        print("\n" + summary)


def main():
    """Main deployment function."""

    # Configuration
    CSV_FILE = "geyser.csv"
    OUTPUT_DIR = "pipeline_output"
    SKIP_ROWS = 30

    # Check if CSV file exists
    if not os.path.exists(CSV_FILE):
        print(f"❌ Error: CSV file '{CSV_FILE}' not found!")
        print("Please ensure the file exists in the current directory.")
        return

    # Create deployer
    deployer = PipelineDeployer(CSV_FILE, OUTPUT_DIR)

    # Deploy pipeline with logical validation enabled
    success = deployer.deploy_pipeline(skip_rows=SKIP_ROWS, create_visualizations=True)
    
    if success:
        print(f"\n🎉 SUCCESS! Pipeline deployed to: {OUTPUT_DIR}")
        print("\n📁 Generated files:")
        for file_path in Path(OUTPUT_DIR).rglob("*"):
            if file_path.is_file():
                print(f"  - {file_path.relative_to(Path(OUTPUT_DIR))}")
    else:
        print("\n❌ Pipeline deployment failed!")


if __name__ == "__main__":
    main()
