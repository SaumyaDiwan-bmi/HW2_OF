#COMMENTS added in by Saumya using CHATGPT for LLM based code review for class purposes 
#prompts used: What are the strengths of the code in terms of structure? , What are possible fallbacks that could be added into the code? , in what cases could the plotting funcntions fail?, In what cases could file loading fail?, Do all file paths exist or have fall backs? 

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
        # ⚠️ Possible fallback: check if path is valid before assigning.
        # If file is missing, corrupted, or wrong format → fail early with user-friendly error.
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
        # ⚠️ Possible fallback: try multiple encodings (utf-8, latin-1) and auto-detect delimiters
        data = self.manager.load_csv(skip_rows=skip_rows)
        
        if data is None:
            print("❌ Failed to load data. Exiting.")
            # Suggestion: instead of exiting, could offer to re-run with different settings
            return False
        
        self._log_step("Load CSV", len(data), "Data loaded successfully")
        
        # Step 2: Clean data
        print("\n🔄 STEP 2: Cleaning data...")
        cleaned_data = self.manager.clean_data(fix_typos=True, handle_missing='drop', validate_logical=True)
        # ⚠️ Possible fallback: before dropping rows/columns, keep a backup copy so raw data isn’t lost
        # Also allow "soft cleaning" (replace invalids with NaN) if dropping is too destructive.
        
        if cleaned_data is None:
            print("❌ Failed to clean data. Exiting.")
            return False
        
        self._log_step("Clean Data", len(cleaned_data), "Data cleaned successfully")
        
        # Step 3: Validate data
        print("\n🔄 STEP 3: Validating data...")
        validation_report = self.manager.validate_data()
        # ⚠️ Possible fallback: If validation fails, warn user but continue to next step.
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
        
        # ⚠️ Possible fallback: if preprocessing_log is empty, generate a minimal report instead of failing
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
        
        # Save JSON and text reports
        with open(self.output_dir / "pipeline_report.json", "w") as f:
            json.dump(pipeline_report, f, indent=2)
        
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
            plot_paths = self.manager.create_visualizations(
                output_dir=str(self.output_dir / "plots"),
                figsize=(15, 10)
            )
            
            if plot_paths:
                print(f"  ✅ Created {len(plot_paths)} visualization files")
                with open(self.output_dir / "visualizations_index.html", "w") as f:
                    f.write(self._generate_visualization_html(plot_paths))
                print("  ✅ Visualization index created")
            else:
                print("  ⚠️  No visualizations created (libraries not available)")
                # ⚠️ Fallback: print summary stats instead of failing silently
        except Exception as e:
            print(f"  ⚠️  Visualization creation failed: {e}")
            # ⚠️ Fallback: pipeline continues without plots
    
    def _generate_mermaid_diagrams(self):
        """Generate the main pipeline diagram."""
        # ⚠️ Possible fallback: if diagram fails, write a minimal “start → end” flow to file
        main_diagram = self._create_advanced_pipeline_diagram()
        with open(self.output_dir / "pipeline_diagram.mmd", "w") as f:
            f.write(main_diagram)
        print("  ✅ Pipeline diagram generated")
    
    def _export_data_formats(self):
        """Export data in a single, practical format."""
        if self.manager.cleaned_data is not None:
            success = self.manager.save_data(
                str(self.output_dir / "cleaned_data.csv"),
                format="csv"
            )
            if success:
                print("  ✅ Exported cleaned data: cleaned_data.csv")
            else:
                print("  ⚠️  Failed to export cleaned data")
                # ⚠️ Fallback: try saving in JSON or Parquet
    
    def _generate_deployment_summary(self):
        """Generate deployment summary."""
        # ⚠️ Possible fallback: always print to console even if file save fails
        summary = f"""
CSV DATA PIPELINE DEPLOYMENT SUMMARY
====================================

Deployment Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Input File: {self.csv_path}
Output Directory: {self.output_dir}
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
        with open(self.output_dir / "DEPLOYMENT_SUMMARY.txt", "w") as f:
            f.write(summary)
        print("  ✅ Deployment summary generated")
        print("\n" + summary)


def main():
    """Main deployment function."""

    CSV_FILE = "geyser.csv"
    OUTPUT_DIR = "pipeline_output"
    SKIP_ROWS = 30

    if not os.path.exists(CSV_FILE):
        print(f"❌ Error: CSV file '{CSV_FILE}' not found!")
        # ⚠️ Fallback: prompt user to provide a valid file or exit gracefully
        return

    deployer = PipelineDeployer(CSV_FILE, OUTPUT_DIR)

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
