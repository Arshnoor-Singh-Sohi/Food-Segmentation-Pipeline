#!/usr/bin/env python3
"""
Training Logger - Captures Everything That Happens
This script runs your training commands and saves all output to organized log files

Usage:
    python run_with_logging.py --command "python fix_training_issues.py"
    python run_with_logging.py --command "python scripts/train_custom_food_model.py --mode check_setup"
    python run_with_logging.py --command "python scripts/train_custom_food_model.py --mode quick_test"
    python run_with_logging.py --quick-test
    python run_with_logging.py --full-training --epochs 50
"""

import sys
import subprocess
import argparse
import os
from pathlib import Path
from datetime import datetime
import threading
import queue
import time

class TrainingLogger:
    """
    Comprehensive logger that captures all training output
    Creates organized log files with timestamps and real-time display
    """
    
    def __init__(self):
        self.start_time = datetime.now()
        self.session_id = self.start_time.strftime("%Y%m%d_%H%M%S")
        
        # Create logs directory
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Create session-specific log directory
        self.session_dir = self.logs_dir / f"training_session_{self.session_id}"
        self.session_dir.mkdir(exist_ok=True)
        
        # Log file paths
        self.main_log = self.session_dir / "complete_log.txt"
        self.error_log = self.session_dir / "errors_only.txt"
        self.summary_log = self.session_dir / "session_summary.txt"
        
        print(f"[LOG] Training Logger Started")
        print(f"[TIME] Session ID: {self.session_id}")
        print(f"[FOLDER] Logs directory: {self.session_dir}")
        print(f"[FILE] Main log: {self.main_log}")
        print("=" * 60)
    
    def log_system_info(self):
        """Log system information at the start"""
        info_lines = [
            f"Training Session Started: {self.start_time}",
            f"Session ID: {self.session_id}",
            f"Working Directory: {os.getcwd()}",
            f"Python Version: {sys.version}",
            f"Platform: {sys.platform}",
            "=" * 50
        ]
        
        # Check for GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info = f"GPU Available: {torch.cuda.get_device_name(0)}"
                gpu_memory = f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
                info_lines.extend([gpu_info, gpu_memory])
            else:
                info_lines.append("GPU: Not available (CPU training)")
        except ImportError:
            info_lines.append("PyTorch: Not installed")
        
        info_lines.append("=" * 50)
        
        # Write to log file
        with open(self.main_log, 'w') as f:
            f.write("\n".join(info_lines) + "\n\n")
        
        # Display system info
        for line in info_lines:
            print(line)
    
    def run_command_with_logging(self, command):
        """
        Run a command and capture all output with real-time display
        
        Args:
            command: Command string to execute
            
        Returns:
            dict: Results including return code and captured output
        """
        print(f"[RUN] Executing: {command}")
        print("=" * 60)
        
        # Log command start
        self._log_to_file(self.main_log, f"\n{'='*60}")
        self._log_to_file(self.main_log, f"COMMAND: {command}")
        self._log_to_file(self.main_log, f"STARTED: {datetime.now()}")
        self._log_to_file(self.main_log, f"{'='*60}")
        
        # Prepare to capture output
        all_output = []
        error_output = []
        
        try:
            # Start the process
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Read output line by line in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                
                if output:
                    line = output.strip()
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    
                    # Display in console with timestamp
                    print(f"[{timestamp}] {line}")
                    
                    # Save to log file
                    log_line = f"[{timestamp}] {line}"
                    all_output.append(log_line)
                    self._log_to_file(self.main_log, log_line)
                    
                    # Check for errors or warnings
                    if any(keyword in line.lower() for keyword in ['error', 'exception', 'failed', 'warning']):
                        error_output.append(log_line)
                        self._log_to_file(self.error_log, log_line)
            
            # Get return code
            return_code = process.poll()
            
            # Log completion
            end_time = datetime.now()
            duration = end_time - self.start_time
            
            completion_info = [
                f"COMPLETED: {end_time}",
                f"DURATION: {duration}",
                f"RETURN CODE: {return_code}",
                f"{'='*60}"
            ]
            
            for line in completion_info:
                print(line)
                self._log_to_file(self.main_log, line)
            
            return {
                'return_code': return_code,
                'success': return_code == 0,
                'output': all_output,
                'errors': error_output,
                'duration': duration
            }
            
        except Exception as e:
            error_msg = f"EXECUTION ERROR: {str(e)}"
            print(f"[FAIL] {error_msg}")
            self._log_to_file(self.main_log, error_msg)
            self._log_to_file(self.error_log, error_msg)
            
            return {
                'return_code': -1,
                'success': False,
                'output': all_output,
                'errors': [error_msg],
                'duration': datetime.now() - self.start_time
            }
    
    def _log_to_file(self, file_path, message):
        """Append a message to a log file"""
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    
    def create_session_summary(self, results):
        """Create a summary of the entire session"""
        summary_lines = [
            f"TRAINING SESSION SUMMARY",
            f"=" * 40,
            f"Session ID: {self.session_id}",
            f"Started: {self.start_time}",
            f"Ended: {datetime.now()}",
            f"Total Duration: {datetime.now() - self.start_time}",
            f"",
            f"COMMANDS EXECUTED:",
            f"-" * 20
        ]
        
        for i, result in enumerate(results, 1):
            status = "[OK] SUCCESS" if result['success'] else "[FAIL] FAILED"
            summary_lines.append(f"{i}. {status} (Return code: {result['return_code']})")
            summary_lines.append(f"   Duration: {result['duration']}")
            
            if result['errors']:
                summary_lines.append(f"   Errors found: {len(result['errors'])}")
        
        summary_lines.extend([
            f"",
            f"LOG FILES:",
            f"-" * 10,
            f"Complete log: {self.main_log}",
            f"Errors only: {self.error_log}",
            f"This summary: {self.summary_log}",
            f"",
            f"{'='*40}"
        ])
        
        # Write summary to file
        with open(self.summary_log, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        # Display summary
        print("\n" + "="*60)
        print("[STATS] SESSION SUMMARY")
        print("="*60)
        for line in summary_lines:
            print(line)

def main():
    """Main entry point with command line options"""
    parser = argparse.ArgumentParser(
        description="Run training commands with comprehensive logging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run any command with logging
  python run_with_logging.py --command "python fix_training_issues.py"
  
  # Quick shortcuts for common commands
  python run_with_logging.py --setup-check
  python run_with_logging.py --quick-test
  python run_with_logging.py --full-training --epochs 50
  
  # Run multiple commands in sequence
  python run_with_logging.py --setup-check --quick-test
        """
    )
    
    # Command options
    parser.add_argument('--command', type=str, help='Custom command to run with logging')
    parser.add_argument('--setup-check', action='store_true', help='Run setup troubleshooter')
    parser.add_argument('--quick-test', action='store_true', help='Run quick test training')
    parser.add_argument('--full-training', action='store_true', help='Run full training')
    parser.add_argument('--epochs', type=int, help='Number of epochs for training')
    
    args = parser.parse_args()
    
    # Create logger
    logger = TrainingLogger()
    logger.log_system_info()
    
    # Collect commands to run
    commands = []
    
    if args.command:
        commands.append(args.command)
    
    if args.setup_check:
        commands.append("python fix_training_issues.py")
        commands.append("python scripts/train_custom_food_model.py --mode check_setup")
    
    if args.quick_test:
        epoch_arg = f" --epochs {args.epochs}" if args.epochs else ""
        commands.append(f"python scripts/train_custom_food_model.py --mode quick_test{epoch_arg}")
    
    if args.full_training:
        epoch_arg = f" --epochs {args.epochs}" if args.epochs else ""
        commands.append(f"python scripts/train_custom_food_model.py --mode full_training{epoch_arg}")
    
    if not commands:
        print("[FAIL] No commands specified. Use --help for options.")
        return
    
    # Execute all commands
    results = []
    
    for i, command in enumerate(commands, 1):
        print(f"\n[STEP] STEP {i}/{len(commands)}: {command}")
        result = logger.run_command_with_logging(command)
        results.append(result)
        
        # If a command fails, ask if we should continue
        if not result['success'] and i < len(commands):
            print(f"\n[WARN]  Command {i} failed (return code: {result['return_code']})")
            
            response = input("Continue with next command? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                print("[STOP] Stopping execution due to user choice")
                break
    
    # Create final summary
    logger.create_session_summary(results)
    
    print(f"\n[FOLDER] All logs saved to: {logger.session_dir}")
    print("[TARGET] Check the logs for detailed information about what happened!")

if __name__ == "__main__":
    main()