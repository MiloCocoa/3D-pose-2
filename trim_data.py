import json
import os
import argparse
import sys

def process_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: '{file_path}' is not a valid JSON file.")
        return

    if 'pose_sequence' not in data:
        print(f"Error: 'pose_sequence' key not found in '{file_path}'.")
        return

    original_length = len(data['pose_sequence'])
    print(f"\nLoaded '{file_path}'")
    print(f"Total frames: {original_length}")
    
    if 'metadata' in data:
        print(f"Rep number: {data['metadata'].get('rep_number', 'N/A')}, Subject: {data['metadata'].get('subject_id', 'N/A')}")

    while True:
        print("\nEnter trim command:")
        print("  '-N'    -> Remove first N frames (keep N to end, e.g., '-60')")
        print("  'N-'    -> Remove everything after N (keep 0 to N, e.g., '250-')")
        print("  'A-B'   -> Keep frames from A to B (e.g., '60-250')")
        print("  's'     -> Skip file without saving")
        print("  'q'     -> Quit script")
        
        command = input("> ").strip().lower()

        if command == 'q':
            sys.exit(0)
        elif command == 's':
            print("Skipping...")
            return
        
        try:
            start_idx = 0
            end_idx = original_length

            if command.startswith('-'):
                # -N
                start_idx = int(command[1:])
            elif command.endswith('-'):
                # N-
                end_idx = int(command[:-1])
            elif '-' in command:
                # A-B
                parts = command.split('-')
                start_idx = int(parts[0])
                end_idx = int(parts[1])
            elif not command:
                print("Invalid command. Please try again.")
                continue
            else:
                print("Invalid command format. Please use '-N', 'N-', or 'A-B'.")
                continue

            # Basic validation
            if start_idx < 0 or start_idx >= original_length:
                print(f"Error: Start index {start_idx} out of bounds (0 to {original_length-1}).")
                continue
            if end_idx <= start_idx or end_idx > original_length:
                print(f"Error: End index {end_idx} out of bounds or less than start index.")
                continue

            trimmed_sequence = data['pose_sequence'][start_idx:end_idx]
            new_length = len(trimmed_sequence)
            
            print(f"Trimmed from {original_length} frames to {new_length} frames (Keep {start_idx} to {end_idx}).")
            confirm = input("Save this? (y/n): ").strip().lower()
            if confirm == 'y':
                data['pose_sequence'] = trimmed_sequence
                # Output to same file or new file?
                # Overwriting is easier for the user's manual process, but we'll ask or backup.
                backup_path = file_path + ".bak"
                if not os.path.exists(backup_path):
                    os.rename(file_path, backup_path)
                    print(f"Original file backed up to {backup_path}")
                
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=4)
                print(f"Saved '{file_path}' successfully.")
                break
            else:
                print("Discarding changes...")
                
        except ValueError:
            print("Error: Invalid number format.")

def main():
    print("=== Data Trimming Utility ===")
    if len(sys.argv) > 1:
        # Process files passed as arguments
        for file_path in sys.argv[1:]:
            process_file(file_path)
    else:
        # Interactive mode
        while True:
            file_path = input("\nEnter path to JSON file (or 'q' to quit): ").strip()
            # remove surrounding quotes if dragged into terminal
            file_path = file_path.strip('\"\'')
            if file_path.lower() == 'q':
                break
            if file_path:
                # support wildcards? let's keep it simple
                if os.path.isfile(file_path):
                    process_file(file_path)
                else:
                    print(f"File not found: {file_path}")

if __name__ == "__main__":
    main()
