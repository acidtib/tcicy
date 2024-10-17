import json
from pathlib import Path

def create_metadata(directory, output_file):
    try:
        directory_path = Path(directory)
        metadata_list = []
        
        # Iterate over subdirectories (card names / labels)
        for label_dir in directory_path.iterdir():
            if label_dir.is_dir():
                label = label_dir.name
                
                # Iterate over PNG files in each subdirectory
                for file_path in label_dir.glob("*.png"):
                    # Create the metadata dictionary
                    metadata = {
                        "file_name": str(file_path.relative_to(directory_path)),
                        "label": label
                    }
                    
                    # Append the metadata to the list
                    metadata_list.append(metadata)
        
        # Write all metadata to the output file as JSONL
        with open(output_file, 'w') as f_out:
            for metadata in metadata_list:
                f_out.write(json.dumps(metadata) + "\n")
        
        print(f"Metadata file created: {output_file}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        
# Specify the train and test directories
data_dir = Path('./datasets/tcg_magic/data')
train_directory = data_dir / 'train'
train_metadata_file = train_directory / 'metadata.jsonl'

test_directory = data_dir / 'test'
test_metadata_file = test_directory / 'metadata.jsonl'

# Create metadata files
create_metadata(train_directory, train_metadata_file)
create_metadata(test_directory, test_metadata_file)
