from dataset.data_processor import create_dataset
import os


raw_data_dir = os.path.join("dataset", "raw")
save_dir = os.path.join("dataset", "data")

# process cornell dataset
# create_dataset(raw_data_dir, save_dir, dataset_name="cornell")

# process twitter dataset
# create_dataset(raw_data_dir, save_dir, dataset_name="twitter")

# process cmudict dataset
create_dataset(raw_data_dir, save_dir, dataset_name="cmudict")
