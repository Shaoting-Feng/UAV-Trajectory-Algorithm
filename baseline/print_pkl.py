import pickle

# Specify the path to your pickle file
pkl_file_path = '/home/baseline/output/objects.pkl'

# Open the file in binary read mode
with open(pkl_file_path, 'rb') as file:
    # Load the data from the pickle file
    data = pickle.load(file)

# Now you can use the data as per your requirements
print(data)
