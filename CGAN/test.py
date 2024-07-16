import torch
import csv
import numpy as np
from generator import Generator
from get_flows import get_flows
from decode import decode_tensor
def get_num_attributes(flows):
    sample_flow = flows[0]
    num_condition = sample_flow['flow_vector'].shape[0]
    num_output = sample_flow['remaining_features'].shape[1]
    return num_condition, num_output

def load_generator(checkpoint_path, noisy_size, output_dim, condition_dim):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # Create generator model
    generator = Generator(noisy_size, output_dim, condition_dim)
    
    # Load state dictionary into the model
    generator.load_state_dict(checkpoint)
    generator.eval()
    
    return generator

def generate_samples(generator, condition_data, noisy_dim, num_samples):
    # Convert condition data to tensor
    condition_data = torch.tensor(condition_data, dtype=torch.float32)
    
    # Generate noise
    noise = torch.randn(num_samples, noisy_dim)
    
    # Generate samples
    with torch.no_grad():
        generated_data = generator(noise, condition_data)
    
    return generated_data

def save_samples_to_csv(samples, file_path):
    # Convert samples to numpy array for easier manipulation
    samples = samples.cpu().numpy()
    
    # Write samples to CSV file
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Sample Index'] + [f'Feature {i}' for i in range(samples.shape[1])])
        for i, sample in enumerate(samples):
            writer.writerow([i] + sample.tolist())

def main():
    # Load the flows to get the data structure and dimensions
    flows = get_flows('caida_small.pcap')
    condition_dim, output_dim = get_num_attributes(flows)
    
    # Specify noisy size, output dim, and condition dim manually
    noisy_size = 1024  # Example value, should match the value used during training
    output_dim = output_dim  # This should be the same as the output_dim used during training
    condition_dim = condition_dim  # This should be the same as the condition_dim used during training

    # Load the trained generator model
    checkpoint_path = 'generator.pth'
    generator = load_generator(checkpoint_path, noisy_size, output_dim, condition_dim)

    # Prepare condition data for inference
    condition_data = [flow['flow_vector'] for flow in flows]  # Example condition data
    
    # Ensure all condition_data elements have the same shape
    max_len = max(len(cond) for cond in condition_data)
    condition_data_padded = [np.pad(cond, (0, max_len - len(cond)), 'constant') for cond in condition_data]
    
    condition_data = np.array(condition_data_padded, dtype=np.float32)  # Ensure it is in numpy array format
    num_samples = len(condition_data)

    # Generate new samples
    generated_samples = generate_samples(generator, condition_data, noisy_size, num_samples)
    generated_samples = generated_samples.reshape(-1, output_dim)
    generated_samples = decode_tensor(generated_samples, 32)
    # Save generated samples to CSV file
    output_csv_path = 'generated_samples.csv'
    save_samples_to_csv(generated_samples, output_csv_path)
    print(f"Generated samples saved to {output_csv_path}")

if __name__ == "__main__":
    main()
