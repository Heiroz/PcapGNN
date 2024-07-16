import torch

def decode_binary_vector_to_integers(binary_vector, bit_length=256):
    decoded_values = []
    for i in range(0, binary_vector.size(1), bit_length):
        # Filter only '0' and '1' characters
        binary_str = ''.join(str(int(bit)) for bit in binary_vector[0, i:i+bit_length] if str(int(bit)) in ['0', '1'])
        if len(binary_str) > 0:
            decoded_values.append(int(binary_str, 2))
        else:
            decoded_values.append(0)  # Add 0 if no valid binary digit found
    return torch.tensor(decoded_values, dtype=torch.long)

def decode_tensor(encoded_tensor, bit_length=256):
    num_samples = encoded_tensor.size(0)
    decoded_tensor = torch.zeros(num_samples, 6, dtype=torch.long)
    
    for i in range(num_samples):
        decoded_tensor[i] = decode_binary_vector_to_integers(encoded_tensor[i].unsqueeze(0), bit_length)
    
    return decoded_tensor

# Your main function and other parts of the code should remain the same
