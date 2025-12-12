import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Load raw data from file
with open("data.txt", "r") as file:
    raw_data = file.readlines()

# Parse PFMs from text file
data = []
for datum_start in range(1, len(raw_data), 5):
    nucleotides = []
    for i in range(4):
        words = raw_data[datum_start + i].split()
        frequencies = [int(word) for word in words[2:-1]]
        nucleotides.append(frequencies)
    tens = torch.tensor(nucleotides, dtype=torch.float32).to(device) + 0.25  # Add pseudocount
    tens = tens / tens.sum(dim=0, keepdim=True)  # Normalize to PPM
    data.append(tens)
data = data[:1000]

# Determine max length and optionally limit number of PFMs for testing
max_len = max([datum.shape[1] for datum in data])

# Pad all PPMs to max_len with uniform (0.25) columns
padded_data = []
for tens in data:
    l = tens.shape[1]
    if l < max_len:
        pad_width = max_len - l
        tens = torch.nn.functional.pad(tens, (0, pad_width), mode='constant', value=0.25)
    padded_data.append(tens)
data = padded_data

# Create PWMs (log-odds scores) from PPMs assuming uniform background (0.25)
pwms = [torch.log2(ppm / 0.25) for ppm in data]

# Helper function to sample a one-hot sequence from a PPM (multinomial per position)
def sample_sequence_from_ppm(ppm):
    _, length = ppm.shape
    bases = []
    for pos in range(length):
        probs = ppm[:, pos]
        base = torch.multinomial(probs, num_samples=1).item()
        bases.append(base)
    one_hot = torch.zeros(4, length, device=device)
    one_hot[bases, torch.arange(length)] = 1
    return one_hot

# Helper function to sample a random uniform one-hot sequence
def sample_random_sequence(length):
    bases = torch.randint(0, 4, (length,), device=device)
    one_hot = torch.zeros(4, length, device=device)
    one_hot[bases, torch.arange(length)] = 1
    return one_hot

# Helper function to compute hard affinity (score) of a one-hot sequence against PWM
def compute_affinity(one_hot_seq, pwm):
    return torch.sum(one_hot_seq * pwm)

# Helper function to convert sequence indices to string
def indices_to_string(indices):
    base_map = ['A', 'C', 'G', 'T']
    return ''.join(base_map[i] for i in indices)

# Generate training examples: For refinement, set target = current_aff + delta
class AffinityDataset(Dataset):
    def __init__(self, training_examples):
        self.examples = training_examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

num_samples_per_ppm = 10  # Increased for variety
delta = 5.0  # Adjustable increment for refinement
training_examples = []
print("Making training samples")
for i in tqdm(range(len(data))):
    ppm = data[i]
    pwm = pwms[i]
    for _ in range(num_samples_per_ppm):
        input_seq = sample_random_sequence(max_len)  # Or start from mutated sampled seq
        current_aff = compute_affinity(input_seq, pwm)
        desired_aff = current_aff + delta  # Refine by increment, clamp if needed
        training_examples.append((ppm, pwm, input_seq, desired_aff))

dataset = AffinityDataset(training_examples)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Simpler network: Outputs delta_logits to add to input's implied logits
input_size = max_len * 8 + 1  # input_seq flat (len*4) + ppm flat (len*4) + desired (1)
hidden_size = max_len * 4  # Reduced
output_size = max_len * 4  # Delta logits for refined sequence (len positions x 4 bases)
affin_net = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, output_size)
).to(device)

# Large value for input one-hot to logits conversion (to make it strongly peaked)
logit_scale = 10.0

# Regularization lambda for minimal changes (L1 on deltas)
reg_lambda = 0.01  # Adjustable; set to 0 to disable

optimizer = torch.optim.AdamW(affin_net.parameters(), lr=0.001, weight_decay=1e-5)
num_epochs = 50
losses = []

print("About to train")
for epoch in range(num_epochs):
    total_loss = 0.0
    for batch in tqdm(dataloader):
        ppms, pwms, input_seqs, desireds = batch  # Unpack batch (already collated/stacked)
        ppms = ppms.to(device)  # (batch, 4, len)
        pwms = pwms.to(device)
        input_seqs = input_seqs.to(device)  # (batch, 4, len)
        desireds = desireds.to(device)

        optimizer.zero_grad()
        
        # Flatten batch-wise
        input_seq_flat = input_seqs.permute(0, 2, 1).contiguous().view(input_seqs.shape[0], -1)  # (batch, len*4)
        ppm_flat = ppms.permute(0, 2, 1).contiguous().view(ppms.shape[0], -1)
        input_flat = torch.cat((input_seq_flat, ppm_flat, desireds.unsqueeze(1)), dim=1)  # (batch, input_size)
        
        # Forward: Get delta_logits
        delta_logits = affin_net(input_flat)  # (batch, len*4)
        delta_logits_reshaped = delta_logits.view(input_seqs.shape[0], max_len, 4)  # (batch, len, 4)
        
        # Convert input one-hot to strong logits
        input_logits = (input_seqs.permute(0, 2, 1) * logit_scale) - ((1 - input_seqs.permute(0, 2, 1)) * logit_scale)
        # (batch, len, 4)
        
        # Apply changes
        final_logits = input_logits + delta_logits_reshaped
        
        # Soft prob for affinity
        soft_prob = torch.softmax(final_logits, dim=2)  # (batch, len, 4)
        soft_ppm = soft_prob.permute(0, 2, 1)  # (batch, 4, len)
        aff_out = torch.sum(soft_ppm * pwms, dim=(1, 2))  # (batch,)
        
        # MSE loss on affinities
        mse_loss = torch.mean((aff_out - desireds) ** 2)
        
        # Optional: L1 reg on deltas to encourage minimal changes
        reg_loss = reg_lambda * torch.mean(torch.abs(delta_logits))
        
        loss = mse_loss + reg_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * input_seqs.shape[0]
    
    avg_loss = total_loss / len(training_examples)
    losses.append(avg_loss)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# Plot loss curve
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# After training, generate and visualize examples for each PPM
for i in range(len(data)):
    print(f"\n--- Example for PPM {i+1} ---")
    example_ppm = data[i]
    example_pwm = pwms[i]
    # Use a desired affinity from training (e.g., first sample for this PPM)
    example_desired = training_examples[i * num_samples_per_ppm][3]
    example_input_seq = sample_random_sequence(max_len)
    # Compute input affinity (hard)
    aff_in = compute_affinity(example_input_seq, example_pwm)
    # Prepare input
    input_seq_flat = example_input_seq.t().contiguous().view(-1)
    ppm_flat = example_ppm.t().contiguous().view(-1)
    example_input_flat = torch.cat((input_seq_flat, ppm_flat, torch.tensor([example_desired], device=device)))
    # Generate delta_logits
    delta_output = affin_net(example_input_flat)
    delta_logits_reshaped = delta_output.view(max_len, 4)
    # Input to logits
    input_logits = (example_input_seq.t() * logit_scale) - ((1 - example_input_seq.t()) * logit_scale)
    # Final
    final_logits = input_logits + delta_logits_reshaped
    # Get sequences as strings
    input_indices = torch.argmax(example_input_seq, dim=0).cpu().tolist()  # argmax over bases
    input_string = indices_to_string(input_indices)
    output_indices = torch.argmax(final_logits, dim=1).cpu().tolist()  # argmax over bases
    generated_string = indices_to_string(output_indices)
    # Compute output affinities (soft for training consistency, and hard for comparison)
    soft_prob = torch.softmax(final_logits, dim=1)
    soft_ppm = soft_prob.t()
    aff_out_soft = torch.sum(soft_ppm * example_pwm)
    output_one_hot = torch.zeros(4, max_len, device=device)
    output_one_hot[output_indices, torch.arange(max_len)] = 1
    aff_out_hard = compute_affinity(output_one_hot, example_pwm)
    # Print results
    print(f"Input sequence: {input_string}")
    print(f"Generated sequence: {generated_string}")
    print(f"Desired affinity: {example_desired:.4f}")
    print(f"Input affinity (hard): {aff_in:.4f}")
    print(f"Generated affinity (soft): {aff_out_soft:.4f}")
    print(f"Generated affinity (hard): {aff_out_hard:.4f}")
    # Visualize
    # Pause to view before next
    input("Press Enter to continue...")