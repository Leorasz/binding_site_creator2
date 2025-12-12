import torch
import random
import torch.nn.functional as F
import time
import itertools
from tqdm import tqdm

dev = "mps"
#set these parameters manually
choose_index = 1001 #choose which PFM to evolve to from the data
target = 4.17 #choose the affinity to evolve to
population_size = 100 #population to be evolving
elite_prop = 0.1 #proportion of elites to reproduce (0-1)
generations = 100 #number of generations
mutation_size = 0.1 #how much mutation there should be
batch_size = 10048576  # Adjust based on memory; 1M sequences

for choose_index in [0, 5, 12, 44, 45, 135]:
    with open("data.txt", "r") as file:
        raw_data = file.readlines()
    lines = raw_data[choose_index*5:][1:5]
    nucleotides = []
    for i in range(4):
        words = lines[i].split()
        frequencies = [int(word) for word in words[2:-1]]
        nucleotides.append(frequencies)
    tens = torch.tensor(nucleotides, dtype=torch.float32, device=dev) + 0.1
    tens = tens / tens.sum(dim=0, keepdim=True)
    pwm = torch.log2(tens / 0.25)
    seq_len = pwm.shape[1]
    print(f"Sequence length is {seq_len}")
    # Brute force with batching for parallel computation
    total_seq = 4 ** seq_len
    if total_seq == 0:
        print("Sequence length is 0, skipping")
        continue
    min_diff = float('inf')
    start = time.time()
    num_batches = (total_seq + batch_size - 1) // batch_size
    with tqdm(total=num_batches, desc="Brute forcing") as pbar:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_seq)
            current_batch_size = end_idx - start_idx
            indices = torch.arange(start_idx, end_idx, device=dev)
            population = torch.zeros((current_batch_size, seq_len), dtype=torch.long, device=dev)
            temp = indices.clone()
            for pos in range(seq_len):
                population[:, pos] = temp % 4
                temp //= 4
            one_hot = F.one_hot(population, num_classes=4).permute(0, 2, 1).float()
            blown_up_pwm = pwm.unsqueeze(0).repeat(current_batch_size, 1, 1)
            scores = torch.sum(torch.sum(one_hot * blown_up_pwm, dim=2), dim=1)
            diffs = torch.abs(scores - target)
            min_diff = min(min_diff, torch.min(diffs).item())
            pbar.update(1)
    print(f"Best score is {min_diff:.4f}, ran in {time.time() - start}")