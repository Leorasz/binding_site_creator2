from flask import Flask, render_template, request, flash, redirect, url_for
import torch
import random
from tqdm import tqdm  # Note: tqdm might not be visible in web, but can log to console
import torch.nn.functional as F
import io
import os

app = Flask(__name__)
app.secret_key = 'super_secret_key'  # Needed for flash messages

# Device configuration - fallback to CPU for server compatibility
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Mapping from indices to nucleotides
NUC_MAP = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

def parse_pfm(file_content):
    """
    Parse the uploaded PFM text file.
    Assumes format: 4 lines, each starting with nucleotide label or similar,
    but extracts integers from words[2:-1] as in the original code.
    """
    lines = file_content.splitlines()
    if len(lines) != 4:
        raise ValueError("PFM file must have exactly 4 lines.")
    
    nucleotides = []
    seq_len = None
    for line in lines:
        words = line.strip().split()
        if len(words) < 3:
            raise ValueError("Each line must have at least 3 words.")
        try:
            frequencies = [int(word) for word in words[2:-1]]
        except ValueError:
            raise ValueError("Frequencies must be integers.")
        
        if seq_len is None:
            seq_len = len(frequencies)
        elif len(frequencies) != seq_len:
            raise ValueError("All lines must have the same number of frequencies.")
        
        nucleotides.append(frequencies)
    
    if seq_len == 0:
        raise ValueError("PFM must have at least one position.")
    
    return nucleotides, seq_len

def run_genetic_algorithm(pfm_data, target, population_size, elite_prop, generations, mutation_size):
    """
    Adapted from the user's code to run the GA and return the best sequence and its affinity.
    """
    tens = torch.tensor(pfm_data, dtype=torch.float32, device=device) + 0.1
    tens = tens / tens.sum(dim=0, keepdim=True)
    pwm = torch.log2(tens / 0.25)
    seq_len = pwm.shape[1]
    
    elite_count = int(elite_prop * population_size)
    tournament_size = int(population_size / elite_count)
    if population_size % tournament_size != 0:
        raise ValueError("Population size must be divisible by 1/elite_prop for clean grouping.")
    
    blown_up_pwm = pwm.unsqueeze(0).repeat(population_size, 1, 1)  # for scoring
    argmax_fixer = torch.tensor(list(range(0, population_size, tournament_size)), device=device)
    
    # Initial population
    population = torch.randint(0, 4, (population_size, seq_len), device=device)
    
    for gen in range(generations):
        # Compute affinities
        one_hot = F.one_hot(population, num_classes=4).permute(0, 2, 1).float()
        affinities = torch.sum(torch.sum(one_hot * blown_up_pwm, dim=2), dim=1)
        # Scores to minimize: abs(affinity - target)
        scores = torch.abs(affinities - target)
        
        # Print to console (for server logs)
        print(f"Generation [{gen+1}/{generations}] best score is {torch.min(scores).item():.4f}")
        
        # Tournament selection for elites
        tourney_scores = scores.reshape(-1, tournament_size)
        elite_indices = torch.argmin(tourney_scores, dim=1) + argmax_fixer
        elite = population[elite_indices]
        
        # Create new population through recombination
        number_of_new_cohorts = tournament_size - 1
        new_cohorts = []
        for _ in range(number_of_new_cohorts):
            divide = random.randint(0, seq_len)
            first_part = elite.clone()[:, :divide]
            second_part = elite.clone()[torch.randperm(elite_count)][:, divide:]
            new_cohorts.append(torch.cat([first_part, second_part], dim=1))
        new_pop = torch.cat(new_cohorts, dim=0)
        
        # Mutate new population
        one_hot = F.one_hot(new_pop, num_classes=4).permute(0, 2, 1).float()
        logits = one_hot + torch.randn_like(one_hot) * mutation_size
        probs = F.softmax(logits, dim=1)
        flattened_probs = probs.permute(0, 2, 1).reshape(-1, 4)
        samples = torch.multinomial(flattened_probs, num_samples=1).squeeze(1)
        new_pop = samples.view(new_pop.shape[0], seq_len)
        
        # Combine and shuffle
        population = torch.cat([elite, new_pop], dim=0)[torch.randperm(population_size)]
    
    # After all generations, compute final affinities and scores
    one_hot = F.one_hot(population, num_classes=4).permute(0, 2, 1).float()
    affinities = torch.sum(torch.sum(one_hot * blown_up_pwm, dim=2), dim=1)
    scores = torch.abs(affinities - target)
    
    # Find the best
    best_idx = torch.argmin(scores)
    best_sequence = ''.join(NUC_MAP[int(nuc)] for nuc in population[best_idx])
    best_affinity = affinities[best_idx].item()
    best_score = scores[best_idx].item()
    
    return best_sequence, best_affinity, best_score

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        pfm_file = request.files.get('pfm_file')
        target_str = request.form.get('target')
        pop_size_str = request.form.get('population_size')
        elite_prop_str = request.form.get('elite_prop')
        generations_str = request.form.get('generations')
        mutation_size_str = request.form.get('mutation_size')
        
        # Validate required fields
        if not pfm_file or not target_str:
            flash('PFM file and target affinity are required.')
            return redirect(url_for('index'))
        
        try:
            # Read file content
            file_content = io.TextIOWrapper(pfm_file.stream, encoding='utf-8').read()
            
            # Parse PFM
            pfm_data, seq_len = parse_pfm(file_content)
            
            # Parse parameters
            target = float(target_str)
            population_size = int(pop_size_str) if pop_size_str else 100
            elite_prop = float(elite_prop_str) if elite_prop_str else 0.1
            generations = int(generations_str) if generations_str else 100
            mutation_size = float(mutation_size_str) if mutation_size_str else 0.1
            
            # Validate parameters
            if population_size <= 0 or elite_prop <= 0 or elite_prop >= 1 or generations <= 0 or mutation_size < 0:
                raise ValueError("Invalid parameter values.")
            
            # Run GA
            best_seq, best_aff, best_score = run_genetic_algorithm(
                pfm_data, target, population_size, elite_prop, generations, mutation_size
            )
            
            return render_template('result.html', sequence=best_seq, affinity=best_aff, score=best_score)
        
        except ValueError as e:
            flash(str(e))
            return redirect(url_for('index'))
        except Exception as e:
            flash(f"An unexpected error occurred: {str(e)}")
            return redirect(url_for('index'))
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=8501, debug=True)