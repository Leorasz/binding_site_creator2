import torch
import random
import torch.nn.functional as F
import time

dev = "mps"

#set these parameters manually
choose_index = 1001 #choose which PFM to evolve to from the data
target = 4.17 #choose the affinity to evolve to
population_size = 100 #population to be evolving
elite_prop = 0.1 #proportion of elites to reproduce (0-1)
generations = 100 #number of generations
mutation_size = 0.1 #how much mutation there should be

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
    elite_count = int(elite_prop*population_size)
    tournament_size = int(population_size/elite_count) #this might screw things up if elite_count was rounded
    assert population_size % tournament_size == 0, "Population size must be divisible by 1/elite_prop for clean grouping"

    blown_up_pwm = pwm.unsqueeze(0).repeat(population_size, 1, 1) #for scoring
    argmax_fixer = torch.tensor(list(range(0, population_size, tournament_size)), device=dev) #fixes argmax

    for generations in [50, 100, 500, 1000]:
        #original population
        population = torch.randint(0, 4, (population_size, seq_len), device=dev)
        start = time.time()

        for gen in range(generations):
            #compute scores
            one_hot = F.one_hot(population, num_classes=4).permute(0, 2, 1).float() 
            scores = torch.abs(torch.sum(torch.sum(one_hot*blown_up_pwm, dim=2), dim=1) - target)
            if gen == generations - 1:
                print(f"Generation [{gen+1}/{generations}] best score is {torch.min(scores).item():.4f}, ran in {time.time() - start}")

            #get elite, tournament style
            tourney_scores = scores.reshape(-1, tournament_size)
            elite_indices = torch.argmin(tourney_scores, dim=1) + argmax_fixer
            elite = population[elite_indices]

            #create new population through recombination
            number_of_new_cohorts = tournament_size - 1
            new_cohorts = []
            for _ in range(number_of_new_cohorts):
                divide = random.randint(0,seq_len)
                first_part = elite.clone()[:, :divide]
                second_part = elite.clone()[torch.randperm(elite_count)][:, divide:]
                new_cohorts.append(torch.cat([first_part, second_part], dim=1))
            new_pop = torch.cat(new_cohorts, dim=0)

            #mutate new population
            one_hot = F.one_hot(new_pop, num_classes=4).permute(0, 2, 1).float()
            logits = one_hot + torch.randn_like(one_hot) * mutation_size #add Gaussian noise
            probs = F.softmax(logits, dim=1)
            flattened_probs = probs.permute(0, 2, 1).reshape(-1, 4)
            samples = torch.multinomial(flattened_probs, num_samples=1).squeeze(1) #sample from "blurred" probs
            new_pop = samples.view(new_pop.shape[0], seq_len)

            population = torch.cat([elite, new_pop], dim=0)[torch.randperm(population_size)] #this shuffle is nrecessary for the next tournament