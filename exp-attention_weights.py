import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import numpy as np

# 1. Load model and tokenizer
model_name = "bert-base-uncased"  # You can use any transformer model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True)

# 2. Prepare input text
text = "Item number seven offered by mckitrick ordinance authorizing the mayor or his design to enter into a contractor contracts without the formality of publicly advertising for bids for the purchase of vehicles for the fire Division and declaring an emergency councilman mckitrick uh the committee's report's favorable we're asking for suspension of the rle are there any objections to suspension of the rules seeing and hearing none the rules have been suspended uh most of the time when uh the fire department's purchasing Vehicles they go through a bid process uh we have two specialized vehicles that they're looking to replace that have over 100,000 miles uh and it's starting to get costly on the uh the maintenance on these vehicles so many times they find them in a uh at a car dealership to be able to purchase but if they have to go through the bid process process many times they lose these vehicles so they're not trying to circumvent the bidding process they just want to be able to obtain these vehicles why they're available without drawing it out any further and we're asking for passage thank you the rules have been suspended the committee's report is favorable all in favor signify by saying I I any oppos the eyes have it this ordinance passes 11 to zer number eight is offered by President Somerville resolution appointing Bruce Balden to fill the vacancy in the akan city council w 8 position until a new w 8 representative can be elected at the next regularly scheduled primary and general elections at which all electors of the city are eligible to vote and declaring an emergency okay thank you so much the screening committee has recommended Bruce Balden at this time may I have a motion to nominate Bruce Balden as Ward 8 representative have a motion is there a second second all in favor signify by saying I I I any oppose the eyes have it at this time we're going to open up the floor are there any other nominations do we have a motion to close nominations nominations is there a second okay so in accordance with open Record Law and the opinion of the Attorney General secret ballots are prohibited just so that you know what's going on therefore the vote will be by ballot however your name is placed on the ballot so each ballot has the council person's name on it for for and at this time we're going to ask our clerk to announce the results Madame President Bruce Balden received 11 votes"
tokens = tokenizer(text, return_tensors="pt")
input_ids = tokens["input_ids"]

# 3. Get attention weights
model.eval()
with torch.no_grad():
    outputs = model(**tokens)
    attentions = outputs.attentions  # List of attention tensors for each layer

# 4. Analyze attention shifts
layer_idx = -1  # Analyze the last layer (you can choose others)
head_idx = 0    # Analyze the first head (try others too)
attention_matrix = attentions[layer_idx][0, head_idx].numpy()

# 5. Visualize attention weights
def plot_all_attention_heads(attentions, input_ids, tokenizer):
    layer_idx = -1  # Last layer
    num_heads = attentions[layer_idx].shape[1]  # Should be 12 for BERT-base
    
    # Create a grid of subplots (4x3)
    fig, axes = plt.subplots(4, 3, figsize=(12, 8))
    fig.suptitle("Attention Patterns Across All Heads", fontsize=16)
    
    # Get tokens for labels
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Plot each head's attention pattern
    for head_idx in range(num_heads):
        row = head_idx // 3
        col = head_idx % 3
        
        attention_matrix = attentions[layer_idx][0, head_idx].numpy()
        
        # Create heatmap
        im = axes[row, col].imshow(attention_matrix, cmap="viridis")
        axes[row, col].set_title(f'Head {head_idx}')
        
        # Only label axes for outer plots
        if col == 0:  # leftmost plots
            axes[row, col].set_yticks(range(len(tokens)))
            axes[row, col].set_yticklabels(tokens, fontsize=8)
        else:
            axes[row, col].set_yticks([])
            
        if row == 3:  # bottom plots
            axes[row, col].set_xticks(range(len(tokens)))
            axes[row, col].set_xticklabels(tokens, rotation=90, fontsize=8)
        else:
            axes[row, col].set_xticks([])
        
        plt.colorbar(im, ax=axes[row, col])
    
    plt.tight_layout()
    plt.show()

# Replace the original plotting call with:
plot_all_attention_heads(attentions, input_ids, tokenizer)

# Add after the visualization:
def analyze_topic_shift(attention_matrix, tokens):
    # Calculate average attention per token
    avg_attention = attention_matrix.mean(axis=0)
    
    # Find tokens with highest attention
    top_indices = np.argsort(avg_attention)[-5:]  # Top 5 attended tokens
    print("\nMost attended tokens:")
    for idx in top_indices:
        print(f"{tokens[idx]}: {avg_attention[idx]:.3f}")
    
    # Look for sudden changes in attention patterns
    attention_diff = np.abs(np.diff(avg_attention))
    shift_points = np.where(attention_diff > np.mean(attention_diff) + np.std(attention_diff))[0]
    
    print("\nPotential topic shift points:")
    for point in shift_points:
        print(f"Between tokens: '{tokens[point]}' and '{tokens[point + 1]}'")

# Analyze the topic shifts
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
analyze_topic_shift(attention_matrix, tokens)

def find_attention_streaks(attentions, tokens, threshold=0.8):
    layer_idx = -1
    num_heads = attentions[layer_idx].shape[1]
    
    for head_idx in range(num_heads):
        attention_matrix = attentions[layer_idx][0, head_idx].numpy()
        # Calculate column-wise mean (average attention received by each token)
        column_means = attention_matrix.mean(axis=0)
        
        # Find indices where attention is very high
        high_attention_indices = np.where(column_means > threshold)[0]
        
        if len(high_attention_indices) > 0:
            print(f"\nHead {head_idx} high attention tokens:")
            for idx in high_attention_indices:
                print(f"Token index {idx}: '{tokens[idx]}' (attention score: {column_means[idx]:.3f})")

# Add this after your visualization
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
find_attention_streaks(attentions, tokens)
