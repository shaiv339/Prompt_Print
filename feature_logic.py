import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler

MODEL_NAME = "all-MiniLM-L6-v2"  # SBERT model

'''
def load_data(csv_path, num_users, seed=42):
    np.random.seed(seed)
    df = pd.read_csv(csv_path)
    all_users = df['user_id'].unique()
    sampled_users = np.random.choice(all_users, size=num_users, replace=False)
    sampled_users = sorted(sampled_users)
    df_sample = df[df['user_id'].isin(sampled_users)].copy()
    return df_sample, sampled_users
'''



def load_data(csv_path, num_users=5):
    df = pd.read_csv(csv_path)
    unique_users = df['user_id'].unique()
    top_users = unique_users[:num_users].tolist()

    
    df_sample = df[df['user_id'].isin(top_users)].copy()

    print(f"Selected first {num_users} users:")
    print(top_users)

    return df_sample, top_users



'''
# Stylometric feature extraction
def extract_stylometric_features(text):
    words = text.split()
    return [
        np.mean([len(w) for w in words]) if words else 0,          # avg_word_length
        len(words),                                               # word_count
        len(text),                                                # char_count
        text.count('!'),                                          # exclamation_count
        text.count('?'),                                          # question_count
        text.count(','),                                          # comma_count
        text.count('.'),                                          # period_count
        sum(1 for c in text if c.isupper()) / len(text) if text else 0,  # uppercase_ratio
        len(set(w.lower() for w in words))/len(words) if words else 0    # vocab_richness
    ]



def get_stylometric_matrix(prompts):
    return np.array([extract_stylometric_features(p) for p in prompts])


# SBERT embeddings
def get_sbert_embeddings(prompts):
    model = SentenceTransformer(MODEL_NAME)
    return model.encode(prompts, show_progress_bar=True)
'''

def get_style_embeddings(prompts):
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(prompts, show_progress_bar=True)

    # Debug info
    print(f"StyleDistance embedding shape: {embeddings.shape}")
    return embeddings



# Combine SBERT + Stylometry
def build_feature_matrix(df_sample):
    '''
    sbert_embeds = get_sbert_embeddings(df_sample['prompt'].tolist())
    style_features = get_stylometric_matrix(df_sample['prompt'].tolist())
    scaler = StandardScaler()
    style_scaled = scaler.fit_transform(style_features)
    combined_features = np.hstack([sbert_embeds, style_scaled])
    #breakpoint()
    return sbert_embeds
    '''
    style_embeds = get_style_embeddings(df_sample['prompt'].tolist())
    return style_embeds
