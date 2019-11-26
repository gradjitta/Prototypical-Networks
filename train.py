import spacy
import pandas as pd
import numpy as np
import torch
import argparse
import pickle
import os
import random
import fire

from model.protonet_text import ProtoNetText, ProtoDummyNet
from model.protonet_text import ProtoLoss
from data.load_data import TextGenerator

def get_latents(x,y, embed_size, n_way, n_query, k_shot):
    x_support, x_query = x[:,:,:k_shot,:], x[:,:,k_shot:,:]
    y_support, y_query = y[:,:,:k_shot,:], y[:,:,k_shot:,:]
    labels_onehot = y_query.reshape(n_way, n_query, n_way)
    support_input_t = torch.Tensor(x_support).view(-1, embed_size)
    query_input_t = torch.Tensor(x_query).view(-1, embed_size)
    return support_input_t, query_input_t, labels_onehot

def main(n_way= 5, k_shot = 5, n_query = 2, proto_dim = 32, train_mode = 'normal'):
    n_meta_test_way = 5
    k_meta_test_shot = 5
    n_meta_test_query = 2
    num_epochs = 20
    num_episodes = 200
    embed_size = 768
    hidden_dim = 100
    text_vectors = pickle.load(open('data/mini_newsgroup_vectors.pkl','rb'))
    mini_df = pickle.load(open('data/mini_newsgroup_data.pkl','rb'))
    text_generator_ = TextGenerator(mini_df, n_way, k_shot+n_query, n_meta_test_way, k_meta_test_shot+n_meta_test_query)
    if train_mode == 'normal':
        model_text = ProtoNetText(embed_size, hidden_dim, proto_dim)
        optimizer_text = torch.optim.Adam(model_text.parameters(), lr=1e-4)
    else:
        model_text = ProtoDummyNet()
        proto_dim = 768
    criterion = ProtoLoss(n_way, k_shot, n_query, proto_dim)
    for ep in range(num_epochs):
        print(f'Epoch: {ep}')
        for epi in range(num_episodes):
            x, y = text_generator_.sample_batch('meta_train', 1, text_vectors, shuffle = False)
            support_input_t, query_input_t, labels_onehot = get_latents(x,y, embed_size, n_way, n_query, k_shot)
            x_latent = model_text(support_input_t)
            q_latent = model_text(query_input_t)
            # Compute and print loss
            if train_mode == 'normal':
                loss, _ = criterion(x_latent, q_latent, torch.tensor(labels_onehot))
                #if epi % 50 == 0:
                # #    print(f'Epoc {ep}/{num_epochs} Episode {epi}/{num_episodes}, Accuracy: {round(accuracy.item(),3)}, Training Loss: {round(loss.item(),3)}')
                # # Zero gradients, perform a backward pass, and update the weights.
                optimizer_text.zero_grad()
                loss.backward()
                optimizer_text.step()
            else:
                loss, _ = criterion(x_latent, q_latent, torch.tensor(labels_onehot))
            if epi % 50 == 0:
                with torch.no_grad():
                    valid_x, valid_y = text_generator_.sample_batch('meta_val', 1,text_vectors, shuffle = False)
                    support_input_valid, query_input_valid, labels_onehot_valid = get_latents(valid_x,valid_y, embed_size, n_way, n_query, k_shot)
                    x_latent_valid = model_text(support_input_valid)
                    q_latent_valid = model_text(query_input_valid)
                    # Compute and print loss
                    valid_loss, valid_acc = criterion(x_latent_valid, q_latent_valid, torch.tensor(labels_onehot_valid))
                    print(f'Epoc {ep}/{num_epochs} Episode {epi}/{num_episodes}, Validation Accuracy: {round(valid_acc.item(),3)}, Validation Loss: {round(valid_loss.item(),3)}')
    print('Testing ..........................')
    meta_test_accuracies = []
    for epi in range(1000):
        test_x, test_y = text_generator_.sample_batch('meta_test', 1,text_vectors, shuffle = False)
        support_input_test, query_input_test, labels_onehot_test = get_latents(test_x,test_y, embed_size, n_way, n_query, k_shot)
        with torch.no_grad():
            x_latent_test = model_text(support_input_test)
            q_latent_test = model_text(query_input_test)
            # Compute and print loss
            test_loss, test_acc = criterion(x_latent_test, q_latent_test, torch.tensor(labels_onehot_valid))
            if (epi + 1) % 50 == 0:
                print(f'Meta test Episode {epi}/{1000}, Test Accuracy: {round(test_acc.item(),3)}, Test Loss: {round(test_loss.item(),3)}')
            meta_test_accuracies.append(test_acc)
    avg_acc = np.mean(meta_test_accuracies)
    stds = np.std(meta_test_accuracies)
    print('Average Meta-Test Accuracy: {:.5f}, Meta-Test Accuracy Std: {:.5f}'.format(avg_acc, stds))
    

if __name__ == "__main__":
    fire.Fire(main)