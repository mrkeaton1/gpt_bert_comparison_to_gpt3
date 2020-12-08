###  Causal Language modeling for autoregressive (GPT) models
from nltk.corpus import stopwords
from transformers import GPT2LMHeadModel, GPT2Tokenizer, top_k_top_p_filtering
import torch
import os
from time import time
from utils import elapsed_time
import matplotlib.pyplot as plt


# Causal Language Modeling for Autoregressive Models (GPT line) on Lambada Dataset
def CLM_Lambada_ARM(model, tokenizer, stop_words):
    start = time()
    corrects = 0
    sample_count = 0
    with open(os.path.join('lambada-dataset', 'lambada_test_plain_text.txt')) as lambada:
        for idx, sequence in enumerate(lambada.readlines()):
            sample_count += 1
            lt_idx = sequence.rfind(' ')
            target_token = sequence[lt_idx+1:].rstrip()
            sequence = sequence[:lt_idx]
            input_ids = tokenizer.encode(sequence, return_tensors='pt')
            next_token_logits = model(input_ids).logits[:, -1, :]
            filtered_next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=50, top_p=1.0)

            # Take highest predicted token not found in stopwords list
            non_sw_found = False
            while not non_sw_found:
                wp_idx = torch.argmax(filtered_next_token_logits)
                word_prediction = tokenizer.decode(wp_idx).strip().lower()
                if word_prediction not in stop_words:
                    non_sw_found = True
                else:
                    filtered_next_token_logits[0][wp_idx] = float('-Inf')

            if target_token == word_prediction:
                corrects += 1
            while target_token.startswith(word_prediction) and target_token != word_prediction:
                tmp_seq = sequence + ' ' + word_prediction
                input_ids = tokenizer.encode(tmp_seq, return_tensors='pt')
                next_token_logits = model(input_ids).logits[:, -1, :]
                filtered_next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=50, top_p=1.0)
                non_sw_found = False
                while not non_sw_found:
                    wp_idx = torch.argmax(filtered_next_token_logits)
                    wp = tokenizer.decode(wp_idx).strip().lower()
                    if wp not in stop_words:
                        non_sw_found = True
                    else:
                        filtered_next_token_logits[0][wp_idx] = float('-Inf')
                word_prediction = word_prediction + wp

            if word_prediction == target_token:
                print('A match!')
            print('{}: Target token: {}\n Predicted token: {}'.format(idx, target_token, word_prediction))
            if idx % 100 == 0:
                print(idx)
    print('Total time for testing on {} samples: {}'.format(sample_count, elapsed_time(time() - start)))
    return corrects / sample_count


if __name__ == '__main__':
    sw = stopwords.words('english')
    sw.append('.'), sw.append(','), sw.append("'"), sw.append("''"), sw.append('"'), sw.append('!'), sw.append('?')
    sw.append(''), sw.append('``'), sw.append('."'), sw.append('*'), sw.append('...'), sw.append(':'), sw.append(';')
    sw.append("'s"), sw.append('`')

    num_params = []
    accuracies = []
    model_names = ['GPT2', 'GPT2-Medium', 'GPT2-Large', 'GPT2-XL', 'GPT-3']

    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    num_params.append(117e6)
    acc = CLM_Lambada_ARM(gpt2_model, gpt2_tokenizer, sw)
    accuracies.append(acc)
    del gpt2_model, gpt2_tokenizer

    gpt2_medium_model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    gpt2_medium_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    num_params.append(345e6)
    acc = CLM_Lambada_ARM(gpt2_medium_model, gpt2_medium_tokenizer, sw)
    accuracies.append(acc)
    del gpt2_medium_model, gpt2_medium_tokenizer

    gpt2_large_model = GPT2LMHeadModel.from_pretrained('gpt2-large')
    gpt2_large_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    num_params.append(774e6)
    acc = CLM_Lambada_ARM(gpt2_large_model, gpt2_large_tokenizer, sw)
    accuracies.append(acc)
    del gpt2_large_model, gpt2_large_tokenizer

    gpt2_xl_model = GPT2LMHeadModel.from_pretrained('gpt2-xl')
    gpt2_xl_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
    num_params.append(1558e6)
    acc = CLM_Lambada_ARM(gpt2_xl_model, gpt2_xl_tokenizer, sw)
    accuracies.append(acc)
    del gpt2_xl_model, gpt2_xl_tokenizer

    num_params.append(175e9)
    accuracies.append(.762)

    plt.figure()
    plt.plot(num_params, accuracies)
    plt.scatter(num_params, accuracies)
    for i in range(len(num_params)):
        plt.annotate(model_names[i], (num_params[i]*1.5, accuracies[i]))
    plt.xscale('log')
    plt.title('GPT-2 Number of Parameters vs. Performance on Lambada Dataset')
    plt.xlabel('Number of Parameters')
    plt.ylabel('Accuracy')
    plt.xlim(right=7e11)
    plt.show()
