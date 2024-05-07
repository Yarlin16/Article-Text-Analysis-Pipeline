import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import chardet
import os

#import os
print("Current Working Directory:", os.getcwd())

def extract_and_save_content(url, url_id):
    file_path = f"{url_id}.txt"  # Define the file path based on url_id
    try:
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        
        content = soup.find(attrs={'class': 'td-post-content tagdiv-type'})
        if not content:
            content = soup.find(attrs={'class': 'td_block_wrap tdb_single_content tdi_130 td-pb-border-top td_block_template_1 td-post-content tagdiv-type'})
        
        if content:
            clean_content = content.text.replace('\n', "")
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(clean_content)
            print(f"Content saved for {url_id}")
        else:
            # If no content found, create an empty file
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write("")  # Write an empty string to the file
            print(f"No content found for {url_id}, empty file created")

    except Exception as e:
        print(f"Error occurred while processing {url_id}: {e}")

def remove_stop_words_from_file(file_path, stop_words_file_paths):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    for stop_words_file_path in stop_words_file_paths:
        # Detect file encoding using chardet
        with open(stop_words_file_path, 'rb') as stop_file:
            raw_data = stop_file.read()
            encoding = chardet.detect(raw_data)['encoding']

        # Read stop words file with detected encoding
        with open(stop_words_file_path, 'r', encoding=encoding) as stop_file:
            stop_words = stop_file.read().split()

        # Remove stop words from text
        text = ' '.join(word for word in text.split() if word.lower() not in stop_words)

    # Clean text from special characters
    cleaned_text = clean_special_characters(text)

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(cleaned_text)
    
    print(f"Content cleaned and saved for {file_path}")
    return cleaned_text


def clean_special_characters(text):
    # Define patterns to remove specific special characters
    patterns = [
        (r'â|â', '"'),  # Replace â and â with regular double quotes
        (r'Â', ''),         # Remove Â
        # Add more patterns as needed to handle other special characters
    ]

    cleaned_text = text
    for pattern, replacement in patterns:
        cleaned_text = re.sub(pattern, replacement, cleaned_text)

    return cleaned_text

def calculate_sentiment_scores(file_path, positive_words_set, negative_words_set):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Tokenize the text using NLTK
    tokens = word_tokenize(text.lower())

    # Calculate Positive and Negative scores
    positive_score = sum(1 for word in tokens if word in positive_words_set)
    negative_score = sum(1 for word in tokens if word in negative_words_set)

    # Calculate Polarity Score
    polarity_score = (positive_score - negative_score) / (positive_score + negative_score + 0.000001)

    # Calculate Subjectivity Score
    subjectivity_score = (positive_score + negative_score) / (len(tokens) + 0.000001)

    return positive_score, negative_score, polarity_score, subjectivity_score

def calculate_readability_metrics(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Tokenize text into sentences
    sentences = sent_tokenize(text)

    # Calculate total number of words and total number of sentences
    words = word_tokenize(text.lower())
    total_words = len(words)
    total_sentences = len(sentences)

    # Calculate average number of words per sentence
    if total_sentences > 0:
        avg_words_per_sentence = total_words / total_sentences
    else:
        avg_words_per_sentence = 0

    # Calculate Average Sentence Length
    if total_sentences > 0:
        avg_sentence_length = total_words / total_sentences
    else:
        avg_sentence_length = 0

    # Calculate Percentage of Complex Words (words with more than 2 syllables)
    complex_words = [word for word in words if syllables(word) > 2]
    percentage_complex_words = len(complex_words) / total_words if total_words > 0 else 0

    # Calculate Fog Index
    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

    return avg_words_per_sentence, avg_sentence_length, percentage_complex_words, fog_index

def calculate_complex_word_count(text):
    words = word_tokenize(text.lower())
    complex_word_count = sum(1 for word in words if syllables(word) > 2)
    return complex_word_count

def calculate_total_word_count(text):
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    cleaned_words = [word for word in words if word not in stop_words and word.isalpha()]
    total_word_count = len(cleaned_words)
    return total_word_count

def calculate_syllable_count_per_word(text):
    words = word_tokenize(text.lower())
    syllable_count = sum(syllables(word) for word in words)
    return syllable_count

def calculate_personal_pronouns_count(text):
    personal_pronouns_regex = r'\b(?:I|we|my|our|us)\b'
    personal_pronouns = re.findall(personal_pronouns_regex, text, flags=re.IGNORECASE)
    return len(personal_pronouns)

def calculate_average_word_length(text):
    words = word_tokenize(text.lower())
    total_characters = sum(len(word) for word in words)
    total_words = len(words)
    if total_words > 0:
        average_word_length = total_characters / total_words
    else:
        average_word_length = 0
    return average_word_length

def syllables(word):
    vowels = 'aeiouy'
    count = 0
    last_char = ''
    word = word.lower()
    
    for char in word:
        if char in vowels and last_char not in vowels:
            count += 1
        last_char = char
        
    if word.endswith(('es', 'ed')):  # Handling exceptions
        count -= 1
        
    return max(1, count)  # At least one syllable per word


if __name__ == '__main__':
    file_path_excel = 'Input.xlsx'
    df = pd.read_excel(file_path_excel)

    # Detect file encoding for positive words
    with open('positive-words.txt', 'rb') as f:
        rawdata = f.read()
    encoding = chardet.detect(rawdata)['encoding']

    # Read positive words file with detected encoding
    with open('positive-words.txt', 'r', encoding=encoding) as pos_file:
        positive_words = pos_file.read().split()
    positive_words_set = set(positive_words)
    
    # Detect file encoding for negative words
    with open('negative-words.txt', 'rb') as f:
        rawdata = f.read()
    encoding = chardet.detect(rawdata)['encoding']

    # Read negative words file with detected encoding
    with open('negative-words.txt', 'r', encoding=encoding) as neg_file:
        negative_words = neg_file.read().split()
    negative_words_set = set(negative_words)

    output_data = []  # List to store output data

    for index, row in df.iterrows():
        url_id = row['URL_ID']
        url = row['URL']

        # Extract and save content from the URL
        extract_and_save_content(url, url_id)

        # Clean text file by removing stop words and special characters
        file_path = f"{url_id}.txt"
        stop_words_files = ['StopWords_Auditor.txt', 'StopWords_Currencies.txt', 'StopWords_DatesandNumbers.txt', 
                            'StopWords_Generic.txt', 'StopWords_GenericLong.txt', 'StopWords_Geographic.txt', 
                            'StopWords_Names.txt']

        cleaned_text = remove_stop_words_from_file(file_path, stop_words_files)

        # Calculate sentiment scores
        positive_score, negative_score, polarity_score, subjectivity_score = calculate_sentiment_scores(
            file_path, positive_words_set, negative_words_set)

        # Calculate readability metrics
        avg_words_per_sentence, avg_sentence_length, percentage_complex_words, fog_index = calculate_readability_metrics(
            file_path)

        # Calculate complex word count
        complex_word_count = calculate_complex_word_count(cleaned_text)

        # Calculate total word count (after cleaning)
        total_word_count = calculate_total_word_count(cleaned_text)

        # Calculate syllable count per word
        syllable_count_per_word = calculate_syllable_count_per_word(cleaned_text)

        # Calculate personal pronouns count
        personal_pronouns_count = calculate_personal_pronouns_count(cleaned_text)

        # Calculate average word length
        average_word_length = calculate_average_word_length(cleaned_text)

        # Append data to output_data list
        output_data.append([
            url_id, url, positive_score, negative_score, polarity_score, subjectivity_score,
            avg_sentence_length, percentage_complex_words, fog_index, avg_words_per_sentence,
            complex_word_count, total_word_count, syllable_count_per_word, personal_pronouns_count,
            average_word_length
        ])

    # Create DataFrame from output_data and save to Excel
    columns = [
        'URL_ID', 'URL', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE', 'SUBJECTIVITY SCORE',
        'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS', 'FOG INDEX', 'AVG NUMBER OF WORDS PER SENTENCE',
        'COMPLEX WORD COUNT', 'WORD COUNT', 'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH'
    ]
    output_df = pd.DataFrame(output_data, columns=columns)

    # Save DataFrame to Excel
    output_excel_path = 'Output Data Structure.xlsx'
    output_df.to_excel(output_excel_path, index=False)

    print("All text files processed successfully.")
    print(f"Output data saved to {output_excel_path}")