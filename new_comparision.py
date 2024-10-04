import json
import re
from difflib import SequenceMatcher
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.translate.bleu_score import sentence_bleu
from nltk.util import ngrams
from nltk import download
import matplotlib.pyplot as plt
from Levenshtein import distance as levenshtein_distance

# Download NLTK data files (only the first time)
download('punkt')

def read_transcript(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    dialogues = ' '.join([entry.get('dialogue', '') for entry in data])
    speakers = [entry.get('speaker', '') for entry in data]
    return dialogues, speakers, data

def clean_text(text):
    # Remove extra spaces and normalize text
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def word_level_accuracy(text1, text2):
    words1 = word_tokenize(text1.lower())
    words2 = word_tokenize(text2.lower())

    matcher = SequenceMatcher(None, words1, words2)
    matches = sum(triple.size for triple in matcher.get_matching_blocks())
    total = max(len(words1), len(words2))

    accuracy = (matches / total) * 100
    return round(accuracy, 2), words1, words2

def sentence_level_coherence(text1, text2):
    sentences1 = sent_tokenize(text1)
    sentences2 = sent_tokenize(text2)

    matcher = SequenceMatcher(None, sentences1, sentences2)
    coherence = matcher.ratio() * 100
    return round(coherence, 2), sentences1, sentences2

def speaker_identification(speakers1, speakers2):
    correct = sum(1 for s1, s2 in zip(speakers1, speakers2) if s1 == s2)
    total = max(len(speakers1), len(speakers2))
    accuracy = (correct / total) * 100
    return round(accuracy, 2)

def calculate_wer(reference, hypothesis):
    r = reference.split()
    h = hypothesis.split()
    # Initialize a table
    d = [[0] * (len(h)+1) for _ in range(len(r)+1)]

    # Fill base cases
    for i in range(len(r)+1):
        d[i][0] = i
    for j in range(len(h)+1):
        d[0][j] = j

    # Compute edit distance
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(d[i-1][j] + 1,    # deletion
                          d[i][j-1] + 1,    # insertion
                          d[i-1][j-1] + cost)  # substitution

    wer = d[len(r)][len(h)] / len(r) * 100
    return round(wer, 2)

def specialized_terms_accuracy(text1, text2, terms):
    words1 = set(word_tokenize(text1.lower()))
    words2 = set(word_tokenize(text2.lower()))
    terms = set(terms.lower().split())

    terms_in_text1 = terms & words1
    terms_in_text2 = terms & words2
    correct_terms = terms_in_text1 & terms_in_text2
    total_terms = len(terms)

    accuracy = (len(correct_terms) / total_terms) * 100 if total_terms > 0 else 0
    return round(accuracy, 2), terms_in_text1, terms_in_text2

def compute_levenshtein(text1, text2):
    distance = levenshtein_distance(text1, text2)
    max_len = max(len(text1), len(text2))
    similarity = ((max_len - distance) / max_len) * 100
    return round(similarity, 2)

def main():
    # File paths
    sdk_file = 'sdk_response.json'
    api_file = 'api_response.json'

    # Read and preprocess transcripts
    sdk_text, sdk_speakers, sdk_data = read_transcript(sdk_file)
    api_text, api_speakers, api_data = read_transcript(api_file)

    sdk_text = clean_text(sdk_text)
    api_text = clean_text(api_text)

    # 1. Word-Level Accuracy
    word_accuracy, sdk_words, api_words = word_level_accuracy(sdk_text, api_text)
    print(f"Word-Level Accuracy: {word_accuracy}%")

    # 2. Sentence-Level Coherence
    sentence_coherence, sdk_sentences, api_sentences = sentence_level_coherence(sdk_text, api_text)
    print(f"Sentence-Level Coherence: {sentence_coherence}%")

    # 3. Speaker Identification
    speaker_accuracy = speaker_identification(sdk_speakers, api_speakers)
    print(f"Speaker Identification Accuracy: {speaker_accuracy}%")

    # 4. Handling of Accents or Non-English Words
    # For this example, let's assume we are checking for Hindi words
    hindi_words = ['आपको', 'नमस्ते', 'धन्यवाद']  # Add more Hindi words as needed
    sdk_hindi_words = set(sdk_words) & set(hindi_words)
    api_hindi_words = set(api_words) & set(hindi_words)
    print(f"SDK Hindi Words: {sdk_hindi_words}")
    print(f"API Hindi Words: {api_hindi_words}")

    # 5. Handling of Pauses and Filler Words
    filler_words = ['um', 'ah', 'you know', 'like', 'so']
    sdk_filler_words = [word for word in sdk_words if word.lower() in filler_words]
    api_filler_words = [word for word in api_words if word.lower() in filler_words]
    print(f"SDK Filler Words Count: {len(sdk_filler_words)}")
    print(f"API Filler Words Count: {len(api_filler_words)}")

    # 6. Formatting and Punctuation
    sdk_punctuation = re.findall(r'[^\w\s]', sdk_text)
    api_punctuation = re.findall(r'[^\w\s]', api_text)
    print(f"SDK Punctuation Count: {len(sdk_punctuation)}")
    print(f"API Punctuation Count: {len(api_punctuation)}")

    # 7. Time-Stamp Accuracy
    # Since we don't have the actual audio, we can only compare the timestamps between SDK and API
    sdk_start_times = [entry.get('startTime') for entry in sdk_data]
    api_start_times = [entry.get('startTime') for entry in api_data]
    # Simple comparison of average start times
    sdk_avg_start = sum(sdk_start_times) / len(sdk_start_times) if sdk_start_times else 0
    api_avg_start = sum(api_start_times) / len(api_start_times) if api_start_times else 0
    print(f"SDK Average Start Time: {sdk_avg_start}")
    print(f"API Average Start Time: {api_avg_start}")

    # 8. Handling of Interruptions or Overlapping Speech
    # Since we don't have overlapping indicators, we can check for interleaved speaker turns
    sdk_speaker_changes = sum(1 for i in range(1, len(sdk_speakers)) if sdk_speakers[i] != sdk_speakers[i-1])
    api_speaker_changes = sum(1 for i in range(1, len(api_speakers)) if api_speakers[i] != api_speakers[i-1])
    print(f"SDK Speaker Changes: {sdk_speaker_changes}")
    print(f"API Speaker Changes: {api_speaker_changes}")

    # 9. Contextual Understanding
    # This is qualitative; however, we can check BLEU score as an approximation
    reference = sdk_text.split()
    hypothesis = api_text.split()
    bleu_score = sentence_bleu([reference], hypothesis) * 100
    print(f"Contextual BLEU Score: {round(bleu_score, 2)}%")

    # 10. Error Rate Comparison (WER)
    wer_sdk_api = calculate_wer(sdk_text, api_text)
    print(f"WER (SDK vs API): {wer_sdk_api}%")

    # 11. Handling of Specialized Terms and Jargon
    specialized_terms = 'AI intern internship developer frontend backend GPT OpenAI LLM Replit'
    terms_accuracy, sdk_terms, api_terms = specialized_terms_accuracy(sdk_text, api_text, specialized_terms)
    print(f"Specialized Terms Accuracy: {terms_accuracy}%")
    print(f"SDK Terms Found: {sdk_terms}")
    print(f"API Terms Found: {api_terms}")

    # 12. Speaker Turn Accuracy
    # Assuming speakers are correctly labeled
    speaker_turn_accuracy = speaker_identification(sdk_speakers, api_speakers)
    print(f"Speaker Turn Accuracy: {speaker_turn_accuracy}%")

    # 13. Noise and Background Sound Handling
    # Since we don't have audio data, we cannot assess this directly
    print("Noise and Background Sound Handling: Cannot assess without audio data.")

    # 14. Real-Time Performance
    # Not applicable in this context
    print("Real-Time Performance: Not applicable for offline transcripts.")

    # 15. Aggregate Similarity and Difference Analysis
    levenshtein_similarity = compute_levenshtein(sdk_text, api_text)
    print(f"Aggregate Levenshtein Similarity: {levenshtein_similarity}%")

    # Visualization (Optional)
    # You can create bar charts or other visualizations here if needed.

if __name__ == '__main__':
    main()
