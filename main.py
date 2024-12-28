import os
import itertools
import random
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim
from gensim import corpora
from gensim.models import Phrases, CoherenceModel, TfidfModel
from gensim.models.phrases import Phraser
import psycopg2
from telethon import TelegramClient
from telethon.errors import FloodWaitError, SessionPasswordNeededError
from telethon.tl.functions.contacts import SearchRequest
from telethon.tl.types import InputPeerEmpty
import nest_asyncio
import asyncio
import time
from langdetect import detect, DetectorFactory, LangDetectException
from datasets import load_dataset
from itertools import cycle
from collections import defaultdict

# Ensure consistent results from langdetect
DetectorFactory.seed = 0

# Apply nested asyncio to allow running inside Jupyter or similar environments
nest_asyncio.apply()

# ---------------------- Configuration ---------------------- #

# Telethon configuration
SESSIONS_FOLDER = 'sessions'  # Folder containing .session files
API_ID = 1234567               # Replace with your actual API ID
API_HASH = "abcdef1234567890abcdef1234567892"  # Replace with your actual API Hash

# PostgreSQL configuration
DB_NAME = "propaganda"
DB_USER = "posts_owner"
DB_PASSWORD = "jYw1bfDnOHW2"
DB_HOST = "ep-holy-glitter-a287uyrp.eu-central-1.aws.neon.tech"
DB_PORT = 5432

# Search configuration
NUM_PHRASES_PER_ITERATION = 5
RELATED_PROBABILITY = 0.7        # 70% related phrases
SINGLE_WORD_PERCENTAGE = 0.7     # 70% single-word phrases (increased importance)
SLEEP_INTERVAL = 30              # Seconds between iterations

# Specific word/topic (optional)
SPECIFIC_WORD = None              # e.g., "market"
SPECIFIC_TOPIC = None             # e.g., "Topic_1"

# Percentage of Wikipedia data to load
DATA_PERCENTAGE = 0.001  # 0.1% of the dataset

# TF-IDF configuration
TOP_TFIDF_WORDS = 5000  # Number of top TF-IDF words to include

# ---------------------- Preprocessing ---------------------- #

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text.lower())
    tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token.isalpha() and token not in stop_words
    ]
    return tokens

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# ---------------------- Load Wikipedia Dataset ---------------------- #

print("Loading Wikipedia dataset...")
NUM_SAMPLES = 800  # Adjust this number as needed

# Load the 'train' split
wiki_dataset = load_dataset('wikipedia', '20220301.en', split='train')

# Shuffle the dataset with a random seed for randomness each run
wiki_dataset = wiki_dataset.shuffle(seed=random.randint(0, 10000))  # Different seed each run

# Select the desired number of samples
wiki_dataset = wiki_dataset.select(range(NUM_SAMPLES))

# Extract the 'text' field from the dataset
documents = wiki_dataset['text']

# Preprocess the documents
print("Preprocessing documents...")
processed_docs = [preprocess_text(doc) for doc in documents]
print("Preprocessing completed.")

# ---------------------- Create Bi-grams ---------------------- #

print("Creating bi-grams...")
bigram = Phrases(processed_docs, min_count=5, threshold=100)
bigram_mod = Phraser(bigram)
processed_docs = [bigram_mod[doc] for doc in processed_docs]
print("Bi-grams created.")

# ---------------------- Create Dictionary and Corpus ---------------------- #

print("Creating dictionary and corpus...")
dictionary = corpora.Dictionary(processed_docs)
dictionary.filter_extremes(no_below=5, no_above=0.7, keep_n=100000)  # Increased keep_n for larger vocabulary
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
print("Dictionary and corpus created.")

# ---------------------- TF-IDF Model ---------------------- #

print("Training TF-IDF model...")
tfidf_model = TfidfModel(bow_corpus, dictionary=dictionary)
corpus_tfidf = tfidf_model[bow_corpus]
print("TF-IDF model trained.")

# ---------------------- Extract Top TF-IDF Words ---------------------- #

print(f"Extracting top {TOP_TFIDF_WORDS} TF-IDF words...")
word_tfidf = defaultdict(float)

for doc in corpus_tfidf:
    for word_id, score in doc:
        word_tfidf[word_id] += score

# Sort words by TF-IDF score in descending order
sorted_words = sorted(word_tfidf.items(), key=lambda item: item[1], reverse=True)

# Select top N words
top_tfidf_word_ids = [word_id for word_id, score in sorted_words[:TOP_TFIDF_WORDS]]
tfidf_keywords = [dictionary[word_id] for word_id in top_tfidf_word_ids]
print(f"Extracted {len(tfidf_keywords)} TF-IDF keywords.")

# ---------------------- LDA Model ---------------------- #

num_topics = 20  # Fewer topics for speed
passes = 10      # Fewer passes for speed

print("Training LDA model...")
lda_model = gensim.models.LdaMulticore(
    bow_corpus,
    num_topics=num_topics,
    id2word=dictionary,
    passes=passes,
    workers=4,          # Number of CPU cores
    random_state=42
)
print("LDA model trained.")

def extract_top_keywords(lda_model, num_keywords=15):
    topics = {}
    for idx, topic in lda_model.show_topics(formatted=False, num_words=num_keywords):
        keywords = [word for word, _ in topic]
        topics[f"Topic_{idx+1}"] = keywords
    return topics

keyword_pools = extract_top_keywords(lda_model, num_keywords=15)  # Increased keywords per topic

print("Keyword Pools from LDA:\n")
for topic, keywords in keyword_pools.items():
    print(f"{topic}: {', '.join(keywords)}\n")

# Combine TF-IDF keywords with LDA keywords
combined_single_words = list(set(tfidf_keywords + [word for keywords in keyword_pools.values() for word in keywords]))
print(f"Total single words after combining TF-IDF and LDA keywords: {len(combined_single_words)}")

# ---------------------- Evaluate Topic Coherence ---------------------- #

print("Evaluating topic coherence...")
coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_docs, dictionary=dictionary, coherence='c_v')
coherence_score = coherence_model_lda.get_coherence()
print(f'Coherence Score: {coherence_score:.4f}')

# ---------------------- Database Connection ---------------------- #

def connect_db():
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    return conn

# ---------------------- Load Multiple Telethon Clients ---------------------- #

def load_telethon_clients(sessions_folder):
    session_files = [f for f in os.listdir(sessions_folder) if f.endswith('.session')]
    clients = []
    for session in session_files:
        session_path = os.path.join(sessions_folder, session)
        client = TelegramClient(session_path, API_ID, API_HASH)
        clients.append(client)
    if not clients:
        raise ValueError("No session files found in the specified folder.")
    return clients  # Return list instead of cycle

clients = load_telethon_clients(SESSIONS_FOLDER)

# ---------------------- Client State Management ---------------------- #

class ClientState:
    def __init__(self, client):
        self.client = client
        self.available = True
        self.wait_until = None  # Timestamp when the client becomes available

# Initialize client states
client_states = [ClientState(client) for client in clients]

# ---------------------- Utility Functions ---------------------- #

def get_topic_for_word(lda_model, dictionary, word):
    if word not in dictionary.token2id:
        print(f"'{word}' not found in the vocabulary.")
        return None
    word_id = dictionary.token2id[word]
    word_probs = lda_model.get_topics()[:, word_id]
    top_topic = word_probs.argmax()
    return f"Topic_{top_topic +1}"

def generate_search_phrase(keyword_pools, combined_single_words, related_probability=0.5, single_word_percentage=0.7, specific_word=None, specific_topic=None):
    """
    Generates a search phrase by selecting keywords or single words.

    Args:
        keyword_pools (dict): Dictionary with topics as keys and keyword lists as values.
        combined_single_words (list): List of single words from TF-IDF and LDA.
        related_probability (float): Probability of generating a related phrase.
        single_word_percentage (float): Percentage of outputs that should be single words (0 to 1).
        specific_word (str, optional): Specific word to base the phrase on.
        specific_topic (str, optional): Specific topic to base the phrase on.

    Returns:
        str: Generated search phrase.
    """
    is_single_word = random.random() < single_word_percentage

    if specific_word:
        topic = get_topic_for_word(lda_model, dictionary, specific_word)
        if topic and topic in keyword_pools:
            if is_single_word:
                return specific_word
            keywords = random.sample(keyword_pools[topic], 2)
            return " ".join(keywords)
        else:
            print(f"Unable to find a topic for the word '{specific_word}'. Falling back to random generation.")

    if specific_topic:
        if specific_topic in keyword_pools:
            if is_single_word:
                return random.choice(keyword_pools[specific_topic])
            keywords = random.sample(keyword_pools[specific_topic], 2)
            return " ".join(keywords)
        else:
            print(f"'{specific_topic}' is not a valid topic. Falling back to random generation.")

    if is_single_word:
        return random.choice(combined_single_words)
    else:
        if random.random() < related_probability:
            selected_topic = random.choice(list(keyword_pools.keys()))
            if len(keyword_pools[selected_topic]) < 2:
                return random.choice(keyword_pools[selected_topic])
            keywords = random.sample(keyword_pools[selected_topic], 2)
            return " ".join(keywords)
        else:
            num_keywords = random.randint(1, 3)
            selected_keywords = []
            for _ in range(num_keywords):
                selected_topic = random.choice(list(keyword_pools.keys()))
                keyword = random.choice(keyword_pools[selected_topic])
                selected_keywords.append(keyword)
            return " ".join(selected_keywords)

def detect_language(text):
    """
    Detects the language of the given text using langdetect.

    Args:
        text (str): Text to detect language for.

    Returns:
        str: ISO 639-1 language code or 'unknown'.
    """
    try:
        language = detect(text)
        return language
    except LangDetectException:
        return 'unknown'

# ---------------------- Main Async Function ---------------------- #

async def fetch_and_detect_language(client, chat_id):
    """
    Fetches the last 20 messages from a chat and detects the predominant language.

    Args:
        client (TelegramClient): Telethon client instance.
        chat_id (int): Telegram chat ID.

    Returns:
        str: Detected language code or 'unknown'.
    """
    try:
        messages = await client.get_messages(chat_id, limit=20)
        text = ' '.join([msg.text for msg in messages if msg.text])
        if not text.strip():
            return 'unknown'
        language = detect(text)
        return language
    except LangDetectException:
        return 'unknown'
    except Exception as e:
        print(f"Error fetching messages for chat_id {chat_id}: {e}")
        return 'unknown'

async def search_and_store_phrases():
    conn = connect_db()
    cursor = conn.cursor()

    while True:
        # Check available clients
        available_clients = [cs for cs in client_states if cs.available]
        if not available_clients:
            # All clients are in FloodWait, find the minimum wait time
            current_time = time.time()
            wait_times = [cs.wait_until - current_time for cs in client_states if cs.wait_until]
            if not wait_times:
                wait_duration = SLEEP_INTERVAL
            else:
                wait_duration = max(0, min(wait_times))
            print(f"All clients are in FloodWait. Sleeping for {wait_duration} seconds.")
            await asyncio.sleep(wait_duration)
            continue

        current_client_state = available_clients[0]
        current_client = current_client_state.client

        # Disconnect any other connected clients
        for cs in client_states:
            if cs.client != current_client and cs.client.is_connected():
                await cs.client.disconnect()

        # Start the current client
        try:
            await current_client.start()
            print(f"Using session: {current_client.session.filename}")
        except Exception as e:
            print(f"Error starting client {current_client.session.filename}: {e}")
            current_client_state.available = False
            current_client_state.wait_until = time.time() + SLEEP_INTERVAL
            continue

        # Select a random word from the combined single words
        random_word = random.choice(combined_single_words)
        print(f"\nSelected Word for this iteration: '{random_word}'\n")

        # Generate search phrases based on the random word
        phrases = []
        for _ in range(NUM_PHRASES_PER_ITERATION):
            phrase = generate_search_phrase(
                keyword_pools,
                combined_single_words,
                related_probability=RELATED_PROBABILITY,
                single_word_percentage=SINGLE_WORD_PERCENTAGE,
                specific_word=random_word,
                specific_topic=None
            )
            phrases.append(phrase)

        print("Generated Search Phrases:")
        for p in phrases:
            print(f"- {p}")

        # Search Telegram groups for each phrase and phrase + " group"
        for phrase in phrases:
            search_variants = [phrase, f"{phrase} group"]  # Original and with "group"

            for search_phrase in search_variants:
                print(f"\nSearching for phrase: '{search_phrase}'")
                try:
                    result = await current_client(SearchRequest(
                        q=search_phrase,
                        limit=100
                    ))

                    for chat in result.chats:
                        if hasattr(chat, 'title') and getattr(chat, 'megagroup', False):
                            group_name = chat.title
                            chat_id = chat.id
                            invite_link = getattr(chat, 'invite_link', None)

                            # Fetch and detect language from last 20 messages
                            language = await fetch_and_detect_language(current_client, chat_id)

                            # Insert into the database
                            try:
                                cursor.execute("""
                                    INSERT INTO telegram_groups (chat_id, group_name, invite_link, language)
                                    VALUES (%s, %s, %s, %s)
                                    ON CONFLICT (chat_id) DO NOTHING;
                                """, (chat_id, group_name, invite_link, language))
                                conn.commit()
                                print(f"Inserted Group: {group_name} (ID: {chat_id}, Language: {language})")
                            except Exception as db_err:
                                print(f"Database error: {db_err}")
                                conn.rollback()
                except FloodWaitError as e:
                    print(f"FloodWaitError on client {current_client.session.filename}: Must wait for {e.seconds} seconds.")
                    current_client_state.available = False
                    current_client_state.wait_until = time.time() + e.seconds
                except Exception as e:
                    print(f"Error searching for phrase '{search_phrase}': {e}")

        # Disconnect the current client after use
        await current_client.disconnect()

        print(f"Iteration complete. Sleeping for {SLEEP_INTERVAL} seconds.\n")
        await asyncio.sleep(SLEEP_INTERVAL)

    cursor.close()
    conn.close()

# ---------------------- Run the Async Function ---------------------- #

if __name__ == "__main__":
    asyncio.run(search_and_store_phrases())