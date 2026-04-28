import re
import string

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
except Exception:  # pragma: no cover - fallback for broken local nltk install
    nltk = None
    stopwords = None
    WordNetLemmatizer = None


FALLBACK_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", "was",
    "were", "will", "with",
}


def _ensure_nltk_resource(resource_name, download_name):
    """Download the NLTK resource once if it is missing."""
    if nltk is None:
        raise LookupError("nltk is unavailable")
    try:
        nltk.data.find(resource_name)
    except LookupError:
        nltk.download(download_name, quiet=True)

def lemmatization(text):
    """Lemmatize the text."""
    if WordNetLemmatizer is None:
        return text
    _ensure_nltk_resource("corpora/wordnet", "wordnet")
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    """Remove stop words from the text."""
    try:
        _ensure_nltk_resource("corpora/stopwords", "stopwords")
        stop_words = set(stopwords.words("english"))
    except LookupError:
        stop_words = FALLBACK_STOPWORDS
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub("[%s]" % re.escape(string.punctuation), " ", text)
    text = text.replace("؛", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(text):
    """Normalize the text data."""

    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)

    return text