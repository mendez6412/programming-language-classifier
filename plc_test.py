from plc import read_prog_files, get_language
from extensions import extensions
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer as tuff
from sklearn.pipeline import Pipeline


samples = []
labels = []

for ext, name in extensions.items():
    x = read_prog_files('bmgame/bmgame/bench/**/*.{}'.format(ext))
    samples += x
    y = len(x) * [name]
    labels += y


pip = Pipeline([('cv', CountVectorizer(analyzer='word', token_pattern=r'[a-zA-Z]{2,}|\s{2,}|[^\w\d\s]')), ('bay', MultinomialNB())])
pip.fit(samples, labels)
pip.score(samples, labels)


def test_read_program_files():
    unknown = []
    for item in range(1, 33):
        testing = read_prog_files('test/{}'.format(item))
        unknown += testing
    assert len(unknown) == 32


def test_get_language():
    test = get_language('numpy')
    assert test in labels
