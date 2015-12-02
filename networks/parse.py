""" Apply the Stanford Parser to command-line arguments.

Accepts a series of quoted sentences or paragraphs (which are
segmented into sentences) and returns all parses of each sentence
in Chomsky Normal Form. The first line of the i-th section contains,
n_i, the number of parses for the i-th input sentence, followed by the
sentence itself, followed by n_i lines, each containing a different parse.
The next section starts one the line that comes after the n_i-th parse.
"""

import os
import argparse
from nltk.parse import stanford
from nltk.treetransforms import chomsky_normal_form, collapse_unary
from nltk.tokenize import word_tokenize, sent_tokenize

arg_parser = argparse.ArgumentParser(
    description='Parse sentences with StanfordParser.')
arg_parser.add_argument(
    'sentences', nargs='+', default=[],
    help="The sentences to parse.")

args = arg_parser.parse_args()

parser_home = '/home/eric/Dropbox/classes/comp599/project/stanford_parser/'

os.environ['STANFORD_PARSER'] = os.path.join(
    parser_home, 'stanford-parser.jar')

os.environ['STANFORD_MODELS'] = os.path.join(
    parser_home, 'stanford-parser-3.5.2-models.jar')

sentences = []
for sentence in args.sentences:
    sentences.extend(sent_tokenize(sentence))


def preprocess(s):
    s = word_tokenize(s)
    s = filter(lambda t: t.isalnum(), s)
    s = map(lambda t: t.lower(), s)
    return ' '.join(s)

sentences = map(preprocess, sentences)

parser = stanford.StanfordParser(
    model_path=os.path.join(parser_home, "englishPCFG.ser.gz"))
parsed_sentences = parser.raw_parse_sents(sentences)

verbosity = 0

for sentence, parses in zip(sentences, parsed_sentences):
    parses = list(parses)
    print len(parses)
    print sentence

    for parse in parses:
        if verbosity > 0:
            print "*" * 80
            print "Original: \n", parse

        chomsky_normal_form(parse)

        if verbosity > 0:
            print "Chomskied: \n", parse

        collapse_unary(parse, collapsePOS=True)

        parse = parse[0]  # Remove root
        result = [s.strip() for s in str(parse).split('\n')]
        result = ' '.join(result)
        print result
