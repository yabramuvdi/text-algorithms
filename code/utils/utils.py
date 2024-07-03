import re

def remove_punctuation(sequence, punctuation):
    # remove punctuation symbols and add a whitespace instead         
    regex = re.compile('[%s]' % re.escape(punctuation))
    clean_seq = regex.sub(' ', sequence)
    return clean_seq

# apply preprocessing steps to a sequence
def clean_sequence(seq, punctuation):
    # lowercase
    clean_seq = seq.lower()
    # remove punctuation
    clean_seq = remove_punctuation(clean_seq, punctuation)
    # remove all numbers
    clean_seq =  ''.join([i for i in clean_seq if not i.isdigit()])
    # remove extra white spaces
    clean_seq = re.sub(r"\s+", " ", clean_seq)

    return clean_seq