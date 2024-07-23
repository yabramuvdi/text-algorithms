#%%

import pandas as pd
import re

#%%

class Dictionary:
    """Handles a dictionary search of terms in strings with great
       flexibilit through the use of regular expressions
    
    Attributes:
        terms: List of strings containing the terms from the dictionary
        part_of_word: Boolean list of the same size as 'terms' specifying if
            each term should be searched as a standalone token or as part of a
            longer word
        ignore_case: Boolean to define if casing of terms is ignored
        flexible_multi_word: Boolean to allow for many possible separators in 
            multi-word dictionary terms (e.g. 'remote work')
        search_type: One of the following strings
            'all': searches for the appereance of all dictionary terms
            'first': searches only for the first appereance of the dictionary terms
        return_matches: Boolean specifying if the matched terms should be returned
        
    """

    def __init__(self, 
                 terms, 
                 part_of_word,
                 ignore_case, 
                 flexible_multi_word, 
                 search_type, 
                 return_matches):
        """Initialize the dictionary based on the user's preferences
        """
        self.terms = terms
        self.part_of_word = part_of_word
        self.ignore_case = ignore_case
        self.flexible_multi_word = flexible_multi_word
        self.search_type = search_type
        self.return_matches = return_matches
        self.dictionary_info = self.build_dictionary_info()
        
    def build_dictionary_info(self):
        """
        Create a pandas DataFrame with 'terms' and 'part_of_word' as columns.
        Adds two more columns:
        - 'is_single_word': Boolean indicating if the term is a single word.
        - 'length_order': Integer specifying the order based on the number of characters in the term.

        Returns:
            pd.DataFrame: The resulting DataFrame.
        """
        # Create the initial DataFrame
        df = pd.DataFrame({
            'terms': self.terms,
            'part_of_word': self.part_of_word
        })

        df['is_single_word'] = df['terms'].apply(lambda x: len(x.split()) == 1)
        df['length_order'] = df['terms'].apply(len)
        df = df.sort_values(by='length_order').reset_index(drop=True)

        return df

    def gen_multiple_word_regex(self, text):
        """Craft a regular expression that matches any combination of hyphens 
           and white spaces separating the words in a dictionary term
        """
        components = text.split(" ")
        term_regex = r"\b"
        for i, c in enumerate(components):
            if i < len(components)-1:
                term_regex += c + r"[- ]+"
            else:
                term_regex += c + r"\b"  
        return term_regex

    def gen_dict_regex(self):
        """Generates a single regular expression that matches all terms 
           according to the user's preferences.

        Returns:
            str: The regular expression to be used
        """
        
        dict_regex = ""
        for row in self.dictionary_info.itertuples():
            if row.is_single_word:
                if row.part_of_word:
                    term = r"\b" + row.terms
                else:
                    term = r"\b" + row.terms + r"\b"

                dict_regex += term + r"|"
            
            else:
                dict_regex += self.gen_multiple_word_regex(row.terms) + r"|"
        
        if self.ignore_case:
            dict_regex = re.compile(dict_regex[:-1], re.IGNORECASE)
        else:
            dict_regex = re.compile(dict_regex[:-1])

        #return dict_regex
        self.dict_regex = dict_regex


    def tag_text(self, text):
        """Searches for the dictionary terms within a given string

        Args:
            text (str): The text in which to search for the dictionary terms.

        Returns:
            tuple: A tuple containing (i) a boolean that indicates if any of the 
                dictionary terms was found in the text and (ii) a list with the
                the matched string and its start and end position.

        """
        
        if self.search_type == "first":
            match = self.dict_regex.search(text)
            if match:
                found_match = True
                if self.return_matches:
                    term_matches = [(match.group(0), (match.start(), match.end()))]
                else:
                    term_matches = []
            else:
                found_match = False
                term_matches = []

            return found_match, term_matches
        
        elif self.search_type == "all":
            matches = list(self.dict_regex.finditer(text))
            if matches:
                found_match = True
                if self.return_matches:
                    term_matches = [(match.group(0), (match.start(), match.end())) for match in matches]
                else:
                    term_matches = []
            else:
                found_match = False
                term_matches = []
            
            return found_match, term_matches


#%%

# #============================
# # TEST
# #============================

# df = pd.read_csv("../FED_classification.csv", sep="\t")
# hawk_dict = pd.read_csv("../dictionaries/hawkish.csv")
# hawk_dict["part"] = hawk_dict["part"].astype(bool)

# #%%

# my_dict = Dictionary(list(hawk_dict["term"].values), 
#                      list(hawk_dict["part"].values),
#                      flexible_multi_word=True,
#                      search_type="first",
#                      return_matches=True
#                      )
          
# my_dict.gen_dict_regex()

# #%%

# text1 = "Monetary policy has hiked and increased a lot in the liftoff"
# text2 = "Such trends could foster inflationary imbalances that would undermine the economy's exemplary performance"
# text3 = "Hola"

# for text in [text1, text2, text3]:
#     print(text)
#     print(my_dict.tag_text(text))
#     print("======================\n")


# #%%

# # apply to all text of a pandas dataframe
# results = df["text"].apply(my_dict.tag_text)
# hawkish_boolean = [match[0] for match in results]
# hawkish_terms = [match[1] for match in results]
# df["hawkish"] = hawkish_boolean
# df["hawkish_matches"] = hawkish_terms

# #%%
