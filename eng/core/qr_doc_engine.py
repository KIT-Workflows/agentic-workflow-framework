from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json, random, ast, re
from IPython.display import display, Markdown
import numpy as np
from typing import Callable, Any, List, Type, Dict, Union
from dataclasses import dataclass

from ..assets.assets_collection import PERC1, PERC3
from ..assets.text_prompt import prompt_answer_instruct, default_prompt_answer_template, markdown_text, keywords_prompt
from ..interfaces.llms_engine import LMEngine, count_tokens, quick_chat, a_quick_chat
from ..interfaces.lil_wrapper import process_output_to_str


with open('eng/assets/xqe_univ_kg_load_v1.json', 'r') as f:
    xqe_kg_load = json.load(f)


needed_keys = ['namelist', 'Card_Name','Parameter_Name', 'Description', 'Usage_Conditions',
               'Relationships_Conditions_to_Other_Parameters_Cards',
                'Possible_Usage_Conditions', 'Parameter_Value_Conditions', 'Final_comments']
tot_corpus = []
tot_corpus_raw = []
for itm in xqe_kg_load:

    tmp_ = {k: v for k,v in itm.items() if k in needed_keys and v != ''}
    tot_corpus.append(process_output_to_str(tmp_))
    tot_corpus_raw.append(itm)
tot_corpus = np.array(tot_corpus, dtype='str')

corpus_len = np.array([len(i.split()) for i in tot_corpus])
factor = corpus_len / np.sum(corpus_len)


class QrDocInstance:
    def __init__(self, lm_api: str, 
                 kw_model_name: str,
                 chat_model_name: str,
                 MAXDocToken: int = 9000, 
                 tot_corpus: list = tot_corpus, 
                 tot_corpus_raw: list = tot_corpus_raw
                 ):
        self.kw_model_name = kw_model_name
        self.chat_model_name = chat_model_name
        self.corpus = tot_corpus
        self.corpus_raw = tot_corpus_raw
        self.factor = factor
        self.lm_api = lm_api
        self.doc_ans = default_prompt_answer_template
        self.doc_ans_instruct = prompt_answer_instruct
        self.MAXDocToken = MAXDocToken
        self.markdown_text = markdown_text
        self.llm_call_output = ''

    @staticmethod
    def collect_data( query, corpus = tot_corpus, 
                    factor = factor, n_features = 17, 
                    token_pattern_ = r"(?u)\b\w\w+\b" 
                    ):
        var_coll = []
        data_coll = []
        #for feat in range(min, max):
        corpus = [str(i) for i in corpus]
        vectorizer = HashingVectorizer(n_features=2**n_features,  
                                        analyzer='word', 
                                        token_pattern=token_pattern_,
                                        )
        X = vectorizer.fit_transform(corpus)
        X2 = vectorizer.transform([query])
        # similarity between X2 and X
        sim = cosine_similarity(X2, X).flatten() 
        sim = sim / (np.sum(sim) + 1e-5)
        sim = sim**(2)/2 * -np.log(factor)
        sim = sim / (np.sum(sim) + 1e-5)
        var_coll.append([np.std(sim), sim.max(), sim.mean(), sim.min() ])
        data_coll.append([ sim, vectorizer])
        return var_coll, data_coll

    @staticmethod
    def detect_outliers_iqr(data):
        q1 = np.percentile(data, PERC1)
        q3 = np.percentile(data, PERC3)
        iqr = q3 - q1
        lower_bound = q1 
        upper_bound = q3 
        j_ = []
        [j_.append(x) for x in data if x <= lower_bound ]
        k_ = []
        [k_.append(x) for x in data if x >= upper_bound ]

        return np.max(j_), np.min(k_)

    @staticmethod
    def print_whats_going_on(coll_slice, corpus = tot_corpus, print_out = False):
        _, data_coll = coll_slice

        sim_coll = data_coll[0][0]
        if all(sim_coll == 0):
            print('All zeros')
            return None, None
        
        out_out = []
        doc_id = []
        non_zero_id = np.where(sim_coll  > 0 )[0]
        sim_coll_nz = sim_coll[non_zero_id]
        threshold = QrDocInstance.detect_outliers_iqr(sim_coll_nz) 
        retrieved_id = np.where(sim_coll  >= threshold[1] )[0]

        for ki in retrieved_id: 
            if print_out:
                print(ki, sim_coll[ki])
                print()
                print(corpus[ki])
                print('####################')

            doc_id.append(ki)
            out_out.append([ki, sim_coll[ki], corpus[ki]])

        return doc_id, out_out


    @staticmethod
    async def a_b_from_query(query: Union[str, List[str]],
                    corpus: List[str] = tot_corpus, corpus_raw: List[str] = tot_corpus_raw, n_features = 17,
                    token_pattern_ = r"(?u)\b\w\w+\b",
                    print_out = True):
        big_coll = {}
        collected = []
        query_ = query.split('+') if isinstance(query, str) else query

        for q in query_:
            if print_out:
                print(q)
            var_coll, data_coll = QrDocInstance.collect_data(n_features=n_features, query=q, token_pattern_=token_pattern_)
            doc_id, out_out = QrDocInstance.print_whats_going_on([var_coll, data_coll], print_out = False)
            if out_out:
                tm_ = []
                for item in out_out:
                    if [item[2][0], item[2][1]] not in collected:
                        tm_.append(item)
                        collected.append([item[2][0], item[2][1]])
                big_coll[q] = [var_coll, data_coll, doc_id, tm_]
            else:
                print('No documents found for the keyword: ', q)
        #a = []
        #b = []
        #for k, v in big_coll.items():
        #    for i in v[3]:
        #        a.append(i[2])
        #        b.append(i[2])

        a = []
        b = []
        index_set = set()
        for _, v in big_coll.items():
            for i in v[2]:
                index_set.add(i)

        # sort the indices
        index_set = list(index_set)
        index_set.sort()
        for i in index_set:
            a.append(corpus[i])
            b.append(corpus_raw[i])

        return a,b, big_coll


    def run_llm_request(self, formatted_prompt: str) -> List[str]:

        with LMEngine(model=self.model_dict[self.model_name], lm_api=self.lm_api, kwarg=self.llm_kwarg) as req:
            out = req(formatted_prompt)
        return out


    @staticmethod
    async def fetch_kw(text, lm_api, keywords_prompt = keywords_prompt, kw_model_name: str = 'mistralai/mixtral-8x22b-instruct'):
        x_ = await a_quick_chat(query= keywords_prompt.format(text= text),
                        lm_api=lm_api,
                        model=kw_model_name, 
                        extras={'temperature': 0.0, 'max_tokens': 1500}
        )
        @dataclass
        class ans:
            keywords: list[str]
            abbreviations: list[str]
        try:
            x_1 = re.findall(r'```python(.*)```', x_, re.DOTALL)[-1]
            # find keywords and abbreviations
            kw_ = re.findall(r'extracted_information = (.*?)(?:$)', x_1, re.DOTALL)[0].strip()
            #print(kw_)
            extracted_keywords = ast.literal_eval(kw_)
            eab_ = re.findall(r'extracted_abbreviations = (.*?)(?:extracted_information)', x_1, re.DOTALL)[0].strip()
            #print(eab_)
            extracted_abbreviations = ast.literal_eval(eab_)
            # create a class to hold the keywords and abbreviations
            
            # create an instance of the class
            if not extracted_abbreviations:
                extracted_abbreviations = ['']
            x_2 = ans( keywords = extracted_keywords, abbreviations = extracted_abbreviations)
            return x_2
        except Exception as e:
            print(e)
            x_er = ans( keywords = [''], abbreviations = [''])
            return x_er

    async def keyword_time(self, user_doc_q):
        """ Extract keywords and abbreviations from the input text """
        self.user_doc_q = user_doc_q
        kws = await QrDocInstance.fetch_kw(self.user_doc_q, lm_api=self.lm_api)
        self.kws = kws
        print('Keywords: ', '\n'.join(kws.keywords) + '\n'.join(kws.abbreviations))
        kws2 = []
        ikw = kws.keywords

        for ii in ikw:
            # remove quantum and espresso, and calculation
            i = ii.strip()
            if re.match(r'[a-zA-Z-_]+', i) and i.lower() not in ['quantum', 'espresso', 'calculation']:
                kws2.append(i.strip())

        iab = kws.abbreviations
        
        for ii in iab:
            i = ii.strip()
            if 'none' not in i.lower() and 'qe' not in i.lower() and re.match(r'[a-zA-Z-_]+', i):
                kws2.append(i.strip())  

        kws2 = [i for i in kws2 if i != '']
        kws3 = ', '.join(kws2)
        print(f"Your Query: {self.user_doc_q}\nThe extracted keywords for your query are: {kws3}\nIf you are satisfied with the keywords, proceed to the next step, and call the 'doc' method")
        self.kws3 = kws2

    async def doc(self):
        """Collect and trim documents to fit within token limit."""
        # Get initial documents
        A, B, BC = await QrDocInstance.a_b_from_query(query=self.kws3, print_out=False)
        
        if not A:
            print('No documents found for the keywords, please try again')
            return None
        
        # Store full collection
        self.coll_docs = A
        self.coll_docs_raw = B
        self.raw_collection = BC
        
        if self.MAXDocToken > 15000:
            self.curated_docs = A
            self.curated_docs_raw = B
            return
        # Check if trimming is needed
        total_tokens = count_tokens('\n'.join(A))
        if total_tokens <= self.MAXDocToken:
            self.curated_docs = A
            self.curated_docs_raw = B
            return
        
        # Calculate initial sample size based on token ratio
        sample_size = max(1, int((self.MAXDocToken / total_tokens) * len(A)))
        
        # Binary search to find optimal sample size
        left, right = 1, len(A)
        optimal_docs = A  # fallback to full collection
        
        while left <= right:
            mid = (left + right) // 2
            np.random.seed(42)  # For reproducibility
            sampled_docs_indx = np.random.choice(range(len(A)), size=mid, replace=False).tolist()
            sampled_docs = [A[i] for i in sampled_docs_indx]
            tokens = count_tokens('\n'.join(sampled_docs))
            
            if tokens <= self.MAXDocToken:
                optimal_docs = sampled_docs
                left = mid + 1  # Try to fit more documents
            else:
                right = mid - 1  # Need fewer documents
        
        self.curated_docs = optimal_docs
        self.curated_docs_raw = [B[i] for i in sampled_docs_indx]
        print(f"the total token count is {tokens}")
        print('You can call the llm_call method')
        

    def llm_call(self , n_kwarg = False):
        out = {}
        if n_kwarg:
            self.llm_kwarg = self.get_kwarg()

        X = self.run_llm_request(self.doc_ans_formatted)[0]    

        try:
            if X is not None:
                out[self.model_name] = X

            ty = re.sub(r'(?<=\n)\w+', r'* \g<0>', out[self.model_name])
            tot_markdown = self.markdown_text.format(text='\n\n'+ty)
            self.markdown = tot_markdown
            display(Markdown(tot_markdown))
            self.llm_call_output = X[0]
        except Exception as e:
            print(e)


    def llm_call_instructed(self, instructions:list[str], n_kwarg:bool = False):
        
        doc_ans_formatted = self.doc_ans.format(documents='\n'.join(self.curated_docs), question=self.question)
        self.doc_ans_formatted = doc_ans_formatted
        print(f"the total token count is {count_tokens(doc_ans_formatted)}")

        if n_kwarg:
            self.llm_kwarg = self.get_kwarg()

        prompt_answer = self.doc_ans_instruct
        pa_ = prompt_answer.format(instructions=instructions)

        prompt_answer_template = self.run_llm_request(pa_)[0]

        assertion_criteria = re.findall(r'{(.*?)}', prompt_answer_template, re.DOTALL)

        try: 
            assert (set(assertion_criteria) == set(('documents', 'question'))) and (len(assertion_criteria) == 2), "The prompt template is not correct"
        except Exception as e:
            print(e)
            self.llm_call_instructed(instructions, n_kwarg)

        try:
            pr_inst = prompt_answer_template.format(documents='\n'.join(self.curated_docs), question=self.question)
            out_ = self.run_llm_request(pr_inst)[0]
            ty = re.sub(r'(?<=\n)\w+', r'* \g<0>', out_)
            tot_markdown = self.markdown_text.format(text='\n\n'+ty)
            self.markdown = tot_markdown
            self.llm_call_output = out_
            display(Markdown(tot_markdown))

        except Exception as e:
            print(e)
            self.llm_call_instructed(instructions, n_kwarg = self.llm_kwarg_call())

    @staticmethod
    def dict_to_string(input_dict: Union[Dict, List[Dict]], indent=0, indent_step=2) -> str:

        result = []
        for key, value in input_dict.items():
            # Create the indentation
            space = ' ' * (indent * indent_step)
            key = str(key).title()
            
            if isinstance(value, dict):
                # Recursively format nested dictionaries
                result.append(f"{space}{key}:\n{{")
                result.append(QrDocInstance.dict_to_string(value, indent + 1, indent_step))
                result.append(f"{space}}}")
            elif isinstance(value, list):
                # Format lists
                result.append(f"{space}{key}: [")
                for item in value:
                    result.append(f"{space}  {item},")
                result.append(f"{space}]")
            elif isinstance(value, str):
                # Format strings
                result.append(f"{space}{key}:\n\"{value}\"")
            elif isinstance(value, (int, float)):
                # Format numbers
                result.append(f"{space}{key}:\n{value}")
            elif value is None:
                # Format None
                result.append(f"{space}{key}: None")
            else:
                # Format other types
                result.append(f"{space}{key}:\n{value}")
        
        return '\n'.join(result)


#
#
#class ClassTemplate:
#    @classmethod
#    def create_class(cls, class_name: str, attributes: Dict[str, Type[Any]], methods: Dict[str, Callable] = None) -> Type:
#        
#        def __init__(self, **kwargs):
#            for attr, expected_type in attributes.items():
#                value = kwargs.get(attr)
#                if value is not None and not isinstance(value, expected_type):
#                    raise TypeError(f"Expected type for {attr} is {expected_type.__name__}, but got {type(value).__name__}.")
#                setattr(self, attr, value)
#
#        def __repr__(self):
#            attr_repr = ', '.join(f'{k}={v!r}' for k, v in self.__dict__.items())
#            return f"{class_name}({attr_repr})"
#
#        # Construct the class dictionary with __init__, __repr__, and type annotations
#        class_dict = {
#            "__init__": __init__,
#            "__repr__": __repr__,
#            "__annotations__": attributes
#        }
#
#        # Add any additional methods provided
#        if methods:
#            class_dict.update(methods)
#
#        # Dynamically create and return the class
#        return type(class_name, (object,), class_dict)
