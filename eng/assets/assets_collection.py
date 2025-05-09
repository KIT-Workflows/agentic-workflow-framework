import json, os, yaml

current_path = os.path.dirname(os.path.abspath(__file__))
current_path = os.path.dirname(current_path)
#yml_path = os.path.join(current_path, 'SET_ME_UP.yaml')
#
#with open(yml_path) as file:
#    settings = yaml.load(file, Loader=yaml.FullLoader)

model_dict = {
    'dbrx': 'mistralai/mixtral-8x22b-instruct',
    'meta70o': 'meta-llama/llama-3.3-70b-instruct',
    'meta405o': "meta-llama/llama-3.1-405b-instruct",
    'metaviso': 'meta-llama/llama-3.2-90b-vision-instruct',
    'nous405o': 'nousresearch/hermes-3-llama-3.1-405b',
    'perpl405o': 'perplexity/llama-3.1-sonar-huge-128k-online',
    'perplo': 'perplexity/llama-3.1-sonar-large-128k-online',
    'mistral': 'mistralai/mixtral-8x22b-instruct',
    'referee': 'anthropic/claude-3.5-sonnet'
    }

#LM_API = settings['lm_api']
#MP_API = settings['mp_api']

current_path = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.dirname(current_path)
#settings['MAIN_DIR'] = main_dir

#with open(yml_path, 'w') as file:
#    yaml.dump(settings, file)

pp_json_path = os.path.join(current_path, 'SSSP_1.3.0_PBE_efficiency.json')
mc3d_db_path = os.path.join(current_path, 'mc3d.json')
mc2d_db_path = os.path.join(current_path, 'mc2d.json')
#json_docs_path = os.path.join(current_path, 'assets/coll_docs_TOT3.json')
sg_dict_path = os.path.join(current_path, 'sg_dict.json')

PERC1 = 30
PERC3 = 70

LEARNING_RATE = 0.1
TRAINING_ITERATIONS = 500

default_llm_kwarg = {'n': 1, 'stream': False,
                    'top_k': 5000, 'top_p': 0.8, 'temperature': 0.7, 'max_tokens': 1500}

default_llm_kwarg_qe_gen = {'n': 1, 'stream': False,
                    'top_k': 1000, 'top_p': 0.7, 'temperature': 0.0, 'max_tokens': 5000}


#with open(json_docs_path, 'r') as file:
#    json_docs = json.load(file)

with open(sg_dict_path, 'r') as f:
    sg_dict = json.load(f)

with open(mc3d_db_path, 'r') as f:
    mc3d_db = json.load(f)['data']

with open(mc2d_db_path, 'r') as f:
    mc2d_db = json.load(f)['data']
