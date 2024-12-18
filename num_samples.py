from strhub.data.dataset import LmdbDataset
import argparse
import yaml
import os


parser = argparse.ArgumentParser(description="take lmdb path")
parser.add_argument("--lmdb_root", type=str, help="path to the LMDB root directory")
parser.add_argument("--language", type=str, help="language charset to use")

args = parser.parse_args()
lmdb_root = args.lmdb_root
char_root = os.path.join("/ssd_scratch/cvit/lalitha/code/configs/charset/",args.language+".yaml")
with open(char_root, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)
charset= config['model']['charset_train']
# charset = "0123456789\"',.:;?-_!#$%&()*+/<=>@[]^{|}~\\।॥ਁਂਃਅਆਇਈਉਊਏਐਓਔਕਖਗਘਙਚਛਜਝਞਟਠਡਢਣਤਥਦਧਨਪਫਬਭਮਯਰਲਲ਼ਵਸ਼ਸਹ਼ਾਿੀੁੂੇੈੋੌ੍ੑਖ਼ਗ਼ਜ਼ੜਫ਼੦੧੨੩੪੫੬੭੮੯ੰੱੲੳੴੵ "

max_label_len = 35  # specify the maximum label length allowed
min_image_dim = 0
remove_whitespace = False  # specify if you want to remove whitespace from labels
normalize_unicode = False  # specify if you want to normalize Unicode characters
unlabelled = False  # specify if your dataset is unlabelled

# Create an instance of the LmdbDataset class
lmdb_dataset = LmdbDataset(lmdb_root, charset, max_label_len, min_image_dim,
                           remove_whitespace, normalize_unicode, unlabelled)

# Get the number of samples
num_samples = len(lmdb_dataset)
print("Number of samples:", num_samples)

