import os
import argparse
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from minioset import (
    connect_server,
    load_object,
    save_object,
    compress_object,
    uncompress_object,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--bucket", help="bucket name", default="petcharts")
    parser.add_argument("-c", "--corpusdata", help="corpus file", default="pet_wiki.txt")
    parser.add_argument(
        "-k", "--tokenizer", help="tokenizer zip file", default="tokenizer.zip"
    )
    parser.add_argument("-v", "--vocabsize", help="vocabsize", type=int, default=40000)
    parser.add_argument("-H", "--host", help="object server")
    parser.add_argument("-A", "--accesskey", help="access key")
    parser.add_argument("-K", "--secretkey", help="secret key")
    args = parser.parse_args()

    try:
        client = connect_server(args.host, args.accesskey, args.secretkey)
        load_object(client, args.bucket, args.corpusdata)
#    except Exception as e:
#	print('error', e)
    except :
        pass

    os.makedirs("./pretrained", exist_ok=True)

    paths = [str(x) for x in Path(".").glob("**/{}".format(args.corpusdata))]

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=paths,
        vocab_size=args.vocabsize,
        min_frequency=50,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    )
    tokenizer.save_model("./pretrained")

    compress_object(args.tokenizer, "./pretrained")

    try:
        save_object(client, args.bucket, args.tokenizer)
    except:
        pass
